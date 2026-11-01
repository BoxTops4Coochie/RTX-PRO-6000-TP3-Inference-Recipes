from functools import cache
from typing import TYPE_CHECKING

import torch
from pydantic import BaseModel, ConfigDict
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import get_tp_group
from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, tl_rand32

from brrr import cuext

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

BLOCK = 2048
DRAFT_BLOCK = 128
RANK_CHUNK = 64
DRAFT_WORDS = tl.constexpr(5)
STAT_WORDS = tl.constexpr(3)


class Shard(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    logits: torch.Tensor
    token_logit: torch.Tensor
    lse: torch.Tensor
    start: int
    local: int
    blocks: int
    counts: torch.Tensor
    partials: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    ranks: int


SHARDS: dict[int, Shard] = {}


def words(t: torch.Tensor) -> torch.Tensor:
    return get_tp_group().device_communicator.phased_ar_comm.gather_words(t)


def shard_drafts(spec: "DSparkSpeculator", model: torch.nn.Module) -> Shard | None:
    head, markov = getattr(model, "lm_head", None), model.model.markov_head
    w2 = markov.markov_w2
    si = getattr(head, "shard_indices", None)
    processor = model.logits_processor
    ranks = cuext.extension("phased_all_reduce").RANKS
    ok = (
        spec.draft_logits is not None
        and get_tensor_model_parallel_world_size() == ranks
        and si is not None
        and markov.shard_across_tp
        and isinstance(markov.markov_w1, torch.nn.Embedding)
        and getattr(w2, "shard_indices", None) == si
        and w2.weight.dtype == markov.markov_w1.weight.dtype
        and w2.weight.shape == (si.num_org_elements, markov.markov_w1.weight.shape[1])
        and w2.weight.is_contiguous()
        and si.num_org_vocab_padding == 0
        and si.num_added_elements == 0
        and si.num_org_elements * ranks == spec.vocab_size
        and model.draft_id_to_target_id is None
        and spec.draft_watermarker is None
        and spec.acceptance_estimator is None
        and not spec.use_acceptance_estimator
        and spec._draft_topk is None
        and not spec.use_fp64_gumbel
        and processor.soft_cap is None
        and processor.scale == 1.0
    )
    if not ok:
        return None
    local = si.num_org_elements
    blocks = triton.next_power_of_2(triton.cdiv(local, DRAFT_BLOCK))
    rows = spec.max_num_reqs * spec.num_query_per_req
    device = spec.draft_logits.device
    slots = (spec.max_num_reqs + 1, spec.num_speculative_steps)
    weight = w2.weight.float()
    w2_scale = weight.abs().amax(dim=1).clamp_min(1e-12) / torch.finfo(torch.float8_e4m3fn).max
    spec.draft_logits = torch.zeros((spec.max_num_reqs, spec.num_speculative_steps, local), dtype=spec.draft_logits.dtype, device=device)
    shard = Shard(
        logits=spec.draft_logits,
        token_logit=torch.zeros(slots, dtype=torch.float32, device=device),
        lse=torch.zeros(slots, dtype=torch.float32, device=device),
        start=si.org_vocab_start_index,
        local=local,
        blocks=blocks,
        counts=torch.zeros(rows, dtype=torch.int32, device=device),
        partials=torch.empty((rows, blocks, DRAFT_WORDS), dtype=torch.float32, device=device),
        w2=(weight / w2_scale[:, None]).to(torch.float8_e4m3fn),
        w2_scale=w2_scale,
        ranks=ranks,
    )
    SHARDS[shard.logits.data_ptr()] = shard
    return shard


def full_draft_logits(draft_logits: torch.Tensor | None) -> torch.Tensor | None:
    return draft_logits if draft_logits is None or draft_logits.data_ptr() not in SHARDS else tensor_model_parallel_all_gather(draft_logits, dim=-1)


@triton.jit
def _last_program(counts_ptr, row, NB: tl.constexpr):
    tl.debug_barrier()
    last = tl.atomic_add(counts_ptr + row, 1, sem="acq_rel", scope="gpu") == NB - 1
    if last:
        tl.store(counts_ptr + row, 0)
    return last


@triton.jit
def _block_stats(x):
    m = tl.max(x, axis=0)
    return m, tl.where(m > float("-inf"), tl.sum(tl.exp(x - m), axis=0), 0.0)


@triton.jit
def _merge(m, s):
    top = tl.max(m, axis=0)
    return top, tl.sum(tl.where(m > float("-inf"), s * tl.exp(m - top), 0.0), axis=0)


@triton.jit
def _logsumexp(m, s):
    top, total = _merge(m, s)
    return top + tl.log(total)


@triton.jit
def _pick(v, lanes, i):
    return tl.sum(tl.where(lanes == i, v, 0), axis=0)


@triton.jit
def _draft_kernel(
    out_ptr,
    part_ptr,
    counts_ptr,
    base_ptr,
    base_stride,
    emb_ptr,
    w2_ptr,
    w2_scale_ptr,
    cache_ptr,
    cache_stride_0,
    cache_stride_1,
    col,
    idx_ptr,
    idx_stride,
    pos_ptr,
    pos_stride,
    seeds_ptr,
    temp_ptr,
    local_vocab,
    vocab_start,
    RANK: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
    NB: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    lanes = tl.arange(0, BLOCK)
    offs = b * BLOCK + lanes
    mask = offs < local_vocab
    bias = tl.zeros((BLOCK,), dtype=tl.float32)
    for k0 in tl.static_range(0, RANK, CHUNK):
        k = k0 + tl.arange(0, CHUNK)
        w = tl.load(w2_ptr + offs[:, None] * RANK + k[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        bias += tl.sum(w * tl.load(emb_ptr + row * RANK + k).to(tl.float32)[None, :], axis=1)
    bias *= tl.load(w2_scale_ptr + offs, mask=mask, other=0.0)
    logits = tl.load(base_ptr + row * base_stride + offs, mask=mask, other=float("-inf")).to(tl.float32) + bias
    req = tl.load(idx_ptr + row * idx_stride).to(tl.int64)
    valid = req >= 0
    temp = tl.load(temp_ptr + req, mask=valid, other=0.0).to(tl.float32)
    tl.store(cache_ptr + req * cache_stride_0 + col * cache_stride_1 + offs, logits, mask=mask & valid)
    seed = tl.load(seeds_ptr + req, mask=valid, other=0)
    pos = tl.load(pos_ptr + row * pos_stride) - 1
    value, idx = gumbel_noised_argmax(logits, vocab_start + offs, mask, seed, pos, temp, IS_DRAFTING=True, USE_FP64=False, APPLY_TEMPERATURE=True)
    t = tl.where(temp > 0.0, temp, 1.0)
    m, s = _block_stats(tl.where(mask, logits / t, float("-inf")))
    p = part_ptr + (row * NB + b) * DRAFT_WORDS
    tl.store(p, value)
    tl.store(p + 1, (vocab_start + b * BLOCK + idx).to(tl.float32))
    tl.store(p + 2, _pick(logits, lanes, idx))
    tl.store(p + 3, m)
    tl.store(p + 4, s)
    if _last_program(counts_ptr, row, NB):
        blocks = tl.arange(0, NB)
        q = part_ptr + (row * NB + blocks) * DRAFT_WORDS
        vals = tl.load(q, volatile=True)
        win = tl.argmax(vals, axis=0)
        o = out_ptr + row * DRAFT_WORDS
        tl.store(o, tl.max(vals, axis=0))
        tl.store(o + 1, _pick(tl.load(q + 1, volatile=True), blocks, win))
        tl.store(o + 2, _pick(tl.load(q + 2, volatile=True), blocks, win))
        top, total = _merge(tl.load(q + 3, volatile=True), tl.load(q + 4, volatile=True))
        tl.store(o + 3, top)
        tl.store(o + 4, total)


@triton.jit
def _draft_combine_kernel(
    gathered_ptr,
    rows,
    tokens_ptr,
    tokens_stride,
    col,
    emb_ptr,
    w1_ptr,
    idx_ptr,
    idx_stride,
    token_logit_ptr,
    lse_ptr,
    vp_stride,
    sink,
    TP: tl.constexpr,
    RANK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    ranks = tl.arange(0, TP)
    q = gathered_ptr + (ranks * rows + row) * DRAFT_WORDS
    win = tl.argmax(tl.load(q), axis=0)
    token = _pick(tl.load(q + 1), ranks, win).to(tl.int64)
    req = tl.load(idx_ptr + row * idx_stride).to(tl.int64)
    slot = tl.where(req >= 0, req, sink)
    tl.store(tokens_ptr + row * tokens_stride + col, token)
    tl.store(token_logit_ptr + slot * vp_stride + col, _pick(tl.load(q + 2), ranks, win))
    tl.store(lse_ptr + slot * vp_stride + col, _logsumexp(tl.load(q + 3), tl.load(q + 4)))
    r = tl.arange(0, RANK)
    tl.store(emb_ptr + row * RANK + r, tl.load(w1_ptr + token * RANK + r))


def draft(spec: "DSparkSpeculator", num_reqs: int, head_hidden: torch.Tensor) -> None:
    shard: Shard = spec.vp
    model = spec.model
    head, markov = model.lm_head, model.model.markov_head
    w1 = markov.markov_w1.weight
    steps = spec.num_speculative_steps
    num_sample = num_reqs * steps
    sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
    base = head.quant_method.apply(head, model.model.norm(sample_hidden)).view(num_reqs, steps, -1)
    idx_map = spec.sample_idx_mapping[:num_sample].view(num_reqs, steps)
    sample_pos = spec.sample_pos[:num_sample].view(num_reqs, steps)
    emb = markov.embed(spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]])
    rank = w1.shape[1]
    embs: list[torch.Tensor] = []
    for i in range(steps):
        embs.append(emb)
        out = torch.empty((num_reqs, DRAFT_WORDS), dtype=torch.float32, device=emb.device)
        _draft_kernel[(num_reqs, shard.blocks)](
            out,
            shard.partials,
            shard.counts,
            base[:, i],
            base.stride(0),
            emb,
            shard.w2,
            shard.w2_scale,
            shard.logits,
            shard.logits.stride(0),
            shard.logits.stride(1),
            i,
            idx_map[:, i],
            idx_map.stride(0),
            sample_pos[:, i],
            sample_pos.stride(0),
            spec.seeds,
            spec.temperature,
            shard.local,
            shard.start,
            RANK=rank,
            CHUNK=RANK_CHUNK,
            BLOCK=DRAFT_BLOCK,
            NB=shard.blocks,
            num_warps=4,
        )
        gathered = words(out)
        emb = torch.empty((num_reqs, rank), dtype=w1.dtype, device=emb.device)
        _draft_combine_kernel[(num_reqs,)](
            gathered,
            num_reqs,
            spec.draft_tokens,
            spec.draft_tokens.stride(0),
            i,
            emb,
            w1,
            idx_map[:, i],
            idx_map.stride(0),
            shard.token_logit,
            shard.lse,
            shard.lse.stride(0),
            spec.max_num_reqs,
            TP=shard.ranks,
            RANK=rank,
            num_warps=1,
        )
    if spec.use_confidence_head:
        confidence = model.compute_confidence(sample_hidden, torch.stack(embs, dim=1).flatten(0, 1))
        spec.draft_token_confidence_probs[:num_reqs] = confidence.view(num_reqs, steps)


@triton.jit
def _ratio_terms(g1_ptr, num_logits, vp_logit_ptr, vp_lse_ptr, vp_stride, req, start, count, temp, TP: tl.constexpr, S: tl.constexpr):
    lanes = tl.arange(0, S)
    live = lanes < count
    q = g1_ptr + (tl.arange(0, TP)[:, None] * num_logits + start + lanes[None, :]) * STAT_WORDS
    m = tl.load(q, mask=live[None, :], other=float("-inf"))
    s = tl.load(q + 1, mask=live[None, :], other=0.0)
    lse = _logsumexp(m, s)
    owned = tl.max(tl.load(q + 2, mask=live[None, :], other=float("-inf")), axis=0)
    d_lp = tl.load(vp_logit_ptr + req * vp_stride + lanes, mask=live, other=0.0) / temp - tl.load(vp_lse_ptr + req * vp_stride + lanes, mask=live, other=0.0)
    return owned - lse - d_lp, m, s


@triton.jit
def _target_stats_kernel(
    out_ptr,
    part_ptr,
    counts_ptr,
    logits_ptr,
    logits_stride,
    draft_sampled_ptr,
    num_logits,
    expanded_idx_ptr,
    temp_ptr,
    local_vocab,
    vocab_start,
    BLOCK: tl.constexpr,
    NB: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    mask = offs < local_vocab
    temp = tl.load(temp_ptr + tl.load(expanded_idx_ptr + row)).to(tl.float32)
    t = tl.where(temp > 0.0, temp, 1.0)
    x = tl.load(logits_ptr + row * logits_stride + offs, mask=mask, other=float("-inf")).to(tl.float32) / t
    if temp == 0.0:
        m, i = tl.max(x, axis=0, return_indices=True)
        s = (vocab_start + b * BLOCK + i).to(tl.float32)
    else:
        m, s = _block_stats(x)
    d = tl.load(draft_sampled_ptr + row + 1, mask=row + 1 < num_logits, other=-1) - vocab_start - b * BLOCK
    owned = tl.load(logits_ptr + row * logits_stride + b * BLOCK + d, mask=(d >= 0) & (d < BLOCK) & (b * BLOCK + d < local_vocab), other=float("-inf")).to(tl.float32) / t
    p = part_ptr + (row * NB + b) * STAT_WORDS
    tl.store(p, m)
    tl.store(p + 1, s)
    tl.store(p + 2, owned)
    if _last_program(counts_ptr, row, NB):
        blocks = tl.arange(0, NB)
        q = part_ptr + (row * NB + blocks) * STAT_WORDS
        ms = tl.load(q, volatile=True)
        ss = tl.load(q + 1, volatile=True)
        top, total = _merge(ms, ss)
        o = out_ptr + row * STAT_WORDS
        tl.store(o, top)
        if temp == 0.0:
            tl.store(o + 1, _pick(ss, blocks, tl.argmax(ms, axis=0)))
        else:
            tl.store(o + 1, total)
        tl.store(o + 2, tl.max(tl.load(q + 2, volatile=True), axis=0))


@triton.jit
def _target_residual_kernel(
    out_ptr,
    part_ptr,
    counts_ptr,
    g1_ptr,
    logits_ptr,
    logits_stride,
    cache_ptr,
    cache_stride_0,
    cache_stride_1,
    vp_logit_ptr,
    vp_lse_ptr,
    vp_stride,
    draft_sampled_ptr,
    num_logits,
    expanded_idx_ptr,
    local_pos_ptr,
    pos_ptr,
    temp_ptr,
    seeds_ptr,
    local_vocab,
    vocab_start,
    num_spec,
    BLOCK: tl.constexpr,
    NB: tl.constexpr,
    TP: tl.constexpr,
    S: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    mask = offs < local_vocab
    req = tl.load(expanded_idx_ptr + row).to(tl.int64)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    seed = tl.load(seeds_ptr + req)
    step = tl.load(local_pos_ptr + row).to(tl.int64)
    start = row - step
    is_bonus = (row + 1 >= num_logits) | (tl.load(local_pos_ptr + row + 1, mask=row + 1 < num_logits, other=0) == 0)
    rejected = tl.load(draft_sampled_ptr + row + 1, mask=row + 1 < num_logits, other=-1)
    lanes = tl.arange(0, S)
    terms, _, _ = _ratio_terms(g1_ptr, num_logits, vp_logit_ptr, vp_lse_ptr, vp_stride, req, start, step, temp, TP, S)
    drafted = tl.load(draft_sampled_ptr + start + lanes + 1, mask=lanes < step, other=-1)
    log_p = 0.0
    for s in tl.static_range(S):
        take = (temp != 0.0) & (s < step) & (_pick(drafted, lanes, s) >= 0)
        log_p = tl.where(take, tl.minimum(log_p + _pick(terms, lanes, s), 0.0), log_p)
    x = tl.load(logits_ptr + row * logits_stride + offs, mask=mask, other=float("-inf")).to(tl.float32) / tl.where(temp > 0.0, temp, 1.0)
    stats = g1_ptr + (tl.arange(0, TP) * num_logits + row) * STAT_WORDS
    lse = _logsumexp(tl.load(stats), tl.load(stats + 1))
    use_draft = (temp != 0.0) & (is_bonus == 0) & (rejected >= 0)
    t_lp = x - lse
    d_lp = (
        tl.load(cache_ptr + req * cache_stride_0 + step * cache_stride_1 + offs, mask=mask & use_draft, other=float("-inf")).to(tl.float32) / temp
        - tl.load(vp_lse_ptr + req * vp_stride + step, mask=use_draft, other=0.0)
    )
    mass = 0.0
    if use_draft & (step >= 1) & (step < num_spec):
        mass = tl.sum(tl.maximum(tl.exp(log_p) * tl.exp(t_lp) - tl.exp(d_lp), 0.0), axis=0)
    value = float("-inf")
    token = vocab_start + b * BLOCK
    if (temp != 0.0) | is_bonus:
        if use_draft:
            shifted = t_lp + log_p
            ratio = tl.exp(d_lp - shifted)
            residual = tl.where(mask & (ratio < 1.0), shifted + tldevice.log1p(-ratio), float("-inf")).to(tl.float32)
        else:
            residual = x
        value, i = gumbel_noised_argmax(residual, vocab_start + offs, mask, seed, tl.load(pos_ptr + row), temp, IS_DRAFTING=False, USE_FP64=False, APPLY_TEMPERATURE=False)
        token = vocab_start + b * BLOCK + i
    p = part_ptr + (row * NB + b) * STAT_WORDS
    tl.store(p, mass)
    tl.store(p + 1, value)
    tl.store(p + 2, token.to(tl.float32))
    if _last_program(counts_ptr, row, NB):
        blocks = tl.arange(0, NB)
        q = part_ptr + (row * NB + blocks) * STAT_WORDS
        vals = tl.load(q + 1, volatile=True)
        vals = tl.where(vals != vals, float("-inf"), vals)
        o = out_ptr + row * STAT_WORDS
        tl.store(o, tl.sum(tl.load(q, volatile=True), axis=0))
        tl.store(o + 1, tl.max(vals, axis=0))
        tl.store(o + 2, _pick(tl.load(q + 2, volatile=True), blocks, tl.argmax(vals, axis=0)))


@triton.jit
def _target_reject_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    g1_ptr,
    g2_ptr,
    vp_logit_ptr,
    vp_lse_ptr,
    vp_stride,
    draft_sampled_ptr,
    num_logits,
    cu_ptr,
    idx_ptr,
    pos_ptr,
    temp_ptr,
    seeds_ptr,
    TP: tl.constexpr,
    S: tl.constexpr,
):
    r = tl.program_id(0)
    req = tl.load(idx_ptr + r).to(tl.int64)
    start = tl.load(cu_ptr + r).to(tl.int64)
    end = tl.load(cu_ptr + r + 1).to(tl.int64)
    num_draft = end - start - 1
    seed = tl.load(seeds_ptr + req)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    greedy = temp == 0.0
    lanes = tl.arange(0, S)
    ranks = tl.arange(0, TP)
    live = lanes < num_draft
    drafted = tl.load(draft_sampled_ptr + start + lanes + 1, mask=live, other=-1).to(tl.int64)
    following = tl.load(draft_sampled_ptr + start + lanes + 2, mask=lanes < num_draft - 1, other=-1)
    u = tl_rand32(seed, tl.load(pos_ptr + start + lanes, mask=live, other=0), includes_zero=False)
    terms, m, s = _ratio_terms(g1_ptr, num_logits, vp_logit_ptr, vp_lse_ptr, vp_stride, req, start, num_draft, temp, TP, S)
    best = _pick(s, ranks[:, None], tl.argmax(m, axis=0)[None, :]).to(tl.int64)
    mass = tl.sum(tl.load(g2_ptr + (ranks[:, None] * num_logits + start + lanes[None, :] + 1) * STAT_WORDS, mask=(lanes < num_draft - 1)[None, :], other=0.0), axis=0)
    accepted = tl.zeros((), tl.int64)
    verifying = num_draft >= 0
    log_p = tl.zeros((), tl.float32)
    for i in tl.static_range(S):
        if i < num_draft:
            raw = _pick(drafted, lanes, i)
            valid = raw >= 0
            token = tl.maximum(raw, 0)
            if greedy:
                if verifying:
                    top = _pick(best, lanes, i)
                    ok = (top == token) & valid
                    verifying = ok
                    accepted += ok
                    tl.store(sampled_ptr + r * sampled_stride + i, tl.where(ok, token, top))
            else:
                if valid:
                    log_p = tl.minimum(log_p + _pick(terms, lanes, i), 0.0)
                verifying &= valid
                if verifying:
                    prefix = tl.exp(log_p)
                    if _pick(following, lanes, i) >= 0:
                        weight = _pick(mass, lanes, i)
                        denom = weight + 1.0 - prefix
                        h = tl.where(denom > 0.0, weight / denom, 1.0)
                    else:
                        h = prefix
                    accepted = tl.where(_pick(u, lanes, i) <= h, i + 1, accepted)
                    tl.store(sampled_ptr + r * sampled_stride + i, token)
    tl.store(num_sampled_ptr + r, (accepted + 1).to(tl.int32))
    row = start + accepted
    if (temp != 0.0) | (row == end - 1):
        q = g2_ptr + (ranks * num_logits + row) * STAT_WORDS
        vals = tl.load(q + 1)
        vals = tl.where(vals != vals, float("-inf"), vals)
        token = _pick(tl.load(q + 2), ranks, tl.argmax(vals, axis=0)).to(tl.int64)
        tl.store(sampled_ptr + r * sampled_stride + accepted, token)


@triton.jit
def _mask_stops_kernel(
    logits_ptr,
    logits_stride,
    idx_ptr,
    pos_ptr,
    min_lens_ptr,
    counts_ptr,
    stops_ptr,
    stops_stride,
    vocab_start,
    local_vocab,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req = tl.load(idx_ptr + row).to(tl.int64)
    count = tl.load(counts_ptr + req)
    if (count > 0) & (tl.load(pos_ptr + row) + 1 < tl.load(min_lens_ptr + req)):
        lanes = tl.arange(0, BLOCK)
        local = tl.load(stops_ptr + req * stops_stride + lanes, mask=lanes < count, other=-1).to(tl.int64) - vocab_start
        tl.store(logits_ptr + row * logits_stride + local, float("-inf"), mask=(lanes < count) & (local >= 0) & (local < local_vocab))


def mask_stops(logits: torch.Tensor, input_batch: "InputBatch", bias: "LogitBiasState", shard: Shard) -> None:
    stops = bias.stop_token_ids.gpu
    _mask_stops_kernel[(logits.shape[0],)](
        logits,
        logits.stride(0),
        input_batch.expanded_idx_mapping,
        input_batch.positions[input_batch.logits_indices],
        bias.min_lens.gpu,
        bias.num_stop_token_ids.gpu,
        stops,
        stops.stride(0),
        shard.start,
        shard.local,
        BLOCK=triton.next_power_of_2(stops.shape[1]),
        num_warps=1,
    )


@triton.jit
def _stage_kernel(temp_out_ptr, seed_out_ptr, temp_ptr, seeds_ptr, n, N: tl.constexpr):
    i = tl.arange(0, N)
    live = i < n
    tl.store(temp_out_ptr + i, tl.load(temp_ptr + i, mask=live, other=0.0), mask=live)
    tl.store(seed_out_ptr + i, tl.load(seeds_ptr + i, mask=live, other=0), mask=live)


class Scratch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    counts: torch.Tensor
    partials: torch.Tensor


@cache
def scratch(device: torch.device, rows: int, blocks: int) -> Scratch:
    return Scratch(
        counts=torch.zeros(rows, dtype=torch.int32, device=device), partials=torch.empty((rows, blocks, STAT_WORDS), dtype=torch.float32, device=device)
    )


def verify(logits: torch.Tensor, input_batch: "InputBatch", spec: "DSparkSpeculator", temperature: torch.Tensor, seeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shard, num_spec = spec.vp, spec.num_speculative_steps
    n = temperature.shape[0]
    temp, seed = torch.empty(n, dtype=temperature.dtype, device=temperature.device), torch.empty(n, dtype=seeds.dtype, device=seeds.device)
    _stage_kernel[(1,)](temp, seed, temperature, seeds, n, N=triton.next_power_of_2(n), num_warps=1)
    num_logits, local = logits.shape
    num_reqs = input_batch.num_reqs
    blocks = triton.next_power_of_2(triton.cdiv(local, BLOCK))
    draft_sampled = input_batch.input_ids[input_batch.logits_indices]
    pos = input_batch.positions[input_batch.logits_indices]
    lanes = triton.next_power_of_2(num_spec + 1)
    work = scratch(logits.device, num_logits, blocks)
    stats = torch.empty((num_logits, STAT_WORDS), dtype=torch.float32, device=logits.device)
    _target_stats_kernel[(num_logits, blocks)](
        stats,
        work.partials,
        work.counts,
        logits,
        logits.stride(0),
        draft_sampled,
        num_logits,
        input_batch.expanded_idx_mapping,
        temp,
        local,
        shard.start,
        BLOCK=BLOCK,
        NB=blocks,
        num_warps=8,
    )
    g1 = words(stats)
    resid = torch.empty((num_logits, STAT_WORDS), dtype=torch.float32, device=logits.device)
    _target_residual_kernel[(num_logits, blocks)](
        resid,
        work.partials,
        work.counts,
        g1,
        logits,
        logits.stride(0),
        shard.logits,
        shard.logits.stride(0),
        shard.logits.stride(1),
        shard.token_logit,
        shard.lse,
        shard.lse.stride(0),
        draft_sampled,
        num_logits,
        input_batch.expanded_idx_mapping,
        input_batch.expanded_local_pos,
        pos,
        temp,
        seed,
        local,
        shard.start,
        num_spec,
        BLOCK=BLOCK,
        NB=blocks,
        TP=shard.ranks,
        S=lanes,
        num_warps=8,
    )
    g2 = words(resid)
    sampled = torch.empty((num_reqs, num_spec + 1), dtype=torch.int64, device=logits.device)
    num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=logits.device)
    _target_reject_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        g1,
        g2,
        shard.token_logit,
        shard.lse,
        shard.lse.stride(0),
        draft_sampled,
        num_logits,
        input_batch.cu_num_logits,
        input_batch.idx_mapping,
        pos,
        temp,
        seed,
        TP=shard.ranks,
        S=lanes,
        num_warps=1,
    )
    return sampled, num_sampled

