import json
from functools import cache
from pathlib import Path

import torch
import torch.distributed as dist
from b12x.moe.fused_moe._impl import _PreparedW4A8Weights
from pydantic import BaseModel
from safetensors import safe_open
from torch import nn
from torch.distributed import ProcessGroup
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.b12x import B12xExperts

from brrr import cuext

import os
PROLOGUE_MAXM = 8
# TP3 A/B: DS41_FUSED_LAYERS=N fuses layers [0, N); 0 disables the fused path (control arm), 40 = all.
LAYERS = int(os.environ.get("DS41_FUSED_LAYERS", "40"))
DUMP = os.environ.get("DS41_FUSED_DUMP", "")
DUMP_LAYERS = {int(x) for x in os.environ.get("DS41_FUSED_DUMP_LAYERS", "").split(",") if x.strip()}  # empty = all
# TP3 A/B: fuse only decode batches of at most DS41_FUSED_MAXM rows (fused c8 measured below native b12x); 0 = MAXM.
FUSED_MAXM = int(os.environ.get("DS41_FUSED_MAXM", "0"))
DENSE_NATIVE = os.environ.get("DS41_FUSED_DENSE_NATIVE", "1") == "1"  # kernel reads vLLM dense weights, no staged copies
CSF_INKERNEL = os.environ.get("DS41_FUSED_CSF", "1") == "1"  # read b12x CSF scale payloads inside the fused kernel
CSF_DEBUG_LAYER = int(os.environ.get("DS41_FUSED_CSF_DEBUG_LAYER", "-1"))  # mirror this layer's expanded scale words and verify them at runtime


@cache
def opened(path: Path) -> safe_open:
    return safe_open(path, framework="pt")


class Checkpoint(BaseModel, frozen=True):
    root: Path
    index: dict[str, str]

    def get(self, name: str, rows: slice = slice(None), cols: slice = slice(None)) -> torch.Tensor:
        tensor = opened(self.root / self.index[name]).get_slice(name)
        picked = tensor[rows, cols] if len(tensor.get_shape()) == 2 else tensor[rows]
        return picked.contiguous().view(torch.uint8)


def stages(w: torch.Tensor) -> torch.Tensor:
    w = w.contiguous().view(torch.uint8)
    w = torch.nn.functional.pad(w, (0, -w.shape[1] % 128))
    return w.reshape(w.shape[0] // 64, 4, 2, 8, -1, 4, 4, 8).permute(0, 4, 5, 1, 3, 6, 2, 7).contiguous().view(-1)


def interleave(w: torch.Tensor, block: int) -> torch.Tensor:
    return w.reshape(2, -1, block, *w.shape[1:]).transpose(0, 1).reshape(w.shape).contiguous()


def shard(rank: int, size: int) -> slice:
    return slice(rank * size, (rank + 1) * size)


def empty(device: torch.device, *shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.empty(*shape, dtype=dtype, device=device)


def mixes(device: torch.device, m: int, hc: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return empty(device, m, hc, dtype=torch.float32), empty(device, m, hc, hc, dtype=torch.float32), empty(device, m, hc, dtype=torch.float32)


class Engine:
    def __init__(self, model_dir: str, rank: int, cpu_group: ProcessGroup) -> None:
        self.ext = cuext.extension("decoder")
        self.rank = rank
        self.device = torch.device("cuda", torch.cuda.current_device())
        root = Path(model_dir)
        self.checkpoint = Checkpoint(root=root, index=json.loads((root / "model.safetensors.index.json").read_text())["weight_map"])
        self.handle = self.ext.create(cuext.exchange_ipc(self.ext, self.ext.flag_bytes() + self.ext.inbox_bytes(), rank, cpu_group), rank)
        dist.barrier(group=cpu_group)
        self.layers: dict[int, int] = {}
        self.pending_mix: torch.Tensor | None = None
        self.dumped: dict[int, int] = {}

    def raw(self, name: str, rows: slice = slice(None), cols: slice = slice(None)) -> torch.Tensor:
        return self.checkpoint.get(name, rows, cols).to(self.device)

    def raw_padded(self, name: str, rows: slice = slice(None), cols: slice = slice(None)) -> torch.Tensor:
        """TP3 port: a rank's shard may run past the checkpoint tensor (kraken pads 8 output groups to 9 and
        72 heads at TP3); read the in-range part and zero-fill the rest, like kraken's padded weight loader."""
        shape = opened(self.checkpoint.root / self.checkpoint.index[name]).get_slice(name).get_shape()
        r0, r1 = (rows.start or 0), (rows.stop if rows.stop is not None else shape[0])
        c0, c1 = (cols.start or 0), (cols.stop if cols.stop is not None else (shape[1] if len(shape) == 2 else 1))
        rr1, cc1 = min(r1, shape[0]), (min(c1, shape[1]) if len(shape) == 2 else c1)
        got = self.checkpoint.get(name, slice(r0, max(rr1, r0)), slice(c0, max(cc1, c0)) if len(shape) == 2 else slice(None)).to(self.device)
        if len(shape) == 2:
            out = torch.zeros((r1 - r0, c1 - c0), dtype=torch.uint8, device=self.device)
            out[: got.shape[0], : got.shape[1]] = got
        else:
            out = torch.zeros((r1 - r0,), dtype=torch.uint8, device=self.device)
            out[: got.shape[0]] = got
        return out

    def dump_csf(self, owner, rep, index: int) -> None:
        """TP3 debug: save a CSF layer's compact scale payload + geometry + fully decoded reference scales."""
        impl = owner._prepared()._impl
        dec = getattr(impl, "nvfp4_csf", None)
        info = {"layer": index, "rank": self.rank, "w13_layout": getattr(rep, "w13_layout", None), "n64_repack": rep.n64_repack,
                "w13_sfb_shape": tuple(rep.w13_sfb.shape), "w2_sfb_shape": tuple(rep.w2_sfb.shape), "csf": dec is not None}
        if dec is not None:
            for name, plane in (("first", dec.first), ("second", dec.second)):
                sc = plane.scales
                info[name] = {"geometry": tuple(int(g) for g in plane.geometry), "rows": sc.rows, "columns": sc.columns,
                              "num_experts": sc.num_experts, "quantization": sc.quantization, "native_codes": sc.native_codes,
                              "group_record_bytes": sc.group_record_bytes, "destination": sc.destination, "sections": sc.sections,
                              "storage_bytes": sc.storage.numel(), "plane_rows": plane.rows, "plane_columns": plane.columns,
                              "codec": plane.codec, "layout": plane.layout}
                info[name]["storage"] = sc.storage.detach().cpu()
            dec.decode_all(impl.w1_blockscale, impl.w2_blockscale)
            torch.cuda.synchronize()
            info["ref_w13_sfb"] = rep.w13_sfb.detach().cpu().clone()
            info["ref_w2_sfb"] = rep.w2_sfb.detach().cpu().clone()
        torch.save(info, f"{DUMP}/csf{index}_rank{self.rank}.pt")

    @staticmethod
    def csf_args(scales) -> dict:
        """TP3 CSF: describe a b12x CsfBlockPayload (destination 2, mxfp4, 3-byte group records) for the fused kernel and
        build the per-(payload expert, group) record descriptors first | count << 24 (mirrors BlockStorageLayout)."""
        st = scales.storage
        R, C, Es = int(scales.rows), int(scales.columns), int(scales.num_experts)
        if not (scales.destination == 2 and scales.native_codes and scales.group_record_bytes == 3 and scales.quantization == "mxfp4"):
            raise RuntimeError(f"ds41 fused kernel needs CSF destination 2 / mxfp4 / 3-byte records; got {scales}")
        a16 = lambda v: (v + 15) // 16 * 16
        cg, groups = (C + 3) // 4, (R + 127) // 128 * ((C + 3) // 4)
        pages = (groups + 31) // 32
        prefixes, code_bytes = groups + pages, R * C // 8
        bases = 1024
        codes = a16(bases + Es * R)
        page_bases = a16(codes + Es * code_bytes)
        offsets = a16(page_bases + Es * pages * 4)
        recpos = a16(offsets + Es * prefixes * 2)
        value_off = int(st[:8].view(torch.int64).item())
        pb = st[page_bases:page_bases + Es * pages * 4].view(torch.int32).view(Es, pages)
        off = st[offsets:offsets + Es * prefixes * 2].view(torch.int16).view(Es, prefixes).to(torch.int32) & 0xFFFF
        sg = torch.arange(groups, device=st.device)
        page = sg // 32
        first = pb[:, page] + off[:, sg + page]
        count = pb[:, page] + off[:, sg + page + 1] - first
        if int(first.max()) >= 1 << 22 or int(count.max()) >= 1 << 10 or int(count.min()) < 0:
            raise RuntimeError(f"CSF record descriptor out of range: first<{int(first.max())} count<{int(count.max())}")
        if os.environ.get("BRRR_TP3_LOG") == "1":
            print(f"[brrr-tp3] csf plane R={R} C={C} Es={Es}: records/group max={int(count.max())} mean={float(count.float().mean()):.3f} "
                  f"records={(value_off - recpos) // 2} storage={st.numel()}", flush=True)
        gdesc = (first | (count << 22)).to(torch.int32).contiguous()  # first: 22 bits, count: 10 bits (<= 512 scales/group)
        return dict(storage=st, gdesc=gdesc, meta=[bases, codes, code_bytes, value_off, recpos, R, cg, groups, int(scales.sections)])

    def register(self, layer: nn.Module) -> int:
        index = layer.layer_index
        if index in self.layers:
            return self.layers[index]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"ds41 layer {index} weights must be prepared before CUDA graph capture")
        prefix, r, inter = f"layers.{index}", self.rank, self.ext.INTER
        ffn, shared = layer.ffn, layer.ffn.shared_experts
        inter_blocks = shard(r, inter // 32)
        owner: B12xExperts = next(m.b12x_experts for m in ffn.modules() if hasattr(m, "b12x_experts"))
        rep: _PreparedW4A8Weights = owner._prepared()._impl.representation_for("w4a8_mx")
        want_rp = bool(getattr(self.ext, "RP_LAYOUT", 0))
        if rep.n64_repack == want_rp or rep.intermediate_size != inter:
            raise RuntimeError(f"ds41 layer kernel needs the {'natural' if want_rp else 'n64'} W4A8 expert layout with intermediate {inter}; got n64_repack={rep.n64_repack}, intermediate {rep.intermediate_size}")
        bias_vl = getattr(ffn.gate, "bias_vl", None)
        if os.environ.get("BRRR_TP3_LOG") == "1":
            impl_ = owner._prepared()._impl
            print(f"[brrr-tp3] layer {index} rep: w13_layout={rep.w13_layout} n64={rep.n64_repack} src={rep.source_format} csf={getattr(impl_, 'nvfp4_csf', None) is not None} "
                  f"w13_rp={tuple(rep.w13_rp.shape)}/{rep.w13_rp.dtype} w2_rp={tuple(rep.w2_rp.shape)} sfb={tuple(rep.w13_sfb.shape)}/{tuple(rep.w2_sfb.shape)} "
                  f"plan_w13={getattr(getattr(getattr(impl_, 'plan', None), 'source', None), 'w13_layout', None)} inline={getattr(getattr(impl_, 'plan', None), 'csf_inline_scales', None)}", flush=True)
        csf = {}
        dec = getattr(owner._prepared()._impl, "nvfp4_csf", None)
        decoded = None
        if dec is not None and os.environ.get("DS41_FUSED_CSF_DECODED") in ("1", "2"):
            # TP3 debug: bypass the in-kernel CSF reader; give the kernel a private fully decoded copy of the scales.
            impl_ = owner._prepared()._impl
            dec.decode_all(impl_.w1_blockscale, impl_.w2_blockscale)
            decoded = (rep.w13_sfb.detach().clone().contiguous(), rep.w2_sfb.detach().clone().contiguous())
            print(f"[brrr-tp3] layer {index}: CSF layer registered with a private decoded scale copy", flush=True)
        if dec is not None and CSF_INKERNEL and os.environ.get("DS41_FUSED_CSF_DECODED") != "1":
            for tag, plane in (("csf13", dec.first), ("csf2", dec.second)):
                a = self.csf_args(plane.scales)
                csf[f"{tag}_storage"], csf[f"{tag}_gdesc"], csf[f"{tag}_meta"] = a["storage"], a["gdesc"], a["meta"]
        elif dec is not None and decoded is None:
            raise RuntimeError(f"ds41 layer {index} uses CSF scales but DS41_FUSED_CSF=0")
        if csf and os.environ.get("DS41_FUSED_CSF_SELFCHECK") == "1":
            impl_ = owner._prepared()._impl
            dec.decode_all(impl_.w1_blockscale, impl_.w2_blockscale)
            for tag, ref in (("csf13", rep.w13_sfb), ("csf2", rep.w2_sfb)):
                out = torch.empty_like(ref.contiguous().view(torch.int32).reshape(-1))
                self.ext.csf_expand_all(csf[f"{tag}_storage"], csf[f"{tag}_gdesc"], csf[f"{tag}_meta"], out)
                torch.cuda.synchronize()
                refw = ref.contiguous().view(torch.int32).reshape(-1)
                bad = out != refw
                msg = f"[brrr-tp3] csf selfcheck layer {index} {tag}: words={out.numel()} mismatching={int(bad.sum())}"
                if int(bad.sum()):
                    i0 = bad.nonzero().reshape(-1)[:4]
                    msg += f" first={i0.tolist()} got={[hex(int(v) & 0xffffffff) for v in out[i0]]} ref={[hex(int(v) & 0xffffffff) for v in refw[i0]]} storage_ptr={csf[f'{tag}_storage'].data_ptr():#x}"
                print(msg, flush=True)
        if csf and index == CSF_DEBUG_LAYER:
            dec.decode_all(owner._prepared()._impl.w1_blockscale, owner._prepared()._impl.w2_blockscale)
            torch.cuda.synchronize()
            self.csf_ref = (rep.w13_sfb.detach().clone().contiguous().view(torch.int32).reshape(-1), rep.w2_sfb.detach().clone().contiguous().view(torch.int32).reshape(-1))
            self.csf_dbg = (torch.full_like(self.csf_ref[0], -1), torch.full_like(self.csf_ref[1], -1))
            csf["csf13_debug"], csf["csf2_debug"] = self.csf_dbg
            self.csf_dbgdesc = tuple(torch.full((t.numel() // 128,), -1, dtype=torch.int32, device=t.device) for t in self.csf_ref)
            csf["csf13_debug_desc"], csf["csf2_debug_desc"] = self.csf_dbgdesc
            self.csf_gdesc = (csf["csf13_gdesc"], csf["csf2_gdesc"])
            self.csf_checks = 0
            self.csf_owner = owner
            self.csf_store = [(plane.scales.storage, plane.scales.storage.data_ptr(), plane.scales.storage.detach().clone()) for plane in (dec.first, dec.second)]
            print(f"[brrr-tp3] csf debug mirror armed for layer {index}: words {tuple(t.numel() for t in self.csf_ref)}", flush=True)
        scales = (self.raw(f"{prefix}.ffn.shared_experts.w1.scale", inter_blocks), self.raw(f"{prefix}.ffn.shared_experts.w3.scale", inter_blocks))
        kwargs = dict(
            # TP3 port: per-rank o-proj shard = ext.OA rows (3 padded groups x 1024); the last rank's shard runs past
            # the checkpoint's 8 real groups and is zero-filled (raw_padded), matching kraken's padded wo_b module weight.
            woa=layer.attn.wo_a.weight if DENSE_NATIVE else stages(self.raw_padded(f"{prefix}.attn.wo_a.weight", shard(r, self.ext.OA))),
            woa_s=self.raw_padded(f"{prefix}.attn.wo_a.scale", shard(r, self.ext.OA // 32)),
            wob=layer.attn.wo_b.weight if DENSE_NATIVE else stages(layer.attn.wo_b.weight),
            wob_s=self.raw_padded(f"{prefix}.attn.wo_b.scale", cols=shard(r, self.ext.OA // 32)),
            gate=ffn.gate.weight if DENSE_NATIVE else stages(ffn.gate.weight),
            gate_b=ffn.gate.e_score_correction_bias.float().contiguous(),
            bias_vl=None if bias_vl is None else bias_vl.float().contiguous(),
            sh13=shared.gate_up_proj.weight if DENSE_NATIVE else stages(interleave(shared.gate_up_proj.weight, 32)),
            sh13_s=interleave(torch.cat(scales), 1),
            sh2=shared.down_proj.weight if DENSE_NATIVE else stages(shared.down_proj.weight),
            sh2_s=torch.nn.functional.pad(self.raw(f"{prefix}.ffn.shared_experts.w2.scale", cols=inter_blocks), (0, self.ext.SH2_LD - inter // 32)).contiguous(),
            w13=rep.w13_rp.contiguous(),
            s13=(decoded[0] if decoded is not None else rep.w13_sfb.contiguous()),
            w2=rep.w2_rp.contiguous(),
            s2=(decoded[1] if decoded is not None else rep.w2_sfb.contiguous()),
            fn=layer.hc_ffn_fn.float().contiguous(),
            hc_scale=layer.hc_ffn_scale.float().contiguous(),
            hc_base=layer.hc_ffn_base.float().contiguous(),
            norm=layer.ffn_norm.weight.contiguous(),
            image_lo=int(getattr(ffn, "image_sentinel_lo", 0) or 0),
            dense_native=DENSE_NATIVE,
            **csf,
        )
        if os.environ.get("DS41_FUSED_NATIVE_PROBE") == "1" and index in (0, 20, 39) and not DENSE_NATIVE:
            # TP3: can the kernel read vLLM's own dense weights instead of staged copies?
            wa = getattr(layer.attn.wo_a, "weight", None)
            cand = {"woa": wa, "wob": layer.attn.wo_b.weight, "gate": ffn.gate.weight,
                    "sh13": shared.gate_up_proj.weight, "sh2": shared.down_proj.weight}
            tot = 0
            for k, t in cand.items():
                staged = kwargs[k]; tot += staged.numel()
                if t is None:
                    print(f"[brrr-tp3] native probe L{index} {k}: no module weight", flush=True); continue
                mapped = interleave(t, 32) if k == "sh13" else t
                try:
                    eq = torch.equal(stages(mapped), staged)
                except Exception as ex:
                    eq = f"err {ex}"
                print(f"[brrr-tp3] native probe L{index} {k}: shape={tuple(t.shape)} stride={tuple(t.stride())} dtype={t.dtype} contig={t.is_contiguous()} "
                      f"ptr%256={t.data_ptr() % 256} stages(native)==staged: {eq}", flush=True)
            print(f"[brrr-tp3] native probe L{index}: staged dense bytes/layer = {tot / 2**20:.1f} MiB", flush=True)
            if DUMP:
                torch.save({k: (v.detach().cpu() if v is not None else None) for k, v in cand.items()}, f"{DUMP}/native{index}_rank{r}.pt")
        if DUMP and (not DUMP_LAYERS or index in DUMP_LAYERS):  # TP3 debug: offline replay (harness/replay.py)
            self.dump_csf(owner, rep, index)
            torch.save({k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}, f"{DUMP}/layer{index}_rank{r}.pt")
        self.layers[index] = self.ext.add_layer(self.handle, **kwargs)
        if os.environ.get("BRRR_TP3_LOG") == "1":
            self.ptrs = getattr(self, "ptrs", {})
            self.ptrs[index] = (owner, {"w13": rep.w13_rp.data_ptr(), "w2": rep.w2_rp.data_ptr(), "s13": rep.w13_sfb.data_ptr(),
                                        **({"csf13": csf["csf13_storage"].data_ptr(), "csf2": csf["csf2_storage"].data_ptr()} if csf else {})}, {})
        return self.layers[index]

    def prologue(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        pre_mix: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        m, attn, device, ext = residual.shape[0], layer.attn, residual.device, self.ext
        resid_out, layer_input = torch.empty_like(residual), empty(device, m, ext.HID)
        post, comb, pre = mixes(device, m, ext.HC)
        qr, kv, q = empty(device, m, attn.q_lora_rank), empty(device, m, ext.KVW), empty(device, m, attn.n_local_heads, attn.head_dim)
        qr_q = empty(device, m, attn.q_lora_rank, dtype=torch.uint8)
        qr_s = empty(device, ext.act_scale_bytes(m), dtype=torch.uint8)
        metadata = get_forward_context().attn_metadata
        swa = attn._query_metadata(metadata[attn.swa_cache_layer.prefix]) if isinstance(metadata, dict) else None
        cache = (attn.swa_cache_layer.kv_cache, attn.swa_cache_layer.block_size, swa.slot_mapping[:m]) if swa is not None else (None, 0, None)
        ext.prologue(
            self.handle, x.contiguous(), residual.contiguous(), post_mix.contiguous(), res_mix.contiguous(), pre_mix.contiguous(),
            layer.hc_attn_fn, layer.attn_norm.weight, *attn.fused_wqa_wkv.skinny, attn.q_norm.weight, attn.kv_norm.weight,
            attn.q_norm.variance_epsilon, *attn.wq_b.skinny, positions.long().contiguous(), attn.rotary_emb.cos_sin_cache,
            resid_out, layer_input, qr, kv, qr_q, qr_s, q, *cache,
        )
        self.pending_mix = post
        qr.mx = (qr_q, qr_s)
        kv.cached = swa is not None
        return resid_out, post, comb, layer_input, pre, q, kv, qr

    def ffn(
        self,
        layer: nn.Module,
        o: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        attn_pre: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        m, device, ext = residual.shape[0], residual.device, self.ext
        index = self.register(layer)
        outputs = (
            empty(device, m, ext.HID),
            torch.empty_like(residual),
            *mixes(device, m, ext.HC),
        )
        mix = (layer.hc_attn_scale, layer.hc_attn_base) if post_mix is self.pending_mix else (None, None)
        if os.environ.get("BRRR_TP3_LOG") == "1" and layer.layer_index in getattr(self, "ptrs", {}) and not torch.cuda.is_current_stream_capturing():
            owner_, reg, calls = self.ptrs[layer.layer_index]
            calls["n"] = calls.get("n", 0) + 1
            if calls["n"] <= 3:
                impl_ = owner_._prepared()._impl
                rep_ = impl_.representation_for("w4a8_mx")
                dec_ = getattr(impl_, "nvfp4_csf", None)
                now = {"w13": rep_.w13_rp.data_ptr(), "w2": rep_.w2_rp.data_ptr(), "s13": rep_.w13_sfb.data_ptr(),
                       **({"csf13": dec_.first.scales.storage.data_ptr(), "csf2": dec_.second.scales.storage.data_ptr()} if dec_ is not None else {})}
                moved = {k: (hex(reg[k]), hex(now.get(k, 0))) for k in reg if now.get(k) != reg[k]}
                print(f"[brrr-tp3] ptrcheck layer {layer.layer_index} call {calls['n']} m={m}: {'RELOCATED ' + str(moved) if moved else 'pointers unchanged'}", flush=True)
        if DUMP and (not DUMP_LAYERS or layer.layer_index in DUMP_LAYERS) and not torch.cuda.is_current_stream_capturing():
            self.dumped[layer.layer_index] = self.dumped.get(layer.layer_index, 0) + 1
        if DUMP and (not DUMP_LAYERS or layer.layer_index in DUMP_LAYERS) and self.dumped.get(layer.layer_index) == int(os.environ.get("DS41_FUSED_DUMP_CALL", "1")) and not torch.cuda.is_current_stream_capturing():
            self.dumped[layer.layer_index] = 10**9
            args = (o, positions.long().contiguous(), input_ids, layer.attn.rotary_emb.cos_sin_cache, residual.contiguous(), post_mix.contiguous(),
                    res_mix.contiguous(), attn_pre.contiguous(), *mix)
            torch.save({"layer": index, "m": m, "args": [a.cpu() if isinstance(a, torch.Tensor) else a for a in args]},
                       f"{DUMP}/run{layer.layer_index}_rank{self.rank}.pt")
            torch.cuda.synchronize()
        ext.run(
            self.handle, index, o, positions.long().contiguous(), input_ids, layer.attn.rotary_emb.cos_sin_cache,
            residual.contiguous(), post_mix.contiguous(), res_mix.contiguous(), attn_pre.contiguous(), *mix, *outputs,
        )
        if layer.layer_index == CSF_DEBUG_LAYER and getattr(self, "csf_checks", 10**9) < int(os.environ.get("DS41_FUSED_CSF_DEBUG_CALLS", "6")) and not torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize()
            self.csf_checks += 1
            dec_now = getattr(self.csf_owner._prepared()._impl, "nvfp4_csf", None)
            for i, (st0, ptr0, copy0) in enumerate(self.csf_store):
                st_now = (dec_now.first, dec_now.second)[i].scales.storage if dec_now is not None else None
                changed = int((st0 != copy0).sum())
                where = int((st0 != copy0).nonzero()[0]) if changed else -1
                print(f"[brrr-tp3] csf storage check #{self.csf_checks} plane{i}: registered ptr={ptr0:#x} live ptr={(st_now.data_ptr() if st_now is not None else 0):#x} "
                      f"same_obj={st_now is st0} registered-tensor bytes changed={changed} first_changed_off={where} numel={st0.numel()} "
                      f"live_equal_registered={bool(torch.equal(st_now, st0)) if st_now is not None and st_now.numel() == st0.numel() else 'n/a'}", flush=True)
            for tag, dbg, ref in (("w13", self.csf_dbg[0], self.csf_ref[0]), ("w2", self.csf_dbg[1], self.csf_ref[1])):
                written = dbg != -1
                bad = written & (dbg != ref)
                msg = f"[brrr-tp3] csf runtime check #{self.csf_checks} layer {layer.layer_index} m={m} {tag}: written={int(written.sum())} mismatching={int(bad.sum())}"
                if int(bad.sum()):
                    idx = bad.nonzero().reshape(-1)[:6]
                    msg += f" first bad idx={idx.tolist()} got={[hex(int(v) & 0xffffffff) for v in dbg[idx]]} ref={[hex(int(v) & 0xffffffff) for v in ref[idx]]}"
                    # staged descriptor the consumer saw for the first bad word's 128-row slice vs the gdesc entry it should be
                    i0 = int(idx[0]); tile, half = i0 // 256, (i0 % 256) // 128
                    if tag == "w13":
                        e, rem = tile // 240, tile % 240; nt256, kt = rem // 40, rem % 40
                        se, sg, groups = e * 2 + nt256 // 3, ((nt256 % 3) * 2 + half) * 40 + kt, 240
                    else:
                        e, rem = tile // 120, tile % 120; nt256, kt = rem // 6, rem % 6
                        se, sg, groups = e, (nt256 * 2 + half) * 6 + kt, 240
                    gi = 0 if tag == "w13" else 1
                    staged = int(self.csf_dbgdesc[gi][i0 // 128]) & 0xffffffff
                    want = int(self.csf_gdesc[gi][se * groups + sg]) & 0xffffffff
                    hits = (self.csf_gdesc[gi] == int(self.csf_dbgdesc[gi][i0 // 128])).nonzero().reshape(-1)[:4].tolist()
                    msg += (f" | slice e={e} nt256={nt256} kt={kt} half={half} se={se} sg={sg}: staged desc={staged:#x} (first={staged & 0x3fffff} count={staged >> 22})"
                            f" want={want:#x} (first={want & 0x3fffff} count={want >> 22}) staged-desc-found-at-gdesc-idx={hits}")
                    nd = self.csf_dbgdesc[gi]; wd = self.csf_gdesc[gi]
                    seen = nd != -1
                    msg += f" | slices seen={int(seen.sum())}"
                print(msg, flush=True)
                dbg.fill_(-1)
            for t_ in self.csf_dbgdesc: t_.fill_(-1)
        return outputs


@cache
def engine(model_dir: str) -> Engine:
    group = get_tp_group()
    return Engine(model_dir, group.rank_in_group, group.cpu_group)


def eligible(layer: nn.Module, positions: torch.Tensor) -> bool:
    ext = cuext.extension("decoder")
    if not (
        getattr(layer, "layer_index", LAYERS) < LAYERS
        and positions.shape[0] <= (FUSED_MAXM or ext.MAXM)
        and not layer.use_sequence_parallel
        and getattr(layer.ffn, "n_routed_experts", 0) == ext.EXPERTS
        and is_forward_context_available()
    ):
        return False
    metadata = get_forward_context().attn_metadata
    native = [m for m in metadata.values() if hasattr(m, "is_decode")] if isinstance(metadata, dict) else []
    return bool(native) and all(m.is_decode and m.decoder is None for m in native)


def fuses_prologue(layer: nn.Module, positions: torch.Tensor, residual: torch.Tensor | None) -> bool:
    return residual is not None and layer.engram is None and positions.shape[0] <= PROLOGUE_MAXM
