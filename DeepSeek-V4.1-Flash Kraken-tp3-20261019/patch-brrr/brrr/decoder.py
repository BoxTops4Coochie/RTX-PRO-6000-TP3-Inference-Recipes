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

PROLOGUE_MAXM = 8
LAYERS = 40


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

    def raw(self, name: str, rows: slice = slice(None), cols: slice = slice(None)) -> torch.Tensor:
        return self.checkpoint.get(name, rows, cols).to(self.device)

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
        if not rep.n64_repack or rep.intermediate_size != inter:
            raise RuntimeError(f"ds41 layer kernel needs the n64 W4A8 expert layout with intermediate {inter}")
        bias_vl = getattr(ffn.gate, "bias_vl", None)
        scales = (self.raw(f"{prefix}.ffn.shared_experts.w1.scale", inter_blocks), self.raw(f"{prefix}.ffn.shared_experts.w3.scale", inter_blocks))
        self.layers[index] = self.ext.add_layer(
            self.handle,
            woa=stages(self.raw(f"{prefix}.attn.wo_a.weight", shard(r, 2048))),
            woa_s=self.raw(f"{prefix}.attn.wo_a.scale", shard(r, 64)),
            wob=stages(layer.attn.wo_b.weight),
            wob_s=self.raw(f"{prefix}.attn.wo_b.scale", cols=shard(r, 64)),
            gate=stages(ffn.gate.weight),
            gate_b=ffn.gate.e_score_correction_bias.float().contiguous(),
            bias_vl=None if bias_vl is None else bias_vl.float().contiguous(),
            sh13=stages(interleave(shared.gate_up_proj.weight, 32)),
            sh13_s=interleave(torch.cat(scales), 1),
            sh2=stages(shared.down_proj.weight),
            sh2_s=torch.nn.functional.pad(self.raw(f"{prefix}.ffn.shared_experts.w2.scale", cols=inter_blocks), (0, 2)).contiguous(),
            w13=rep.w13_rp.contiguous(),
            s13=rep.w13_sfb.contiguous(),
            w2=rep.w2_rp.contiguous(),
            s2=rep.w2_sfb.contiguous(),
            fn=layer.hc_ffn_fn.float().contiguous(),
            hc_scale=layer.hc_ffn_scale.float().contiguous(),
            hc_base=layer.hc_ffn_base.float().contiguous(),
            norm=layer.ffn_norm.weight.contiguous(),
            image_lo=int(getattr(ffn, "image_sentinel_lo", 0) or 0),
        )
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
        ext.run(
            self.handle, index, o, positions.long().contiguous(), input_ids, layer.attn.rotary_emb.cos_sin_cache,
            residual.contiguous(), post_mix.contiguous(), res_mix.contiguous(), attn_pre.contiguous(), *mix, *outputs,
        )
        return outputs


@cache
def engine(model_dir: str) -> Engine:
    group = get_tp_group()
    return Engine(model_dir, group.rank_in_group, group.cpu_group)


def eligible(layer: nn.Module, positions: torch.Tensor) -> bool:
    ext = cuext.extension("decoder")
    if not (
        getattr(layer, "layer_index", LAYERS) < LAYERS
        and positions.shape[0] <= ext.MAXM
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
