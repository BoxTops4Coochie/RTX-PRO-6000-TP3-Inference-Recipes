from collections.abc import Iterable

import torch
from b12x.sequence import engram as native
from torch import nn
from vllm.model_executor.weight_transfer import get_file_tensor_source
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from brrr import cuext

SCALE_BYTES = 8
HEADS = 24
ROWS_PER_PROGRAM = 16


def pinned(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(*shape, dtype=dtype, device="cpu", pin_memory=True)


@triton.jit
def _issue_kernel(counter_ptr, gen_ptr, ids_ready_ptr, num_rows):
    gen = tl.load(counter_ptr) + 1
    tl.store(counter_ptr, gen)
    tl.store(gen_ptr, gen)
    tl.atomic_xchg(ids_ready_ptr, (gen << 32) | num_rows.to(tl.int64), sem="release", scope="sys")


@triton.jit
def _wait_kernel(gen_ptr, rows_ready_ptr):
    gen = tl.load(gen_ptr)
    ready = tl.atomic_add(rows_ready_ptr, 0, sem="acquire", scope="sys")
    while ready < gen:
        ready = tl.atomic_add(rows_ready_ptr, 0, sem="acquire", scope="sys")


@triton.jit(do_not_specialize=["num_rows"])
def _dequant_rows_kernel(
    rows_ptr,
    scales_ptr,
    local_ptr,
    out_ptr,
    num_rows,
    HEAD_START: tl.constexpr,
    PART: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    rows = (tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)).to(tl.int64)
    valid = rows < num_rows
    head = rows % NUM_HEADS - HEAD_START
    source = rows // NUM_HEADS * PART + head
    cols = tl.arange(0, DIM)
    mine = valid & (head >= 0) & (head < PART)
    local = tl.load(local_ptr + source, mask=mine, other=-1)
    owned = mine & (local >= 0)
    values = tl.load(
        rows_ptr + source[:, None] * DIM + cols[None, :], mask=owned[:, None], other=0.0
    )
    scale = tl.load(
        scales_ptr + local[:, None] * (DIM // QUANT_BLOCK) + (cols // QUANT_BLOCK)[None, :],
        mask=owned[:, None],
        other=0,
    )
    scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    tl.store(
        out_ptr + rows[:, None] * DIM + cols[None, :],
        (values.to(tl.float32) * scale).to(tl.bfloat16),
        mask=valid[:, None],
    )


class NvmeEngramEmbedding(nn.Module):
    def __init__(
        self, plan: native.Plan, caps: native.Caps, geometry: native.Geometry, table_memory: str = "device", *, resident_scales: bool = False
    ) -> None:
        super().__init__()
        self.ext = cuext.extension("engram_nvme")
        self.plan = plan
        self.caps = caps
        self.table_memory = table_memory
        self.mapped_host_nbytes = 0
        layer = geometry.layer_ids.index(caps.layer_id)
        self.table_rows = geometry.num_embeddings[layer]
        self.part_n_hash_cols = triton.cdiv(HEADS, caps.tp_size)
        self.head_start = caps.tp_rank * self.part_n_hash_cols
        self.head_end = min(self.head_start + self.part_n_hash_cols, HEADS)
        offsets = (*geometry.offsets[layer], self.table_rows)
        self.vocab_start_idx = offsets[self.head_start]
        self.part_num_embeddings = offsets[self.head_end] - self.vocab_start_idx
        max_rows = caps.max_tokens * self.part_n_hash_cols
        self.ids = pinned(max_rows, dtype=torch.int64)
        self.rows = pinned(max_rows, self.ext.ROW, dtype=torch.float8_e4m3fn)
        self.scales = pinned(self.part_num_embeddings, SCALE_BYTES, dtype=torch.uint8)
        self.ids_ready = torch.zeros(1, dtype=torch.int64, device="cpu", pin_memory=True)
        self.rows_ready = torch.zeros(1, dtype=torch.int64, device="cpu", pin_memory=True)
        self.counter = torch.zeros(1, dtype=torch.int64, device=caps.device)
        self.gen = torch.zeros(1, dtype=torch.int64, device=caps.device)
        self.local = torch.full((max_rows,), -1, dtype=torch.int64, device=caps.device)
        self.rows_view = get_accelerator_view_from_cpu_tensor(self.rows)
        self.scales_view = get_accelerator_view_from_cpu_tensor(self.scales)
        self.ids_ready_view = get_accelerator_view_from_cpu_tensor(self.ids_ready)
        self.rows_ready_view = get_accelerator_view_from_cpu_tensor(self.rows_ready)
        self.store: int | None = None

    def close(self) -> None:
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        for name, value in weights:
            source = get_file_tensor_source(value)
            if source is None or source.shape[0] != self.table_rows:
                raise ValueError(f"NVMe Engram requires the file-backed global {name} table")
            match name:
                case "weight":
                    self.store = self.ext.create_store(
                        source.path,
                        source.offset + self.vocab_start_idx * self.ext.ROW,
                        self.ids,
                        self.rows.view(torch.uint8),
                        self.ids_ready,
                        self.rows_ready,
                    )
                case "weight_scale_inv":
                    with open(source.path, "rb") as handle:
                        handle.seek(source.offset + self.vocab_start_idx * SCALE_BYTES)
                        if handle.readinto(memoryview(self.scales.numpy()).cast("B")) != self.scales.numel():
                            raise RuntimeError(f"short Engram scale read from {source.path}")
                case _:
                    raise ValueError(f"Unknown Engram table weight: {name}")
            loaded.add(name)
        return loaded

    def lookup(self, indices: torch.Tensor, out: torch.Tensor) -> None:
        num_rows = indices.shape[0] * self.part_n_hash_cols
        if not num_rows:
            return
        local = self.local[:num_rows].view(indices.shape[0], self.part_n_hash_cols)
        torch.sub(indices[:, self.head_start : self.head_end], self.vocab_start_idx, out=local)
        self.ids[:num_rows].copy_(self.local[:num_rows], non_blocking=True)
        _issue_kernel[(1,)](self.counter, self.gen, self.ids_ready_view, num_rows)

    def wait(self, out: torch.Tensor, tokens: int) -> None:
        if not tokens:
            return
        _wait_kernel[(1,)](self.gen, self.rows_ready_view)
        _dequant_rows_kernel[(triton.cdiv(tokens * HEADS, ROWS_PER_PROGRAM),)](
            self.rows_view,
            self.scales_view,
            self.local,
            out,
            tokens * HEADS,
            HEAD_START=self.head_start,
            PART=self.head_end - self.head_start,
            NUM_HEADS=HEADS,
            DIM=self.ext.ROW,
            QUANT_BLOCK=self.ext.ROW // SCALE_BYTES,
            BLOCK_R=ROWS_PER_PROGRAM,
        )
