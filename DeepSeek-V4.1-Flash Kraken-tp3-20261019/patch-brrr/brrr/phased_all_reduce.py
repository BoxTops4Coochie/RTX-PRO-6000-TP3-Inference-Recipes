import math

import os
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from brrr import cuext

MIN_BYTES = 16 << 10
MAX_BYTES = 1 << 20


def ctas_for(nbytes: int) -> int:
    return 16 if nbytes <= 128 << 10 else 32 if nbytes <= 512 << 10 else 64


class PhasedAllReduce:
    def __init__(self, group: ProcessGroup, device: torch.device) -> None:
        self.ext = cuext.extension("phased_all_reduce")
        self.ranks = self.ext.RANKS
        self.disabled = dist.get_world_size(group=group) != self.ranks
        (os.environ.get('BRRR_TP3_LOG') == '1' and print(f'[brrr-tp3] PhasedAllReduce: world={dist.get_world_size(group=group)} RANKS={self.ranks} disabled={self.disabled}', flush=True))
        if self.disabled:
            return
        rank = dist.get_rank(group=group)
        with torch.cuda.device(device):
            self.handle = self.ext.create(cuext.exchange_ipc(self.ext, self.ext.flag_bytes() + 2 * self.ranks * MAX_BYTES, rank, group), rank, MAX_BYTES)
            self.ll = self.ext.create_ll(cuext.exchange_ipc(self.ext, self.ext.ll_bytes(), rank, group), rank)
        dist.barrier(group=group)

    def should_use(self, inp: torch.Tensor) -> bool:
        return (
            not self.disabled
            and inp.dtype == torch.bfloat16
            and inp.is_contiguous()
            and MIN_BYTES < inp.nbytes <= MAX_BYTES
            and inp.nbytes % 16 == 0
        )

    _seen: set = set()

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        key = ('ar', inp.nbytes)
        if key not in self._seen and len(self._seen) < 12:
            self._seen.add(key); (os.environ.get('BRRR_TP3_LOG') == '1' and print(f'[brrr-tp3] phased all_reduce engaged: {tuple(inp.shape)} {inp.dtype} {inp.nbytes} B', flush=True))
        out = torch.empty_like(inp)
        self.ext.all_reduce(self.handle, inp, out, ctas_for(inp.nbytes))
        return out

    def gather_words(self, inp: torch.Tensor) -> torch.Tensor:
        out = inp.new_empty((self.ranks, *inp.shape))
        self.ext.gather_words(self.ll, inp, out)
        return out

    def should_gather(self, inp: torch.Tensor, dim: int) -> bool:
        return (
            not self.disabled
            and inp.is_contiguous()
            and 0 < inp.nbytes <= MAX_BYTES
            and inner_bytes(inp, dim) % 4 == 0
            and inp.data_ptr() % 16 == 0
            and inp.untyped_storage().nbytes() - inp.storage_offset() * inp.element_size() >= (inp.nbytes + 15) // 16 * 16
        )

    def all_gather(self, inp: torch.Tensor, dim: int) -> torch.Tensor:
        key = ('ag', inp.nbytes)
        if key not in self._seen and len(self._seen) < 12:
            self._seen.add(key); (os.environ.get('BRRR_TP3_LOG') == '1' and print(f'[brrr-tp3] phased all_gather engaged: {tuple(inp.shape)} {inp.dtype} {inp.nbytes} B dim={dim}', flush=True))
        dim %= inp.dim()
        out = inp.new_empty((*inp.shape[:dim], self.ranks * inp.shape[dim], *inp.shape[dim + 1 :]))
        self.ext.all_gather(self.handle, inp, out, inner_bytes(inp, dim), 1 if inp.nbytes <= MIN_BYTES else ctas_for(inp.nbytes))
        return out


def inner_bytes(inp: torch.Tensor, dim: int) -> int:
    return math.prod(inp.shape[dim % inp.dim() :]) * inp.element_size()
