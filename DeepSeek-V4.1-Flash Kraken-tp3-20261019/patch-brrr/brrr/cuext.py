import hashlib
from functools import cache
from pathlib import Path
from types import ModuleType

import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.utils.cpp_extension import load

KERNELS = Path(__file__).resolve().parent / "kernels"
SOURCES = {path.stem: path for path in sorted((*KERNELS.glob("*.cu"), *KERNELS.glob("*.cpp")))}
LIBRARIES = {"engram_nvme": ["-luring"]}


@cache
def extension(name: str) -> ModuleType:
    source = SOURCES[name]
    digest = hashlib.sha1(b"".join(path.read_bytes() for path in (source, *sorted(KERNELS.glob("*.cuh"))))).hexdigest()[:10]
    cuda = source.suffix == ".cu"
    return load(
        name=f"{name}_{digest}",
        sources=[str(source)],
        extra_cflags=None if cuda else ["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"] if cuda else None,
        extra_ldflags=LIBRARIES.get(name),
        verbose=False,
    )


def exchange_ipc(ext: ModuleType, nbytes: int, rank: int, group: ProcessGroup) -> list[int]:
    mine = ext.alloc_ipc(nbytes)
    handles: list[list[int]] = [[] for _ in range(ext.RANKS)]
    dist.all_gather_object(handles, mine[1:], group=group)
    return [mine[0] if peer == rank else ext.open_ipc(handles[peer]) for peer in range(ext.RANKS)]
