import os
import torch
from torch import nn

from brrr import cuext
_LOGGED = set()


def prepare(linear: nn.Module) -> None:
    weight = linear.weight
    if 'prepare' not in _LOGGED:
        _LOGGED.add('prepare'); (os.environ.get('BRRR_TP3_LOG') == '1' and print(f'[brrr-tp3] skinny.prepare: first linear weight={tuple(weight.shape)} dtype={weight.dtype}', flush=True))
    n, k = weight.shape
    if weight.element_size() != 1 or not weight.is_contiguous() or n % 32 or k % 64:
        return
    scales = linear.weight_scale_inv.view(torch.uint8).repeat_interleave(32, dim=0)[:n]
    linear.skinny = (weight.view(torch.uint8), scales.reshape(n // 8, 4, 2, k // 64, 2).permute(0, 3, 1, 4, 2).contiguous())


def eligible(linear: nn.Module, x: torch.Tensor) -> bool:
    return x.dim() == 2 and x.dtype == torch.bfloat16 and hasattr(linear, "skinny") and 0 < x.shape[0] <= cuext.extension("skinny").MAXM


_LOGGED = set()

def project(linear: nn.Module, x: torch.Tensor) -> torch.Tensor:
    weight, scales = linear.skinny
    if 'project' not in _LOGGED:
        _LOGGED.add('project'); (os.environ.get('BRRR_TP3_LOG') == '1' and print(f'[brrr-tp3] skinny.project engaged: x={tuple(x.shape)} w={tuple(weight.shape)}', flush=True))
    quantized, x_scales = getattr(x, "mx", (x, None))
    return cuext.extension("skinny").gemm(quantized, x_scales, weight, scales)


def split_norm(x: torch.Tensor, first: nn.Module, second: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    a, b, quantized, scales = cuext.extension("skinny").split_norm(
        x.reshape(x.shape[0], -1).contiguous(), first.weight.numel(), first.weight, second.weight, first.variance_epsilon
    )
    if quantized.numel():
        a.mx = (quantized, scales)
    return a, b
