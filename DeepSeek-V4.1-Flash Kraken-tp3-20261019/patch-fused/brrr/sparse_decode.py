from functools import cache

import os
import torch
from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV41B12xMetadata

from brrr import cuext

MAX_ROWS = 48  # TP3: 8 seqs x (5 drafts + 1); partials are sized MAX_ROWS x MAXCL x H x D fp32 (~28 MB)
ENABLED = os.environ.get("DS41_SPARSE_DECODE", "0") == "1"

type Indexed = tuple[torch.Tensor, int, torch.Tensor, DeepseekV41B12xMetadata]


@cache
def partial(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ext = cuext.extension("sparse_decode")
    return (
        torch.empty(MAX_ROWS * ext.CL * ext.H * ext.D, dtype=torch.float32, device=device),
        torch.empty(MAX_ROWS * ext.CL * ext.H * 2, dtype=torch.float32, device=device),
        torch.zeros(MAX_ROWS, dtype=torch.int32, device=device),  # per-row CTA arrival counters (reset by the last CTA)
    )


def fits(q: torch.Tensor, swa_width: int, indexed_width: int) -> bool:
    if not ENABLED:
        return False
    if 'fits' not in _LOGGED:
        _LOGGED.add('fits'); (os.environ.get('BRRR_TP3_LOG') == '1' and print(f'[brrr-tp3] sparse_decode.fits called: q={tuple(q.shape)} swa_width={swa_width} indexed_width={indexed_width}', flush=True))
    ext = cuext.extension("sparse_decode")
    total = swa_width + indexed_width
    return (
        q.shape[0] <= MAX_ROWS
        and q.shape[1:] == (ext.H, ext.D)
        and q.dtype == torch.bfloat16
        and ext.ctas_for(total) > 0
    )


_LOGGED = set()

def decode(
    q: torch.Tensor,
    out: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_page: int,
    swa: DeepseekV41B12xMetadata,
    window: int,
    swa_width: int,
    draft: bool,
    replay: torch.Tensor | None,
    indexed: Indexed | None,
    sink: torch.Tensor,
    sm_scale: float,
) -> None:
    if 'decode' not in _LOGGED:
        _LOGGED.add('decode'); (os.environ.get('BRRR_TP3_LOG') == '1' and print('[brrr-tp3] sparse_decode.decode engaged', flush=True))
    match indexed:
        case None:
            index = (None, 0, None, None, None, None)
        case (idx_cache, idx_page, idx_logical, main):
            index = (idx_cache, idx_page, idx_logical, main.cache_lengths, main.req_id_per_token, main.block_table)
    cuext.extension("sparse_decode").decode(
        q, out, swa_cache, swa_page, swa.positions, swa.req_id_per_token, swa.block_table, swa.query_start_loc, swa.request_positions, replay,
        window, swa_width, draft, *index, sink, sm_scale, *partial(q.device),
    )
