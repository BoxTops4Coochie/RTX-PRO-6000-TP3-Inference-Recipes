from collections.abc import Callable

import os
import torch

from brrr import cuext

CANDIDATE_BLOCKS = 2048
MODE = "1"  # set by patches.patch_indexer: 1 = both radix paths, sorted / blocks = only that one

type TopK = Callable[..., tuple[torch.Tensor, torch.Tensor]]
type Launch = Callable[..., None]
type Fused = Callable[[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None], None]
DEFERRED: dict[int, Fused] = {}


_LOGGED = set()

def row_topk(stock: TopK) -> TopK:
    if 'row_topk' not in _LOGGED:
        _LOGGED.add('row_topk'); (os.environ.get('BRRR_TP3_LOG') == '1' and print('[brrr-tp3] candidates.row_topk installed', flush=True))
    def select(
        *,
        row_logits: torch.Tensor,
        lengths: torch.Tensor,
        topk: int,
        output_values: torch.Tensor | None = None,
        output_indices: torch.Tensor | None = None,
        output_index_offset: int = 0,
        output_gather_table: torch.Tensor | None = None,
        write_values: bool = True,
        launcher: object = None,
        launcher_sink: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        plain = output_values is not None and output_indices is not None and output_index_offset == 0 and write_values and launcher_sink is None
        sortable = (
            plain
            and (output_gather_table is None or (output_gather_table.dtype == torch.int32 and output_gather_table.is_contiguous()))
            and row_logits.dtype == torch.float32
            and row_logits.stride(1) == 1
            and lengths.dtype == torch.int32
            and lengths.is_contiguous()
        )
        if sortable and MODE in ("1", "sorted"):
            DEFERRED[output_indices.data_ptr()] = lambda out, values, visible, active, out_lengths: cuext.extension("candidates").select_sorted(
                row_logits, lengths, visible, active, topk, out, values, out_lengths, output_gather_table
            )
            return output_values, output_indices
        blocks = (
            MODE in ("1", "blocks")
            and plain
            and topk == CANDIDATE_BLOCKS
            and output_gather_table is None
            and row_logits.shape[0]
            and row_logits.shape[1]
            and cuext.extension("candidates").select_blocks(row_logits, lengths, topk, output_indices, output_values)
        )
        if blocks:
            return output_values, output_indices
        return stock(
            row_logits=row_logits,
            lengths=lengths,
            topk=topk,
            output_values=output_values,
            output_indices=output_indices,
            output_index_offset=output_index_offset,
            output_gather_table=output_gather_table,
            write_values=write_values,
            launcher=launcher,
            launcher_sink=launcher_sink,
        )

    return select


def sorted_launch(stock: Launch) -> Launch:
    def launch(kind: str, recipe: tuple, tensors: tuple, scalars: tuple, *, launcher: object = None) -> None:
        fused = DEFERRED.pop(tensors[0].data_ptr(), None) if kind == "sort" else None
        if fused is None:
            return stock(kind, recipe, tensors, scalars, launcher=launcher)
        _, _, out, out_values, visible, active, out_lengths = tensors
        return fused(out, None, visible, active, out_lengths) if recipe[1] else fused(out, out_values, visible, active, None)

    return launch
