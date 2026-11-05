#!/bin/bash
# Export the package from a checkpoint view (make_checkpoint_view.sh), run from the fork checkout with its .venv
# (uv sync, then re-apply tp3-patches/ to the installed FlashInfer). ~40 min on 3 GPUs with a cold tuning cache.
#   CKPT   checkpoint view directory (required)
#   HOT    hot experts per parked layer (default 300; 313 with PREFILL=2048 gives the 3.47M-KV variant)
#   PREFILL prefill chunk size (default 4096)
# A cold tuning cache needs more free VRAM than serving. If tuning runs out of memory, export once with a lower
# HOT (for example 290), then export the target; the tuning cache is shared between the two.
set -euo pipefail
: "${CKPT:?set CKPT to the checkpoint view directory}"
HERE=$(dirname "$(readlink -f "$0")")
export FLARE_DS41_TP3_GRAPH_GIB=1 FLARE_DS41_HOST_EXPERT_LAYERS=20-33 FLARE_DS41_HOT_EXPERTS=${HOT:-300} \
       FLARE_DS41_MERGED_BANK=1 FLARE_DS41_TP3_VOCAB_PARALLEL=1 \
       FLARE_DS41_EXPERT_COUNTS_DIR=${COUNTS:-$HERE/expert-order} HF_HUB_OFFLINE=1
exec .venv/bin/ignition export "$CKPT" --gpus "${GPUS:-0,1,2}" --max-concurrency 8 \
  --prefill-chunk-size "${PREFILL:-4096}" --mtp-draft-tokens 5 --draft-head-quantization nvfp4 \
  --markov-head-quantization fp8 --vocabulary-parallel-sampling --verbose
