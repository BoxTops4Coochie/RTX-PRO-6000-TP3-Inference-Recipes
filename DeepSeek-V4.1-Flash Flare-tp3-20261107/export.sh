#!/bin/bash
# Export the package from a checkpoint view (make_checkpoint_view.sh). Run it from the fork checkout with its .venv
# (uv sync, ignition.patches, then re-apply tp3-patches/ to the installed FlashInfer).
#   CKPT     checkpoint view directory (required)
#   HOT      hot experts per parked layer (default 342 = 2.37M KV; each expert fewer adds ~92K KV tokens, see EXPORT.md)
#   PREFILL  prefill chunk size (default 4096)
#   DRAFTS   DSpark drafts (default 5)
# A cold tuning cache needs more free VRAM than serving. If tuning runs out of memory, export once with HOT=300,
# then export the target. The second export reuses the shared tuning cache.
set -euo pipefail
: "${CKPT:?set CKPT to the checkpoint view directory}"
HERE=$(dirname "$(readlink -f "$0")")
export FLARE_DS41_TP3_GRAPH_GIB=1 FLARE_DS41_HOST_EXPERT_LAYERS=20-33 FLARE_DS41_HOT_EXPERTS=${HOT:-342} \
       FLARE_DS41_MERGED_BANK=1 FLARE_DS41_TP3_VOCAB_PARALLEL=1 \
       FLARE_DS41_EXPERT_COUNTS_DIR=${COUNTS:-$HERE/expert-order} HF_HUB_OFFLINE=1 \
       FLARE_DS41_BRRR_LAYERS=0-39 FLARE_DS41_BRRR_CKPT="$CKPT" FLARE_DS41_BRRR_PARKED_MAX_ROWS=12 \
       FLARE_DS41_EMBED_HOST=1 FLARE_DS41_VISION_HOST=1 FLARE_DS41_AR_ROWS=1024
exec .venv/bin/ignition export "$CKPT" --gpus "${GPUS:-0,1,2}" --max-concurrency 8 \
  --prefill-chunk-size "${PREFILL:-4096}" --mtp-draft-tokens "${DRAFTS:-5}" --draft-head-quantization nvfp4 \
  --markov-head-quantization fp8 --vocabulary-parallel-sampling --verbose
