#!/bin/bash
# Serve the exported package on 3 GPUs with both Engram tables in host RAM (~97 GiB).
#   FLARE         path to this fork's flare binary   (default: flare/target/release/flare in the fork checkout)
#   PACKAGE       path to the exported package        (default: the default version in Flare's model cache)
#   ENGRAM_RAM=0  read the Engram tables from disk instead: ~90 GiB less host RAM, decode ~4% slower, prefill unchanged
#   PREFIX_CACHE  host RAM for reusable prompt KV (default 16G; 32G/48G cost only RAM)
set -euo pipefail
FLARE=${FLARE:-flare/target/release/flare}
PACKAGE=${PACKAGE:-ds41-1017-engram-mxfp4}
ENV=() STORAGE=()
if [ "${ENGRAM_RAM:-1}" = 1 ]; then ENV=(FLARE_ENGRAM_RAM_TABLES=0); else STORAGE=(--embedding-table-storage disk); fi
# /usr/bin/env explicitly: some shells put another `env` (for example uv's) first on PATH.
exec /usr/bin/env "${ENV[@]}" "$FLARE" run "$PACKAGE" --gpus "${GPUS:-0,1,2}" --port "${PORT:-15015}" \
  --host "${HOST:-127.0.0.1}" --served-model-name DeepSeek-V4.1-Flash --prefix-cache-memory "${PREFIX_CACHE:-16G}" \
  --mtp-draft-tokens "${DRAFTS:-5}" --mtp-adaptive-verification --mtp-max-verification-token-budget 128 \
  --mtp-adaptive-verification-alpha "${ALPHA:-0.8}" --scheduler-policy "${SCHEDULER:-balanced}" "${STORAGE[@]}"
