#!/bin/bash
# Serve the exported package on 3 GPUs with both Engram tables in host RAM (~102 GiB).
#   FLARE    path to this fork's flare binary   (default: flare/target/release/flare in the fork checkout)
#   PACKAGE  path to the exported package        (default: the default version in Flare's model cache)
#   ENGRAM_RAM=0 to read the Engram tables from disk instead (needs less RAM, ~3-5% slower at c1)
set -euo pipefail
FLARE=${FLARE:-flare/target/release/flare}
PACKAGE=${PACKAGE:-ds41-1017-engram-mxfp4}
RAM=()
[ "${ENGRAM_RAM:-1}" = 1 ] && RAM=(FLARE_ENGRAM_RAM_TABLES=0)
# /usr/bin/env explicitly: some shells put another `env` (for example uv's) first on PATH.
exec /usr/bin/env "${RAM[@]}" "$FLARE" run "$PACKAGE" --gpus "${GPUS:-0,1,2}" --port "${PORT:-15015}" \
  --host "${HOST:-127.0.0.1}" --served-model-name DeepSeek-V4.1-Flash --prefix-cache-memory "${PREFIX_CACHE:-16G}" \
  --mtp-draft-tokens 5 --mtp-adaptive-verification --mtp-max-verification-token-budget 128 \
  --mtp-adaptive-verification-alpha 0.6 --scheduler-policy balanced
