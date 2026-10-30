#!/usr/bin/env bash
# Switch the KV cache preset in .env and restart the server.
#   ./kv.sh 2x | 4x | 6x | 8x        (see the preset block at the bottom of .env)
#   ./kv.sh                              -> show the active preset
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] || { cp .env.example .env; echo "created .env from .env.example — set HF_CACHE_DIR in it"; }
declare -A KV=([2x]=2415919104 [4x]=5100273664 [6x]=6979321856 [8x]=9395240960)
if [ $# -eq 0 ]; then
  cur=$(grep -E '^KV_CACHE_MEMORY_BYTES=' .env | cut -d= -f2 | cut -d' ' -f1)
  for k in "${!KV[@]}"; do [ "${KV[$k]}" = "$cur" ] && echo "active: $k ($cur bytes)"; done; exit 0
fi
p=${1}; [ -n "${KV[$p]:-}" ] || { echo "usage: $0 2x|4x|6x|8x"; exit 1; }
# comment every preset line, then uncomment the chosen one
sed -i -E 's/^#?(KV_CACHE_MEMORY_BYTES=[0-9]+ +# (2x|4x|6x|8x))/#\1/' .env
sed -i -E "s/^#(KV_CACHE_MEMORY_BYTES=${KV[$p]} +# $p)/\1/" .env
grep -E "^KV_CACHE_MEMORY_BYTES=" .env
docker compose up -d --force-recreate
echo "restarting with preset '$p' — ready in ~3 min (watch: docker compose logs -f | grep 'armed and warmed')"
