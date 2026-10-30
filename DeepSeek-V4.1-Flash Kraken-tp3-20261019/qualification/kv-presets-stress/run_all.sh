#!/usr/bin/env bash
# Stress every KV preset in turn: 8x (already running), then 6x, 4x, 2x via kv.sh.
set -u
cd /home/aabduh/ds41-tp3/release/ds41-kraken-tp3-20261019
declare -A CAP=([8x]=8499917 [6x]=6314130 [4x]=4614187 [2x]=2185581)
wait_ready() { for i in $(seq 1 100); do docker compose logs 2>/dev/null | grep -q "expert promotion armed and warmed" && return 0; sleep 10; done; echo "TIMEOUT waiting for server"; return 1; }
for p in 8x 6x 4x 2x; do
  if [ "$p" != "8x" ]; then ./kv.sh $p >/dev/null 2>&1; sleep 20; wait_ready || continue; fi
  echo "=== $(date '+%H:%M') preset $p: $(docker compose logs 2>/dev/null | grep -oE 'GPU KV cache size: [0-9,]+ tokens' | head -1) ==="
  python3 /home/aabduh/ds41-tp3/stress/stress_kv.py $p ${CAP[$p]} 530000 4 200 2>&1
done
echo "ALL DONE $(date '+%H:%M')"
