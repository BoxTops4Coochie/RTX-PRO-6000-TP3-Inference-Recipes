#!/bin/bash
# Build the checkpoint this release exports from: DeepSeek's original shards 1-46 plus NVIDIA's shards 47-48 (the two
# Engram tables in MXFP4), with this release's config.json and model.safetensors.index.json. Uses symlinks only.
#   usage: make_checkpoint_view.sh <deepseek snapshot dir> <nvidia NVFP4 snapshot dir> <output dir>
# Download first, for example:
#   hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170
#   hf download nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 --include "model-0004[78]-of-00048.safetensors" LICENSE README.md
set -euo pipefail
DS=$(readlink -f "$1"); NV=$(readlink -f "$2"); OUT=$3; HERE=$(dirname "$(readlink -f "$0")")
mkdir -p "$OUT"
for f in "$DS"/*; do
  b=$(basename "$f")
  case "$b" in config.json|model.safetensors.index.json|model-00047-of-00048.safetensors|model-00048-of-00048.safetensors) ;;
    *) ln -sfn "$(readlink -f "$f")" "$OUT/$b" ;;
  esac
done
for s in 00047 00048; do ln -sfn "$(readlink -f "$NV/model-$s-of-00048.safetensors")" "$OUT/model-$s-of-00048.safetensors"; done
cp "$HERE/checkpoint/config.json" "$HERE/checkpoint/model.safetensors.index.json" "$OUT/"
python3 - "$OUT" <<'EOF'
import json, struct, sys
out = sys.argv[1]
idx = json.load(open(f"{out}/model.safetensors.index.json"))
seen, missing = {}, []
for name, shard in idx["weight_map"].items():
    if shard not in seen:
        with open(f"{out}/{shard}", "rb") as h:
            n = struct.unpack("<Q", h.read(8))[0]
            seen[shard] = json.loads(h.read(n))
    if name not in seen[shard]:
        missing.append(name)
e = seen["model-00047-of-00048.safetensors"].get("layers.1.engram.embed.weight", {})
print(f"{len(idx['weight_map'])} tensors in {len(seen)} shards, missing {len(missing)}; "
      f"layer-1 Engram table {e.get('dtype')} {e.get('shape')} (expect U8 [384006168, 128])")
sys.exit(1 if missing or e.get("dtype") != "U8" else 0)
EOF
echo "checkpoint view ready: $OUT"
