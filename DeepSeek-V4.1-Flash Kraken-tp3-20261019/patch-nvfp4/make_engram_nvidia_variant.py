#!/usr/bin/env python3
"""Build `local/DeepSeek-V4.1-Flash-1017-engram-nvidia`: the stock DeepSeek-V4.1-Flash-1017 checkpoint with its
two Engram tables replaced by NVIDIA's MXFP4 tables (from nvidia/DeepSeek-V4.1-Flash-1017-NVFP4). No conversion,
no quantization: every file is a hard link, so the variant costs no extra disk beyond the two NVIDIA shards.

Needs in the HF cache (HF_HOME, default ~/.cache/huggingface):
  * deepseek-ai/DeepSeek-V4.1-Flash-1017 (all 48 shards)
  * the two NVIDIA shards that hold the Engram tables (~49 GiB each; nothing else from that repo is needed):
      hf download nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 --include "model-00047-of-00048.safetensors" "model-00048-of-00048.safetensors" config.json
Then:  python3 make_engram_nvidia_variant.py [--hf-home /path/to/huggingface]
Serve: MODEL=local/DeepSeek-V4.1-Flash-1017-engram-nvidia (image 20261019.4 or compose.nvfp4.yaml), ENGRAM_RAM_LAYERS=1,14
Plain Python 3, no torch.
"""
import argparse, glob, json, os, struct, sys

STOCK = "models--deepseek-ai--DeepSeek-V4.1-Flash-1017"
NVIDIA = "models--nvidia--DeepSeek-V4.1-Flash-1017-NVFP4"
DST = "models--local--DeepSeek-V4.1-Flash-1017-engram-nvidia"
ENGRAM_SHARDS = ("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors")

def snapshot(hub, repo):
    hits = sorted(glob.glob(os.path.join(hub, repo, "snapshots", "*")))
    if not hits:
        sys.exit(f"missing {repo} under {hub}")
    return hits[-1]

def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    a = ap.parse_args()
    hub = os.path.join(a.hf_home, "hub")
    stock, nvidia = snapshot(hub, STOCK), snapshot(hub, NVIDIA)
    # sanity: NVIDIA's engram shards must hold only Engram tensors + tensors identical in dtype/shape to stock
    for sh in ENGRAM_SHARDS:
        p = os.path.join(nvidia, sh)
        if not os.path.exists(os.path.realpath(p)):
            sys.exit(f"NVIDIA shard not downloaded: {sh}")
        hn, hs = header(p), header(os.path.join(stock, sh))
        for k, v in hn.items():
            if k.endswith("engram.embed.weight"):
                if v["dtype"] != "U8" or v["shape"][1] != 128:
                    sys.exit(f"{sh}: {k} is not a packed MXFP4 table ({v['dtype']} {v['shape']})")
            elif k.endswith("engram.embed.scale"):
                if v["dtype"] not in ("U8", "F8_E8M0") or v["shape"][1] != 8:
                    sys.exit(f"{sh}: {k} is not a UE8M0 block-32 scale table ({v['dtype']} {v['shape']})")
            elif k not in hs or hs[k]["dtype"] != v["dtype"] or hs[k]["shape"] != v["shape"]:
                sys.exit(f"{sh}: unexpected tensor {k}; refusing to mix")
    rev = os.path.basename(stock)
    dst = os.path.join(hub, DST, "snapshots", rev)
    os.makedirs(dst, exist_ok=True); os.makedirs(os.path.join(hub, DST, "refs"), exist_ok=True)
    open(os.path.join(hub, DST, "refs", "main"), "w").write(rev)
    for name in sorted(os.listdir(stock)):
        src = os.path.realpath(os.path.join(nvidia if name in ENGRAM_SHARDS else stock, name))
        d = os.path.join(dst, name)
        if name == "config.json" or os.path.lexists(d):
            continue
        if os.path.isdir(src):
            os.symlink(src, d)
        else:
            try: os.link(src, d)
            except OSError: os.symlink(src, d)
    cfg = json.load(open(os.path.join(stock, "config.json")))
    qc = dict(cfg.get("quantization_config") or {})
    qc.update({"engram_embed_quant": "mxfp4", "engram_embed_block_size": 32,
               "engram_embed_scale_dtype": "ue8m0", "engram_embed_layers": list(cfg.get("engram_layer_ids", [1, 14]))})
    cfg["quantization_config"] = qc
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    print(f"built {dst}\n  {len(os.listdir(dst))} entries; Engram shards from NVIDIA, everything else hard-linked to stock")
    print("serve with MODEL=local/DeepSeek-V4.1-Flash-1017-engram-nvidia  (ENGRAM_RAM_LAYERS=1,14 recommended)")

if __name__ == "__main__":
    main()
