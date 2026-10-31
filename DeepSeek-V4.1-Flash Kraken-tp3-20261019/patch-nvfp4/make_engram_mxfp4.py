#!/opt/venv/bin/python
"""Build a DS41 checkpoint variant with ONLY the Engram tables requantized FP8 -> MXFP4.

Stock experts (lossless MXFP4-CSF path) stay untouched: every shard except the two that hold
`layers.{1,14}.engram.embed.{weight,scale}` is hard-linked; those two are rewritten with
  embed.weight : uint8 [rows, 128]  (E2M1 nibbles, even column in the low nibble)
  embed.scale  : uint8 [rows, 8]    (UE8M0, recomputed per 32-block so |x|/2^(e-127) <= 6)
and config.json gains engram_embed_quant/block_size/scale_dtype/layers (vLLM PR #61530 keys).
Streams row chunks, so peak RAM is ~2 GB. Run inside the serving image (needs torch+safetensors):
  docker run --rm --entrypoint /opt/venv/bin/python -v /m2-2/huggingface:/hf -v <dir>:/tools \
     azallaza/ds41-kraken-tp3:20261019.3 /tools/make_engram_mxfp4.py \
     --src /hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/<rev> \
     --dst-repo local/DeepSeek-V4.1-Flash-1017-engram-mxfp4 --hf-home /hf
"""
import argparse, json, os, struct, sys, time
import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
MIDS = (E2M1[1:] + E2M1[:-1]) / 2          # round-to-nearest thresholds

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

def quantize_rows(x_fp8_bytes: torch.Tensor, src_scales: torch.Tensor):
    """x: uint8 [n,256] raw E4M3 bytes, src_scales: uint8 [n,8] stock UE8M0 -> (packed uint8 [n,128], scales uint8 [n,8]).
    The stock value is fp8 * 2^(s-127): apply that first, then requantize to E2M1 with fresh block scales."""
    x = x_fp8_bytes.view(torch.float8_e4m3fn).to(torch.float32).view(-1, 8, 32)
    x = x * torch.ldexp(torch.ones_like(x), (src_scales.to(torch.int32) - 127).view(-1, 8, 1).to(torch.float32))
    amax = x.abs().amax(dim=2, keepdim=True)                       # [n,8,1]
    # UE8M0 exponent: smallest power of two with amax/scale <= 6; zero blocks -> e=0 (2^-127)
    e = torch.where(amax > 0, torch.ceil(torch.log2(amax / 6.0)) + 127, torch.zeros_like(amax))
    e = e.clamp(0, 254)
    scale = torch.ldexp(torch.ones_like(e), e - 127)
    y = (x / scale)                                                # |y| <= 6 (or 0)
    mag = y.abs().clamp(max=6.0)
    idx = torch.bucketize(mag, MIDS)                               # 0..7 nearest E2M1 magnitude
    nib = idx.to(torch.uint8) | ((y < 0).to(torch.uint8) << 3)    # sign in bit 3
    nib = nib.view(-1, 256)
    packed = nib[:, 0::2] | (nib[:, 1::2] << 4)                    # even col -> low nibble
    return packed.contiguous(), e.view(-1, 8).to(torch.uint8).contiguous()

def rewrite_shard(src_path, dst_path, engram_names, chunk_rows):
    hdr, data_start = read_header(src_path)
    meta = hdr.get("__metadata__")
    names = [k for k in hdr if k != "__metadata__"]
    # new header: engram weight -> U8 [rows,128], scale -> U8 [rows,8]; others verbatim
    new_hdr, off = {}, 0
    plan = []
    for k in sorted(names, key=lambda k: hdr[k]["data_offsets"][0]):
        info = hdr[k]; rows = info["shape"][0]
        if k in engram_names["weight"]:
            n = rows * 128; new_hdr[k] = {"dtype": "U8", "shape": [rows, 128], "data_offsets": [off, off + n]}; plan.append(("qw", k, off)); off += n
        elif k in engram_names["scale"]:
            n = rows * 8; new_hdr[k] = {"dtype": "U8", "shape": [rows, 8], "data_offsets": [off, off + n]}; plan.append(("qs", k, off)); off += n
        else:
            a, b = info["data_offsets"]; new_hdr[k] = dict(info, data_offsets=[off, off + (b - a)]); plan.append(("copy", k, off)); off += b - a
    if meta is not None: new_hdr["__metadata__"] = meta
    hb = json.dumps(new_hdr, separators=(",", ":")).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(src_path, "rb") as fi, open(dst_path, "wb") as fo:
        fo.write(struct.pack("<Q", len(hb))); fo.write(hb)
        for kind, k, _ in plan:
            a, b = hdr[k]["data_offsets"]; rows = hdr[k]["shape"][0]
            if kind == "copy":
                fi.seek(data_start + a); left = b - a
                while left: buf = fi.read(min(left, 64 << 20)); fo.write(buf); left -= len(buf)
                print(f"  copied {k}", flush=True)
            elif kind == "qw":
                # weight and its scale are produced together; the scale tensor is written when its turn comes
                sa, _ = hdr[k.replace(".weight", ".scale")]["data_offsets"]
                t0 = time.time(); done = 0; scales = []
                with open(src_path, "rb") as fsc:
                  while done < rows:
                    n = min(chunk_rows, rows - done)
                    fi.seek(data_start + a + done * 256); raw = torch.frombuffer(bytearray(fi.read(n * 256)), dtype=torch.uint8).view(n, 256)
                    fsc.seek(data_start + sa + done * 8); ssc = torch.frombuffer(bytearray(fsc.read(n * 8)), dtype=torch.uint8).view(n, 8)
                    packed, sc = quantize_rows(raw, ssc); fo.write(packed.numpy().tobytes()); scales.append(sc); done += n
                    if done % (chunk_rows * 16) == 0 or done == rows:
                        print(f"  {k}: {done/rows*100:5.1f}%  {(time.time()-t0):.0f}s", flush=True)
                engram_names["pending_scales"][k.replace(".weight", ".scale")] = torch.cat(scales)
            elif kind == "qs":
                sc = engram_names["pending_scales"].pop(k)
                assert sc.shape[0] == rows, (sc.shape, rows)
                fo.write(sc.numpy().tobytes()); print(f"  wrote recomputed scales {k}", flush=True)
    assert os.path.getsize(dst_path) == 8 + len(hb) + off

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--dst-repo", required=True)
    ap.add_argument("--hf-home", required=True); ap.add_argument("--chunk-rows", type=int, default=2_000_000); a = ap.parse_args()
    src = os.path.abspath(a.src); rev = os.path.basename(src.rstrip("/"))
    repo_dir = os.path.join(a.hf_home, "hub", "models--" + a.dst_repo.replace("/", "--")); dst = os.path.join(repo_dir, "snapshots", rev)
    os.makedirs(dst, exist_ok=True); os.makedirs(os.path.join(repo_dir, "refs"), exist_ok=True)
    open(os.path.join(repo_dir, "refs", "main"), "w").write(rev)
    cfg = json.load(open(os.path.join(src, "config.json")))
    layers = list(cfg.get("engram_layer_ids", [1, 14]))
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    eng_w = {k for k in idx if k.endswith("engram.embed.weight")}; eng_s = {k for k in idx if k.endswith("engram.embed.scale")}
    shards = sorted({idx[k] for k in eng_w | eng_s}); print("engram shards:", shards, flush=True)
    for name in sorted(os.listdir(src)):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if name in shards or name == "config.json" or os.path.exists(d): continue
        if os.path.isdir(s): os.symlink(os.path.realpath(s), d); continue
        try: os.link(os.path.realpath(s), d)
        except OSError: os.symlink(os.path.realpath(s), d)
    for shard in shards:
        print(f"rewriting {shard}", flush=True)
        rewrite_shard(os.path.join(src, shard), os.path.join(dst, shard + ".tmp"), {"weight": eng_w, "scale": eng_s, "pending_scales": {}}, a.chunk_rows)
        os.replace(os.path.join(dst, shard + ".tmp"), os.path.join(dst, shard))
    qc = dict(cfg.get("quantization_config") or {}); qc.update({"engram_embed_quant": "mxfp4", "engram_embed_block_size": 32, "engram_embed_scale_dtype": "ue8m0", "engram_embed_layers": layers})
    cfg["quantization_config"] = qc; json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    print("done ->", dst, flush=True)

if __name__ == "__main__":
    main()
