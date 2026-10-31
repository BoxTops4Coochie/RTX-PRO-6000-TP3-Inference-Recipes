"""Standalone check of the packed-MXFP4 Engram decode used in the patched b12x _lookup kernel.
Builds a random packed table (uint8 [rows,128]) + E8M0 scales, runs the exact Triton decode
expression, and compares against a torch reference (E2M1 LUT * 2^(e8m0-127)). Run inside the
serving container:  docker exec ds41-kraken-tp3 /opt/venv/bin/python /tmp/test_packed_lookup.py"""
import torch, triton, triton.language as tl

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

@triton.jit
def _decode(weight, scales, out, ROWS: tl.constexpr):
    r = tl.program_id(0)
    col = tl.arange(0, 256)
    byte = tl.load(weight + r.to(tl.int64) * 128 + (col // 2).to(tl.int64)).to(tl.int32)
    nib = tl.where(col % 2 == 0, byte & 0xF, (byte >> 4) & 0xF)
    e = (nib & 0x7) >> 1
    m = (nib & 0x1).to(tl.float32)
    pow2 = ((e - 2 + 127) << 23).to(tl.float32, bitcast=True)
    mag = tl.where(e == 0, m * 0.5, (2.0 + m) * pow2)
    quant = tl.where((nib & 0x8) != 0, -mag, mag)
    exponent = tl.load(scales + r.to(tl.int64) * 8 + (col // 32).to(tl.int64)).to(tl.uint32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    scale = tl.where(exponent == 0, 2.0**-127, scale)
    tl.store(out + r.to(tl.int64) * 256 + col, (quant * scale).to(tl.bfloat16))

def reference(packed, scales):
    lo = (packed & 0xF).long(); hi = ((packed >> 4) & 0xF).long()
    nib = torch.stack([lo, hi], -1).reshape(packed.shape[0], 256)          # even col = low nibble
    mag = E2M1.to(packed.device)[nib & 0x7]
    val = torch.where((nib & 0x8) != 0, -mag, mag)
    sc = torch.ldexp(torch.ones_like(val), (scales.long() - 127).repeat_interleave(32, dim=1))
    return (val * sc).to(torch.bfloat16)

rows = 4096
g = torch.Generator(device="cuda").manual_seed(0)
packed = torch.randint(0, 256, (rows, 128), dtype=torch.uint8, device="cuda", generator=g)
scales = torch.randint(100, 140, (rows, 8), dtype=torch.uint8, device="cuda", generator=g)
out = torch.empty(rows, 256, dtype=torch.bfloat16, device="cuda")
_decode[(rows,)](packed, scales, out, ROWS=rows, num_warps=4)
ref = reference(packed, scales)
exact = torch.equal(out, ref)
print("packed-FP4 decode vs reference: exact match =", exact, "| max abs diff =", (out.float()-ref.float()).abs().max().item())
# sanity: all 16 nibble codes decode to the expected E2M1 values at scale 2^0
codes = torch.arange(16, dtype=torch.uint8, device="cuda")
p = (codes[1::2] << 4 | codes[0::2]).repeat(16)                       # 8 bytes -> 16 nibbles, repeated to 128 B
tbl = p.view(1, 128); sc = torch.full((1, 8), 127, dtype=torch.uint8, device="cuda"); o = torch.empty(1, 256, dtype=torch.bfloat16, device="cuda")
_decode[(1,)](tbl, sc, o, ROWS=1, num_warps=4)
print("codes 0..15 ->", o[0, :16].float().tolist())
assert exact, "MISMATCH"
print("OK")
