# MXFP4 Engram tables + NVFP4 experts on kraken TP3 (image 20261019.4) — full write-up

Companion to the 2026-10-31 update section in [README.md](README.md). Patched sources: [patch-nvfp4/](patch-nvfp4/).


Two new drop-ins for machines with less host RAM or disk. Both are config-only on the same image
(`20261019.3`): eight patched Python files mounted over site-packages by `compose.nvfp4.yaml`.

| checkpoint | experts | Engram tables | disk | Engram host RAM (per rank, per layer) | decode @32k c1 | coding peak | `compare_models` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stock `deepseek-ai/DeepSeek-V4.1-Flash-1017` (default) | MXFP4-CSF (lossless) | FP8 | 475 GiB | 31.5 GiB | 79.5 steps/s, 214 tok/s | 336 / 374 | 41/43 |
| **`local/DeepSeek-V4.1-Flash-1017-engram-mxfp4`** (built locally, ~10 min) | MXFP4-CSF (lossless) | **MXFP4 (lossy)** | 475 GiB + 94 GiB (46 shards hard-linked) | **16.2 GiB** | **80.6 steps/s, 220 tok/s** (layer 14 on disk) · **84.4 steps/s with `ENGRAM_RAM_LAYERS=1,14`** | **341 / 374** | **43/43** |
| `nvidia/DeepSeek-V4.1-Flash-1017-NVFP4` | NVFP4 (lossless re-container) | MXFP4 (lossy) | 400 GiB | **16.2 GiB** | **73.8 steps/s, 200 tok/s** (online CSF scales; 62.2 without) | 281 / 288 | 43/43 |

**What NVIDIA's checkpoint changes.** Only the Engram tables are numerically different (FP8 ->
MXFP4 E2M1 nibbles with UE8M0 block-32 scales, round-to-nearest; ~12% relative RMS vs the FP8
tables — the same figure the vLLM PR #61530 reviewer measured). The routed experts are the same
4-bit E2M1 values re-containered as NVFP4 (E4M3 per-16 scales + global scale, lossless), and
the MTP/draft experts, attention, shared experts and head are the stock FP8 tensors
(`ignore: ["*.attn.*", "*.ffn.shared_experts.*", "head", "mtp.*"]`).

**Both Engram tables in RAM.** At 16.2 GiB per rank per layer, `ENGRAM_RAM_LAYERS=1,14` fits where one FP8 layer
used to (host ~149 GB used / 101 GB free on 256 GB) and removes the disk table + row cache entirely. Measured
**84.4 / 84.2 steps/s @32k c1** vs 79-80 with layer 14 on disk — the disk-resident Engram was costing ~5% of decode.
This is the recommended configuration on 192-256 GB hosts; 128 GB hosts keep layer 14 on disk.

**NVIDIA's tables are interchangeable with ours.** A hybrid of stock expert shards + NVIDIA's two Engram shards
(hard-linked, no conversion) measured 84.5 steps/s, coding peak 361/386, eval 42/43 — the same as our converted
tables (84.4, 43/43). The two conversions agree on 100% of block scales and 89% of packed bytes (rounding ties).

**Which Engram tables to use.** Prefer **NVIDIA's** (`make_engram_nvidia_variant.py`: hard-links their two Engram shards
next to the stock expert shards; download just those two files, ~98 GiB total). Our converter
(`make_engram_mxfp4.py`) gives equivalent tables but is a quantization step you run yourself — a wrong scale or
nibble convention produces token salad (we hit exactly that once while writing it). Use it only if you cannot fetch
NVIDIA's files, and verify the result with `compare_models.py`-style checks before serving.

**Why the Engram-only variant exists.** The user-visible win of NVIDIA's checkpoint is the Engram
half (host RAM halved: both Engram layers now fit in RAM for what one costs today; 75 GiB less
disk). The NVFP4 expert half gives nothing on this stack — same weight bytes — and costs ~20%
decode because its E4M3 block scales are not CSF-compressed and stay on-device (9.4 GB less VRAM
before promotion: 16.0 vs 25.4 GB -> 11-12/24 chunks promoted instead of 20/24). So we also ship
a converter that requantizes just the two Engram tables of the stock weights
(`patch-nvfp4/make_engram_mxfp4.py`): identical quantization to NVIDIA's (block scales match on
100% of blocks, packed bytes 89% identical, rest are rounding ties), keeps our lossless experts and
full speed, and needs no 400 GiB download.

**How it works (patches).** Native b12x Engram gather kernel (`flashinfer/.../sequence/engram/
_kernels.py`): `PACKED_FP4` decode (even column in the low nibble; e = bits 2:1, m = bit 0);
plan/state/disk plumbing carries `packed_fp4` (table rows 128 B instead of 256 B); vLLM
`deepseek_v4_1/common/engram.py` reads `quantization_config.engram_embed_quant` and allocates
uint8 tables in all three storage modes (mapped-host RAM, lazy RAM, disk + row cache).
`deepseek_v4_1/quant_config.py` routes routed experts listed in `quantized_layers` to the ModelOpt
NVFP4 method (b12x backend); the dspark draft layers (built as `layers.40-42`, MXFP4 in NVIDIA's
checkpoint) keep the stock MXFP4 method — routing them to NVFP4 leaves `weight_scale_2`/
`input_scale` zero and produces NaN drafts (greedy output still looks right; sampled output is
garbage and spec acceptance reads 100%). `fused_moe/b12x.py`: in-place VMM promotion uses the
layer's actual quant mode and falls back to the canonical `w1_fp4/w2_fp4` tensors (NVFP4 has no
repacked representation). Unit test: `patch-nvfp4/test_packed_lookup.py` (bit-exact decode).

**Knobs.** `DS41_NVFP4_ACTIVATIONS=a16` (default; W4A16 decode, lil's GLM/Qwen NVFP4 recipe) or
`a4` (W4A4; worked but ~10 steps/s here). `B12X_W4A16_A4_PREFILL` must stay 0 with DS41: the
hybrid a4-prefill assumes the target forward's padded token ordering and dies on dspark forwards
(`expected 4091 rows, got 128`).

**Online CSF for NVFP4 scales.** b12x already plans online CSF scale compression for `modelopt_nvfp4` sources
(A4/A16); only the fork's `_online_csf_enabled` gate was MXFP4-only, and the shared scratch had to be handed over as
swizzled `[experts, rows, cols]` views instead of flat bytes. With it, promotion rises from 11-12/24 to 15/24 and
decode from 62.2 to **73.8 steps/s** (`DS41_NVFP4_CSF=1`, default). This is the NVFP4 ceiling on this stack: the
W4A8 (FP8-activation) NVFP4 mode cannot take online CSF, so it would give the VRAM back.

**Not done / follow-ups.** Knowledge-heavy evals of the lossy Engram beyond our 43-item check; a 128 GB-host
configuration test (layer 14 on disk) for the Engram-only variant.
