# DeepSeek-V4.1-Flash-1017 on NVIDIA Flare at TP3 — 20261105

DeepSeek-V4.1-Flash-1017 served by **NVIDIA Flare at tensor parallel 3 with the full 1M-token context** on three
96 GB RTX PRO 6000 Blackwell cards. Flare upstream supports this model at TP1/2/4; this fork adds what TP3 on 96 GB
needs: expert parking with a mixed VRAM/host expert bank, MXFP4 Engram tables held in RAM, vocabulary-parallel
sampling at TP3, and the planner and FlashInfer fixes listed below.

Weights: [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @
`1bae0e1088993b9b82de08f59de5be5a34e42170` (MIT) for all dense and expert weights; the two Engram tables in MXFP4
from [nvidia/DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) (MIT).
Runtime: [NVIDIA Flare](https://github.com/NVIDIA/flare-local-inference) `main` @ `78b18b1c` plus this fork
(branch `ds41-tp3-v2`), FlashInfer @ `ddbe93fc` (the gb202 pin) plus four patches in `tp3-patches/`.

**How to get it:** build the package yourself; there is no prebuilt download. [EXPORT.md](EXPORT.md) walks through
it: download the two public checkpoints (~412 GB), build the fork, export (~40 min on the 3 GPUs). Then:

```bash
FLARE=<fork>/flare/target/release/flare PACKAGE=<package path> ./serve.sh   # port 15015, model DeepSeek-V4.1-Flash, ~60 s
```

Measured on the system in [Test system](#test-system): 350 W/GPU, +6000 MHz GPU memory offset, 256 GB DDR5-6000. Serving uses ~150 GB of host RAM (~100 GB free).

## Headline numbers

| | this build | vLLM Kraken TP3 `.11` (same box) |
| --- | --- | --- |
| KV cache | **2,991,488 tokens** (2.99 × 1M) | 2,185,581 |
| Decode 32K, c1 / c4 / c8 (tok/s, `llm_decode_bench`) | **~254 / ~595 / ~805** | 231-260 / 470-490 / ~790 |
| Decode 32K, c1 engine steps/s | ~94-95 | ~97 |
| Prefill 32K / 128K (tok/s) | **12,548 / 12,160** | 14,195 / 12,716 |
| Coding peak (`--coding-peak`, median) | 379-398 tok/s | — |
| Startup | **~60 s** (Engram tables into RAM) | ~3.5 min |
| eval-43 / tool calls | **43/43** / **60/60** | 42/43 / — |
| lavd / hotel-lights (c4, 30 each) | **30/30** / **30/30** | 30/30 / 30/30 |
| LiveBench coding | **82.9** | 81.9 |

Steps/s is engine speed with draft acceptance divided out; tok/s also depends on how predictable the text is (one
c1 run on this build measured 304.6 tok/s at the same ~93 steps/s). c4 is where Flare leads most.

## Contents

1. [Test system](#test-system) · [What it is](#what-it-is) · [How it fits in 3 × 96 GB](#how-it-fits-in-3--96-gb)
2. [Measurements](#measurements) · [Quality and long context](#quality-and-long-context) · [Memory](#memory)
3. [Measured and not adopted](#measured-and-not-adopted) · [Known limitations](#known-limitations)
4. [Building the package](#building-the-package) ([EXPORT.md](EXPORT.md)) · [Files](#files) · [Licenses](#licenses) · [Credits](#credits)

## Test system

All numbers in this folder come from this machine:

| Part | Spec |
| --- | --- |
| GPUs | 3 × NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB (97,887 MiB), each on PCIe Gen5 x16 |
| GPU settings | **350 W** power limit; **+6000 MHz memory offset** (memory clock 16,365 MHz); ECC off; persistence mode on |
| CPU | AMD Ryzen Threadripper PRO 7965WX, 24 cores / 48 threads, 1 socket, 1 NUMA node |
| Motherboard | ASUS Pro WS WRX90E-SAGE SE |
| RAM | 256 GB DDR5: 8 × 32 GB Kingston Fury (KF560R32-32), 8 channels at **6000 MT/s** |
| Package drive | Crucial T710 2 TB, PCIe Gen5 x4: **12 GB/s** direct read |
| Checkpoint drive | Samsung 9100 PRO 2 TB, PCIe Gen5 x4: **12 GB/s** direct read |
| Software | Ubuntu, NVIDIA driver 615.71.09, CUDA 13.1 |

What matters most for speed, in our measurements:

- **GPU memory clock.** Decode reads weights every step, so memory bandwidth sets the per-step time; the +6000
  offset is part of every number here. A user on a similar build at stock memory clock (13,365 MHz) and PCIe Gen4
  measured roughly half our per-step speed on the vLLM build, mostly from memory clock and slower GPU-to-GPU links.
- **PCIe generation and topology.** The three GPUs exchange data twice per layer (all-reduce over PCIe; there is
  no NVLink), and cold experts stream from host RAM over PCIe. Gen5 x16 per GPU on one CPU root complex is the
  tested setup.
- **Power.** Measured at 350 W. Higher limits mainly help prefill (+~10% at 450 W on the vLLM build); decode at c1
  barely changes.
- **Host RAM.** The Engram tables (~102 GiB) and parked experts (~13 GiB) live in RAM. With less RAM, serve with
  `ENGRAM_RAM=0`: tables are then read from the drive through the Linux page cache, and fast NVMe matters.
- **Drives.** Startup reads ~380 GiB of package (about 60 s here); export writes the same amount.

## What it is

| Layer | Source |
| --- | --- |
| Runtime | Flare `main` @ `78b18b1c` (Ignition prepares packages, Helios serves them) plus the TP3 changes on branch `ds41-tp3-v2` |
| Kernels | FlashInfer @ `ddbe93fc` b12x plus `tp3-patches/` (see below) |
| Package | exported locally with `export.sh` (~380 GiB, self-contained: weights, Engram tables, tuned kernels, graphs) for 3 × RTX PRO 6000 96 GB (SM120), max concurrency 8, 1M tokens/request |
| Weights | original DeepSeek checkpoint; Engram tables from NVIDIA's MXFP4 release |

Flare exports a **package**: weights laid out per GPU, kernels autotuned, CUDA graphs captured. Serving loads
that package, which is why startup takes about a minute instead of minutes of compilation. The package is tied to
the hardware and settings it was exported with.

## How it fits in 3 × 96 GB

**Merged hot/cold expert bank.** Routed-expert layers 20-33 keep their **300 most-routed experts in VRAM** and the
other 84 in pinned host memory, in **one contiguous virtual range per layer**: a VRAM prefix and a host-NUMA tail,
mapped with CUDA VMM. One MoE kernel call per layer reads both, so a cold expert's PCIe reads overlap the hot
experts' compute inside the kernel. The earlier design (separate hot and cold banks, two calls) cost 4-7% at c4/c8;
running the two calls on separate streams gives zero overlap, because the hot kernel occupies every SM. Expert IDs
are remapped hot-first with route counts recorded from real traffic (0.6 code / 0.4 prose blend).

**MXFP4 Engram tables in RAM.** DeepSeek V4.1's two Engram n-gram tables are 189 GiB in FP8, too big to keep in
RAM next to everything else. NVIDIA's MXFP4 versions are 97 GiB: packed E2M1 pairs with the same E8M0 scales. This
build adds an `MXFP4_E2M1_E8M0` table encoding and a packed-FP4 path in the b12x Engram lookup kernel (it decodes
bit-exact against a reference decode on real rows), and keeps both tables fully in host RAM. Decode no longer waits
on table reads: Engram wait drops from 3-4.5% of each step to ~0.1%.

**Vocabulary-parallel sampling at TP3.** Each GPU samples and verifies over its own third of the vocabulary instead
of gathering full logits, which frees ~1 GiB per GPU for KV. Upstream gates this to TP2/4. Side effect: it has no
top-p filter, so the package fixes top_p at 1.0 (the model's default); requests asking for lower top_p are clamped
instead of failing.

**Prefill in 4096-token chunks.** Larger prefill chunks are +15-16% prefill at 32K-128K over 2048. They cost KV,
which is why the hot-expert count is 300 rather than 313-326: with the merged bank, hot-expert count barely moves
decode speed (313 vs 326 within noise).

**Speculative decoding.** DSpark, the checkpoint's 3 `mtp.*` layers: 5 drafts, NVFP4 draft head, FP8 Markov head,
Flare's adaptive verification (alpha 0.6, budget 128).

**Fixes required for TP3 at all:**

| Fix | Where |
| --- | --- |
| FlashInfer `tiny_decode` W4A8 MoE kernel ignored `swiglu_limit`; experts above DeepSeek's SwiGLU clamp blew up a layer's output and produced one junk token mid-reply (flare#193-style). Root-caused with per-layer probes and replay | `tp3-patches/flashinfer-tiny-decode-swiglu.patch` |
| PCIe DCP all-to-all and tuning tables lacked world size 3 | `tp3-patches/flashinfer-dcp-a2a-world3.patch`, `flashinfer-tuning-world3.patch` |
| Planner capped context at 356K because it counted parked experts as VRAM | `ignition/recipes/declaration.py` |
| One shared PCIe all-reduce runtime for prefill and decode (hidden size 5120 is not divisible by 3, so DMA reduce-scatter cannot be used) | `ignition/recipes/deepseek41/bootstrap.py` |
| Vocabulary-parallel kernels and gathers at TP3 | `ignition/planning.py` (opt-in `FLARE_DS41_TP3_VOCAB_PARALLEL=1`) |

## Measurements

All with `llm-inference-bench` v0.9.1 (Flare detected from metrics), 350 W, Engram in RAM, on this package.

| Benchmark | Result |
| --- | --- |
| Decode 32K, c1 / c4 / c8 | 253.8 / 595.0 / 804.5 tok/s (93.8 / 62.0 / 48.1 steps/s) |
| Decode 32K, c1, separate run | 304.6 tok/s at 93.2 steps/s (3.27 tokens/step) |
| c1 steps/s at 8K / 32K / 128K | 96.3 / 95.2 / 92.6 |
| Prefill 8K / 32K / 128K | 10,977 / 12,548 / 12,160 tok/s (TTFT 0.75 / 2.57 / 10.57 s) |
| Coding peak (5 runs, median) | 378.8 tok/s (range 350-417) |

**Alternative export** (`HOT=313 PREFILL=2048 ./export.sh`): 313 hot experts, 2048-token prefill chunks, **3.47M KV**. Decode ~255 / ~615 / ~819 tok/s, prefill 32K / 128K 10,842 / 10,611 tok/s. Pick it for more
simultaneous long sessions.

## Quality and long context

Same protocol as the vLLM Kraken TP3 releases (`qualification/`): thinking on, `reasoning_effort` max, temperature
1, top-p 1.0, on the running package without restarts. Flare deltas: eval-43 runs at temperature 1 (Flare fixes
DeepSeek's temperature at 1.0; `.11` used 0); prompt sizing uses `/v1/messages/count_tokens` (Flare has no
`/tokenize`).

| Check | Result |
| --- | --- |
| eval-43 (short reasoning/knowledge, thinking off) | **43/43** |
| Tool calls: 60 tool-requiring prompts at c4, argument validity, second turn | **60/60** |
| lavd, 30 runs at c4 | **30/30** exact (0 wrong, 0 truncated) |
| hotel-lights, 30 runs at c4 | **30/30** exact (0 wrong, 0 truncated) |
| Long retrieval: three markers at 128K and 900K tokens | **pass / pass** |
| Long generation after a 581,537-token conversation, 4 seeds | **4/4**, natural stop, 17.6K-22.5K tokens each, 8-gram repetition ≤ 0.0013 |
| Decode bench with natural EOS (loop check) | **0 loops**; 250.0 / 587.8 / 793.7 tok/s (c1/c4/c8) |
| LiveBench coding (128 questions) | **82.9** (LCB generation 71.8, coding completion 94.0) |
| KV stress: 2 waves × 8 × ~530K unique prompts, KV oversubscribed, plus pinger | **PASS**: 16/16 long, pinger 137/137, 9.5M prompt tokens in 17.2 min, health 200, 0 error log lines |
| Corruption check (teacher-forced comparison, 10 runs) | 0 true corruptions, on every build |

## Memory

| Per GPU | |
| --- | --- |
| Loaded (weights, graphs, scratch) | ~92.2 GiB |
| KV cache | 7.9 GiB → 2,991,488 tokens |
| Host RAM | Engram tables ~102 GiB, parked experts ~13 GiB, prefix cache 16 GiB |

## Measured and not adopted

| Idea | Result |
| --- | --- |
| Hot and cold expert calls on separate streams | 0% overlap: the hot kernel fills every SM |
| Global hot-expert placement across all 40 layers | c4 +3%, c8 −2 to −7% |
| FP8 Engram tables in RAM | +2-5% decode, but 189 GiB leaves ~14 GiB free; one table in RAM is −9% (the other loses the page cache) |
| Larger Engram row cache (48 GiB) | no change; misses are first-touch rows |
| History drafting (prompt-lookup) for DeepSeek | correct, but −12-20% including coding peak: Helios's history scheduling adds ~15% GPU wait |
| Runtime collective tuning | no gain, −0.75 GiB/GPU |
| `--joint-tuning` | does not fit in memory at TP3 |
| `--mla-compute-mode auto` | slower |
| 313 / 326 / 310 hot experts | decode within noise of 300; costs KV |
| Prefill-scratch shrink (#186) | ~220 MiB: too small to matter |

## Known limitations

- **Hardware-specific package:** 3 × 96 GB SM120 (RTX PRO 6000) only, max concurrency 8. The all-reduce tuning came
  from this box's PCIe layout. Other setups need a fresh export (below).
- **top_p is fixed at 1.0** (vocabulary-parallel sampling); temperature is fixed at 1.0 by Flare's DeepSeek recipe.
- **Host RAM:** Engram-in-RAM needs ~102 GiB free. With less RAM, drop `FLARE_ENGRAM_RAM_TABLES=0`; tables are then
  read from disk through the page cache, about 3-5% slower at c1.
- **Prefill** trails vLLM `.11` by ~12% at 32K and ~4% at 128K.
- **Bench loops:** `llm_decode_bench` decode cells default to `ignore_eos`, which forces generation past the answer;
  occasional "Final. Done." repetition loops there are a bench artifact: with `--respect-eos` the same cells show 0 loops.
- Not upstreamable as written: Flare's contribution rules exclude model-specific offload policy and AI-authored
  GPU-execution changes. This is a personal fork.

## Building the package

Full step-by-step guide: **[EXPORT.md](EXPORT.md)** (toolchain, building the fork, assembling the checkpoint from
the two public releases, export settings and variants, checks, troubleshooting). In short:

```bash
./make_checkpoint_view.sh <deepseek snapshot> <nvidia NVFP4 snapshot> ckpt/   # 307 GB + 105 GB of downloads
CKPT=ckpt/ ./export.sh                                                       # ~40 min on 3 GPUs, cold cache
```

## Files

| Path | What |
| --- | --- |
| `EXPORT.md` | how to build the package yourself |
| `serve.sh` / `export.sh` | serve the package (Engram in RAM) / export it from a checkpoint view |
| `make_checkpoint_view.sh`, `checkpoint/` | build the checkpoint from the two public releases (config + corrected index) |
| `source.patch` | the fork's changes against Flare `78b18b1c` (32 files) |
| `expert-order/` | per-layer route counts that set the hot-expert placement |
| `tp3-patches/` | the four FlashInfer patches |
| `tools/` | `bench_steps.py` (steps/s from bench JSON), `verify_fix.py` (corruption check), `tool_calls.py` |
| `qualification/` | the qualification suite (adapted for Flare) and its results summary |

## Licenses

DeepSeek weights and NVIDIA's NVFP4 release: MIT (texts in `licenses/`). Flare and FlashInfer: Apache-2.0 (LICENSE, NOTICE and
`THIRD_PARTY_NOTICES.md` kept; modified files marked). The `flare` binary links only libc/libstdc++ and loads
`libcuda` from the host driver; NVIDIA's CUDA Python wheels are used only for exporting and are not redistributed.

## Credits

NVIDIA Flare and FlashInfer b12x; DeepSeek for the model; NVIDIA for the MXFP4 Engram tables; Local Inference Lab for
`llm-inference-bench`.
