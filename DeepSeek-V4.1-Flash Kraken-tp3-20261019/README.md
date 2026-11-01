# DeepSeek-V4.1-Flash-1017 Kraken TP3 — 20261019.7

DeepSeek-V4.1-Flash-1017 served at **tensor parallel 3 with the full 1M-token context** on three 96 GB RTX PRO 6000
Blackwell cards, from DeepSeek's original weights (FP8 dense layers, MXFP4 routed experts; nothing requantized) or,
recommended, the same weights with NVIDIA's 4-bit Engram tables so both tables fit in host RAM.

Weights: [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @
`1bae0e1088993b9b82de08f59de5be5a34e42170` (MIT); Engram tables from
[nvidia/DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) (see
[README-engram-nvfp4.md](README-engram-nvfp4.md)). Runtime: Local Inference Lab's vLLM "karmic-kraken" image
(`karmic-kraken-beta-20261016.2`, vLLM `6d66eed35`) with FlashInfer `main` b12x and the TP3 changes in `src/`.

Image: **`azallaza/ds41-kraken-tp3:20261019.7`** (`sha256:27a95525e02f4febcaef375ad5c1e8305ba7ad52dc89283f149170df329efc7b`). Every tag since `20261019` shares all but a
few small layers with the previous one, so pulling over an earlier tag downloads little. Older tags stay valid:
`.6` (no vocab-split sampling), `.5` (no BRRR kernels), `.4` (no cap-only adaptive verification), `.3` (stock Engram
tables only). The development history with per-step measurements is kept in [README-history.md](README-history.md).

```bash
docker pull azallaza/ds41-kraken-tp3:20261019.7
# one-time: hard-link NVIDIA's Engram shards next to the stock weights (both repos in the same HF cache)
python3 patch-nvfp4/make_engram_nvidia_variant.py --cache "$HF_CACHE_DIR"
HF_CACHE_DIR=/path/to/huggingface docker compose up -d      # port 15015, model name DeepSeek-V4.1-Flash
docker logs -f ds41-kraken-tp3 2>&1 | grep -m1 'expert promotion armed and warmed'   # then send traffic
```

Host settings: 350 W/GPU (450 W raises prefill ~10%, decode unchanged), +6000 memory offset, 256 GB host RAM, the
checkpoint on a local NVMe drive.

## Contents

1. [What it is](#what-it-is) · [How it fits in 3 × 96 GB](#how-it-fits-in-3--96-gb)
2. [What is in this image](#what-is-in-this-image)
3. [Measurements](#measurements-on-this-image) · [Quality and stress](#quality-and-stress)
4. [Presets and switches](#presets-and-switches) · [Alternative weights](#alternative-weights)
5. [Measured and not adopted](#measured-and-not-adopted) · [Known limitations](#known-limitations) · [Not tested](#not-tested)
6. [Files](#files) · [Variables](#variables) · [compose.yaml](#composeyaml) · [Credits](#credits)

## What it is

| Layer | Source |
| --- | --- |
| Base image | `ghcr.io/local-inference-lab/vllm@sha256:f5f9efe1…` (karmic-kraken-beta-20261016.2, vLLM `6d66eed35`, CUDA 13.4, PyTorch 2.14) |
| b12x kernels | FlashInfer `main` @ `5b065abe`: compact-scale MoE decode/prefill, inline CSF scale decoding, MoE race fixes. Python only, compiled at startup |
| TP3 changes | 16 vLLM files in `src/` (`source.patch` against the base image); 8 more for the 4-bit Engram / NVFP4 support (`patch-nvfp4/`), 1 for cap-only adaptive verification (`patch-rowcap/`), 3 + the `brrr` package for the BRRR kernels (`patch-brrr/`) |
| Weights | the original `deepseek-ai/DeepSeek-V4.1-Flash-1017`; by default the `local/…-engram-nvidia` variant = the same shards with NVIDIA's two 4-bit Engram shards hard-linked in |

The base image already pads DS4.1's 64 attention heads to 72 (8 → 9 output groups) at TP3. What does not fit is
memory: the original checkpoint needs ~95 GiB per GPU before KV, caches and activations.

## How it fits in 3 × 96 GB

**Expert layers parked in host RAM, hottest experts promoted back to GPU.** Routed-expert layers 20-33 (of 40) start
in host-backed CUDA virtual memory (HOST_NUMA pages behind a device address range), so startup fits. Once the server is
healthy, `ds41-serve` arms promotion and sends a warm-up request; on that pass each parked layer remaps 16-expert chunks
of its weights to GPU memory **at the same virtual address** (captured CUDA graphs stay valid) until 1.5 GiB per GPU
remains free. With the default KV pool that is 20 of 24 chunks of the gate/up weights and 19 of 24 of the down weights.

**Hot-first expert order from real traffic.** At load, each parked layer's experts are relabeled most-routed-first
(router rows and expert weights together, so outputs are unchanged). The order (`expert-order-blend/`) was counted on
~650K routed tokens, 0.6 coding-agent traffic + 0.4 prose, so the promoted prefix holds the experts both workloads use:
~98.5% of routes hit a promoted expert and the parked layers run their MoE kernel at resident speed (~100 µs vs ~95).
A code-only order was fast on code and collapsed on prose; the sample must cover every workload served
([README-tuning.md](README-tuning.md) has the procedure to count your own).

**Compressed expert scales.** All routed-expert scales are compressed losslessly at load with FlashInfer's per-row
CSF encoder and decoded inline by the MoE kernels, which is what leaves room for promotion.

**Engram.** Two n-gram tables (layers 1 and 14). With the stock weights they are 94 GiB each: layer 1 pinned in host
RAM, layer 14 read from NVMe behind a 2 GiB-per-GPU host row cache. With NVIDIA's MXFP4 tables (16.2 GiB per rank
each) both sit in host RAM (`ENGRAM_RAM_LAYERS=1,14`): +6% decode and no disk reads while serving. The lookup kernel
decodes the packed 4-bit rows directly (`patch-nvfp4/`).

**Decode.** DSpark speculative decoding (the checkpoint's 3 `mtp.*` layers), 5 drafts, block rejection sampling.
Verification is **cap-only adaptive**: every draft is verified while the decoding requests' target rows fit under
`DS41_VERIFY_ROW_CAP=24` (up to 4 requests), and above that drafts are trimmed best-confidence first across requests,
which is what keeps c8 fast without costing a lone user anything. Draft sampling and verification are **split by
vocabulary shard** across the three GPUs, so only a few words per step cross PCIe instead of the logits.

**Collectives.** b12x PCIe one-shot all-reduce for the small decode all-reduces; BRRR's phased PCIe all-reduce /
all-gather (rebuilt for 3 ranks) for the 16 KB..1 MB messages that used to fall back to NCCL ring.

## What is in this image

Everything below is built in; `compose.yaml` only mounts the launcher settings and the expert order. Effects are
`llm-inference-bench` 32K c1 unless noted, measured step by step as each change landed (details per step in
[README-history.md](README-history.md)).

| Change | Effect |
| --- | --- |
| TP3 port (padded heads/vocab/sinks, in-place VMM expert promotion, CSF scales) | the model runs at TP3 with 1M context; 226 tok/s short-prompt decode on `20261019` |
| `max-parallel-prefills: auto` | a short request behind a 420K-token prefill: TTFT 34 s → 0.6 s |
| NVFP4 drafter head, top-p 1.0 server default | model-card sampling; drafter head VRAM |
| Parked layers 20-25 → 20-33 with a 1.5 GiB reserve | +6% decode (more experts promoted per layer), KV-stress clean |
| Expert order counted on real traffic (blend) | +10% decode: 72 → 79.5 steps/s |
| NVIDIA 4-bit Engram tables, both in RAM | +6% decode: 79.5 → 84 steps/s; host RAM 150 → ~97 GB free |
| `max-num-seqs` 4 → 8 (graphs 48) | c8 611 tok/s aggregate; c8 KV stress PASS |
| Cap-only adaptive verification (`DS41_VERIFY_ROW_CAP=24`) | c6 512 → 635, c8 611 → 752 tok/s aggregate; c1/c4 unchanged |
| BRRR skinny FP8 GEMV + radix indexer top-k | 83 → 85.3 steps/s, coding mean 336 → 354 tok/s |
| BRRR phased PCIe all-reduce (3 ranks) | 85.3 → 86.5 steps/s, coding mean 365 tok/s |
| BRRR vocab-split draft sampling + verification (TP3 padding-aware) | 87.5 steps/s, 238 tok/s, c4 475 / c8 776 aggregate |
| BRRR renderer fix | tool-result-only messages no longer start a new turn (reasoning kept inside agent tool loops); parallel tool results ordered by call |
| NVFP4 expert support (`nvidia/…-NVFP4` as a drop-in) | works, 73.8 steps/s on this stack (ceiling: scale layout + W4A16); kept as an option |

## Measurements on this image

All at 32K context, image `.7`, `local/…-engram-nvidia` weights, **350 W/GPU, +6000 memory offset**, two runs per preset
(ranges below), with [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench):

```bash
python3 llm_decode_bench.py \
    --port 15015 \
    --model DeepSeek-V4.1-Flash \
    --concurrency 1,8 \
    --contexts 32k \
    --duration 30 \
    --coding-peak
```

steps/s is MTP-normalized engine steps; tok/s moves with the draft acceptance of each sampled run (temperature 1), steps/s
is the stable number. Each KV preset is one `docker compose` overlay (below); the KV comes out of expert promotion, so
more KV = fewer promoted experts.

| preset (`./kv.sh`, `kv-<preset>.yaml`) = x 1M-token sessions | KV / GPU | capacity (tokens) | 1M sessions | promoted w13 / w2 | c1 steps/s | c1 tok/s | c8 steps/s | c8 aggregate tok/s | coding mean / max tok/s | c4 aggregate (per user)* |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| **2x (default)** | 2.25 GiB | 2,185,581 | 2.1 | 19/24 + 19/24 | **87.4-87.9** | **238-261** | 348.8-356.2 | **771-782** | 381-388 / 409-416 | 481 (120) |
| 4x | 4.75 GiB | 4,614,187 | 4.4 | 17/24 + 17/24 | 84.4-85.8 | 228-242 | 321.2-338.9 | 713-742 | 339-342 / 359-371 | 435 (109) |
| 6x | 6.5 GiB | 6,314,130 | 6.0 | 16/24 + 15/24 | 80.4-81.2 | 207-211 | 304.4-315.0 | 693-704 | 316-328 / 330-339 | 398 (99) |
| 8x | 8.75 GiB | 8,499,917 | 8.1 | 14/24 + 13/24 | 74.8-75.0 | 198-207 | 274.4-278.0 | 595-598 | 280 / 292-313 | 341 (85) |

\* c4 is a single run at 450 W on the same image (`qualification/brrr-tp3/matrix7/`); the 450 W runs also gave
prefill at 32K of 14,929 / 14,641 / 14,331 / 13,846 tok/s for 2x / 4x / 6x / 8x (TTFT ~2.2 s). Power only affects
prefill (~+10% at 450 W); decode is weight-read-bound and identical at 350 W. Real source code prefills at 10-11K tok/s
and a 983K-token needle at ~7.1K tok/s (`bench/`).

Reading the presets: `6x` still promotes ~256 of 384 experts per parked layer, the set that carries 98-99% of routes;
`8x` cuts into it (~224), so its cost per added 1M tokens is the steepest (c1 -6%, c8 -8%, coding -15% vs `6x`). Use
`8x` when eight concurrent 1M-token sessions are really needed; spare KV beyond the running sessions only feeds prefix
caching.

## Quality and stress

Correctness set run on each shipped change (same protocol as the GLM Kraken TP3 releases: thinking on,
`reasoning_effort` max, temperature 1 unless noted):

| Test | Result on `.7` (BRRR + vocab-split) | Earlier tags |
| --- | --- | --- |
| 43 verifiable-answer questions, thinking off, temperature 0 (`compare_models.py`) | 41/43 (the two borderline items) | 42/43 (`.4`-`.6`), 41-42 stock |
| LiveBench coding (release 2024-11-25, 32K budget) | **83.2** (LCB 72.7 / completion 94.0) | 84.2 (`.6`), 81.9 (`.4`/`.5`), 81.3 stock Engram |
| hotel-lights, 30 runs at 4 concurrent, seed 103 | **30 exact** / 0 wrong / 0 truncated (mean 21,125 tokens) | 30/30 on every tag |
| lavd, 30 runs at 4 concurrent | — | 30 exact (`.3`) |
| Retrieval of three markers at 128K / 900K | — | pass / pass (`.3`) |
| Long generation after a 581K-token conversation, 4 seeds | — | 4/4 clean, worst 8-gram repeat 0.0005 (`.3`) |
| Vision basic / hard (60 + 60 images) | — | 60/60, 117/120 (`20261019`) |
| Coding task (`qualification/coding-task/`) | — | 18/18 tests passed (`.3`) |

Stress: 8 concurrent *unique* ~530K-token prompts in waves until >= 1.15x the KV pool has passed through (forces
eviction), with a short-prompt pinger throughout; min free VRAM sampled every 2 s; container log scanned for OOM /
mapping-failure / preemption / cuBLAS lines.

| Stress | Result |
| --- | --- |
| `.7` (vocab-split), 2x pool, c8 | **PASS**: 16/16 long, 262/262 pings, 9.56M prompt tokens, min free VRAM 494 MiB, 0 bad log lines (39 min) |
| `.6` (BRRR kernels), 2x pool, c8 | PASS: 16/16 long, 136/136 pings, 9.56M prompt tokens, min free 543 MiB |
| `.5` (cap-only adaptive), 2x pool, c8 | PASS: 16/16, 127/127, 9.56M tokens, min free 593 MiB |
| `.4`, 2x / 4x / 6x / 8x pools, c4 | all PASS (min free 517-537 MiB; the 1.5 GiB reserve, not the KV size, governs stability) |
| 3-rank phased all-reduce extension (`patch-brrr/test_phased.py`) | 132 collectives bit/tolerance-equal to torch references |

Results and logs: `qualification/`.

## Presets and switches

**KV presets** (`./kv.sh 2x|4x|6x|8x`, or `docker compose -f compose.yaml -f kv-<preset>.yaml up -d --force-recreate`):
names are 1M-token sessions. `max-num-seqs` is 8, so `6x`/`8x` mean genuinely concurrent 1M-token sessions; spare KV
beyond the running sessions feeds prefix caching (a re-sent long conversation skips its prefill). Measured above.

**7-draft coding preset** (`docker compose -f compose.yaml -f k7.yaml up -d --force-recreate`): 7 DSpark drafts
with the row cap at 32. Higher coding peak (351 / 400 tok/s mean / max on `.5`) but slower on prose (c1 192 vs ~220,
c4 369 vs 460): every step does 8 rows of MoE routing instead of 6. For single-user coding sessions only.

**Adaptive verification**: `DS41_VERIFY_ROW_CAP=24` (default), `32` (5 requests untouched), `0` = upstream's
cost-model adaptive (trims a lone user to ~2 drafts: c1 -11%). `adaptive-verification: false` in `ds41-launch.yaml` =
every draft verified at any concurrency (c8 611 instead of 752).

**BRRR pieces**: `VLLM_DSPARK_SHARD_MARKOV_HEAD=0 VLLM_DSPARK_REPLICATE_MARKOV_W1=0` turns the vocab-split sampler off
(= `.6`); `BRRR_TP3_LOG=1` logs once per worker which kernels engaged.

**Expert order**: `B12X_EXPERT_ORDER_DIR=/opt/ds41/expert-order` = the image's baked order (an 18K-token sample,
~15% cold routes); `B12X_EXPERT_ROUTES_ALL=1` dumps route counts to build your own ([README-tuning.md](README-tuning.md)).

## Alternative weights

Any checkpoint that keeps the original FP8 layout (same tensor names, dtypes, shapes and index) is a drop-in via
`MODEL`. Verified on this recipe: the stock weights (`deepseek-ai/DeepSeek-V4.1-Flash-1017`, set `ENGRAM_RAM_LAYERS=1`),
`mrfakename/DeepSeek-V4.1-Flash-1017-abliterated` and `dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8` (both
third-party, not audited by us; lavd/hotel-lights 30/30, retrieval pass, 41-43/43 on the question set),
`nvidia/DeepSeek-V4.1-Flash-1017-NVFP4` (NVFP4 experts: 73.8 steps/s, eval 43/43), and the recommended
`local/DeepSeek-V4.1-Flash-1017-engram-nvidia` (stock weights + NVIDIA's Engram shards, made by
`patch-nvfp4/make_engram_nvidia_variant.py`). Details, download commands and the A/B tables:
[README-history.md](README-history.md) and [README-engram-nvfp4.md](README-engram-nvfp4.md).

## Measured and not adopted

| Change | Result |
| --- | --- |
| LMCache host-RAM prefix tier (`cache-mode: lmcache`, 48 or 16 GiB) | host thrash and OOM-killed workers: the engine-driven DS4.1 mode takes ~80 GB of host memory regardless of tier size, which the Engram tables already occupy |
| BRRR sparse decode attention | tiles 16 heads; at our 24 padded heads per rank it runs (after resizing to the sm_120 99 KB shared-memory limit) but produces wrong output |
| BRRR fused decoder layer (one persistent kernel per layer incl. both all-reduces) | reads expert scales directly; this stack keeps them CSF-compressed and decoded inline, and running without CSF costs ~4.5 GB/GPU, more than the whole promoted set |
| Upstream adaptive verification (cost model, any cost scale) | c8 +19-21% but c1 -11..-28%: a 6-row verify step really costs ~25% more than a 3-row one here, so a lone user gets trimmed to 2 drafts |
| 7 drafts as the default | prose c1 192 vs 216-228, c4 369 vs 460 (coding peak +4-7%): kept as the `k7.yaml` preset |
| 6-aligned CUDA-graph capture sizes | neutral (padded rows are free) |
| EP all-reduce through the b12x PCIe path | neutral for decode (not on the steady decode path) |
| Host-CPU expert compute (KTransformers/Fiddler style) | ~10x DDR5-vs-GDDR7 bandwidth gap: one CPU expert costs a whole GPU MoE layer |
| Prefill chunks of 8192 / reserve 768 MiB / parking 20-29 / 4-6-7 drafts / indexer chunk 256 / MoE-only tuning | see history: OOM, no gain, or slower |

## Known limitations

- Promotion happens on the warm-up request after `/health`; requests before the `expert promotion armed and warmed`
  log line run with the parked layers in host memory (slower decode).
- Up to 8 concurrent sequences (`max-num-seqs: 8`, CUDA graphs up to 48 decode rows).
- Time to first token is ~0.25 s even for short prompts (per-request host overhead of the eager prefill pass).
- Host RAM: ~150 GB in use while serving; a boot transiently dips to ~18 GB available. Do not run other large
  host-memory consumers (and remove any leftover `/dev/shm/lmcache_*` pools) before starting.

## Not tested

Benchmark accuracy scores beyond LiveBench coding, video inputs, other GPU power limits, more than 8 concurrent
requests (the configured limit), the stock (FP8) Engram tables on the BRRR tags (unchanged code paths).

## Files

- `Dockerfile`: base image + FlashInfer `main` @ `5b065abe` + `src/`; `ds41-serve` (entrypoint); `ds41-launch.yaml`
  (lil launcher settings, mounted by `compose.yaml`: 8 slots, 4096-token chunks, graphs 48, 5 drafts, cap-only
  adaptive verification); `source.patch` (the 16 TP3 vLLM files); `build.log`, `image-inspect.json`.
- `patch-nvfp4/` (`.4`: 4-bit Engram lookup + NVFP4 experts, `make_engram_nvidia_variant.py`, converters, tests;
  write-up `README-engram-nvfp4.md`), `patch-rowcap/` (`.5`: cap-only adaptive verification),
  `patch-brrr/` (`.6`/`.7`: the BRRR port — `brrr/` package incl. kernels, `brrr.pth` hook, `vllm-patched/`,
  `stage1.patch`, `test_phased.py`, `image/Dockerfile`, `LICENSE-brrr`, `NOTICE-brrr`).
- `compose.yaml`, `.env.example`, `kv.sh` + `kv-2x/4x/6x/8x.yaml`, `k7.yaml` + `ds41-launch-k7.yaml`,
  `expert-order-blend/` (the shipped order), `expert-order/` (the image's baked order), `compose.nvfp4.yaml` (dev overlay).
- `qualification/`: `run_qualification.py` and the GLM-derived checks, `vision/`, `coding-task/`, `kv-presets-stress/`
  (harness + per-preset results), `engram-nvidia/` (hotel-lights + eval on `.4`), `brrr-tp3/` (stress, eval, LiveBench
  answers, hotel-lights on `.6`/`.7`), `RESULTS.md`.
- `bench/`, `bench-20261019.3/`: the early prefill/needle/decode measurements and their scripts.
- `README-history.md`: the previous README with every update section; `README-tuning.md`: expert-order procedure.

## Variables

| Variable | Default | Values |
| --- | --- | --- |
| `HF_CACHE_DIR` | required | Hugging Face cache containing the weights |
| `MODEL` | `local/DeepSeek-V4.1-Flash-1017-engram-nvidia` | weights repo to serve (see [Alternative weights](#alternative-weights)) |
| `ENGRAM_RAM_LAYERS` | `1,14` | Engram layers held in pinned host RAM; `1` with the stock 94 GiB tables |
| `KV_CACHE_MEMORY_BYTES` | `2415919104` (2x) | KV bytes per GPU; presets 2x/4x/6x/8x via `./kv.sh` or `kv-<preset>.yaml` |
| `DS41_VERIFY_ROW_CAP` | `24` | cap-only adaptive verification row ceiling; `0` = upstream cost-model adaptive |
| `VLLM_DSPARK_SHARD_MARKOV_HEAD` / `VLLM_DSPARK_REPLICATE_MARKOV_W1` | `1` / `1` | TP-sharded DSpark markov head for the vocab-split sampler; `0`/`0` = stock head, split sampler off (= `.6`) |
| `BRRR_TP3_LOG` | `0` | `1` = log once per worker which BRRR kernels engaged |
| `B12X_EXPERT_ORDER_DIR` | `/opt/ds41/expert-order-blend` | expert promotion order; `/opt/ds41/expert-order` = baked order |
| `B12X_EXPERT_ROUTES_ALL` | `0` | `1` = dump route counts every 30 s to build your own order (~0.5-1% while on) |
| `DS41_OFFLOAD_FROM` / `DS41_OFFLOAD_TO` | `20` / `33` | parked expert layers; do not pass ~34 |
| `RESERVE_BYTES` | `1610612736` | GPU memory left free after promotion (1.5 GiB, stress-clean) |
| `ENGRAM_ROW_CACHE_ROWS` | `8388608` | host row cache for disk Engram tables; `0` = off |
| `DRAFT_NVFP4_HEAD` | `1` | DSpark drafter vocabulary head in NVFP4; `0` = BF16 |
| `DS41_LAUNCH_SETTINGS` | `/etc/ds41/launch.yaml` | lil launcher settings file (`k7.yaml` points it at `ds41-launch-k7.yaml`) |
| `DS41_IMAGE` | `azallaza/ds41-kraken-tp3:20261019.7` | image tag (`.6`, `.5`, `.4`, `.3` remain valid) |
| `HOST_PORT`, `GPU0`/`GPU1`/`GPU2` | `15015`, `0`/`1`/`2` | API port, GPU indices |

## compose.yaml

```yaml
# DeepSeek-V4.1-Flash-1017 TP3 release, image azallaza/ds41-kraken-tp3:20261019.7
# (karmic-kraken-beta-20261016.2 + FlashInfer main 5b065abe b12x + the TP3 port in src/,
#  + the 2026-10-31 MXFP4-Engram / NVFP4-expert patches in patch-nvfp4/, inert for stock weights).
# Defaults: original deepseek-ai weights (MXFP4 experts, lossless), DSpark 5 drafts with
# cap-only adaptive verification (all 5 drafts verified up to 4 concurrent decodes), 8 concurrent sequences, 1M context, 4096-token prefill chunks, KV 2,185,581 tokens
# (2.08x 1M; see .env.example for the 4.4x / 6x presets), expert layers 20-33 parked in
# host RAM with their most-routed experts promoted to GPU after startup using the
# expert-order-blend/ order built from real traffic (2026-10-30), Engram layer 1 in host
# RAM and layer 14 read from the checkpoint (with the 4-bit Engram variants set
# ENGRAM_RAM_LAYERS=1,14 to hold both in RAM: faster, ~150 GB host). Needs 3x 96 GB GPUs
# and ~256 GB host RAM. Tested at 450 W/GPU (350 W works; power only affects prefill).
services:
  ds41-kraken-tp3:
    image: "${DS41_IMAGE:-azallaza/ds41-kraken-tp3:20261019.7}"
    container_name: ds41-kraken-tp3
    # unless-stopped: auto-start on host reboot / daemon restart, but respect a manual `stop`.
    restart: "unless-stopped"
    stop_grace_period: 90s
    ipc: host
    # io_uring (disk Engram reads) needs seccomp unconfined; pinned host memory needs memlock.
    security_opt: [seccomp=unconfined]
    ulimits:
      memlock: -1
      nofile: 1048576
      stack: 67108864
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["${GPU0:-0}", "${GPU1:-1}", "${GPU2:-2}"]
              capabilities: [gpu]
    ports:
      - "${HOST_PORT:-15015}:15015"
    volumes:
      # Populated Hugging Face cache with deepseek-ai/DeepSeek-V4.1-Flash-1017 @ 1bae0e10.
      # Engram layer 14 is read from these files during serving: keep them on an NVMe drive.
      - ${HF_CACHE_DIR:?Set HF_CACHE_DIR to your populated Hugging Face cache}:/root/.cache/huggingface:ro
      # b12x compile/tuning caches and the promotion arm file; a fresh volume makes the
      # first start ~10 minutes longer while kernels compile.
      - runtime-cache:/cache
      # Launcher settings from this release (max-num-seqs 8, graphs 48, 4096-token chunks, dspark k5) over the
      # copy baked into the image (which still says max-num-seqs 4).
      - ./ds41-launch.yaml:/etc/ds41/launch.yaml:ro
      # Expert promotion order built from real traffic (see B12X_EXPERT_ORDER_DIR below).
      - ./expert-order-blend:/opt/ds41/expert-order-blend:ro
    environment:
      # Weights repo (from HF_CACHE_DIR). Any checkpoint in the original FP8 layout is a drop-in;
      # e.g. MODEL=mrfakename/DeepSeek-V4.1-Flash-1017-abliterated. See README "Alternative weights".
      MODEL: "${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash-1017}"
      # Fixed KV pool per GPU. 2.25 GiB = 2,185,581 tokens. More KV comes out of expert
      # promotion (fewer hot experts): 4.75 GiB = 4.61M tokens (~-5% decode), 6.5 GiB =
      # 6.31M tokens (~-10%), 8.75 GiB = 8.50M tokens (~-14%). Presets + measured numbers in .env.example (`./kv.sh`).
      KV_CACHE_MEMORY_BYTES: "${KV_CACHE_MEMORY_BYTES:-2415919104}"
      # Expert layers (ordinals FROM..TO) kept in host RAM at startup; after the server is
      # healthy, their most-routed experts move to GPU memory until RESERVE bytes remain free.
      # 20-33 (14 layers) measured best on llm-bench: 72.3 decode steps @32K vs 68.0 at
      # the old 20-25 (+6.3%). Widening spreads the fixed ~6 GiB of parking thinner per
      # layer (cold-fetch rate ~6.7%->0.5%); the win needs the 1.5 GiB reserve below (at
      # 1 GiB, 20-33 throws OOM-mapping retries under concurrent long-context). Do NOT go
      # past ~34 (20-34/36 OOM at load: cold chunks for too many layers exhaust host RAM),
      # and keep the draft layers 37-39 resident. cf=0.2 to wrap more layers under-promotes
      # and loses. See release README-tuning.md.
      DS41_OFFLOAD_FROM_ORDINAL: "${DS41_OFFLOAD_FROM:-20}"
      DS41_OFFLOAD_TO_ORDINAL: "${DS41_OFFLOAD_TO:-33}"
      # GPU memory left free after promotion: prefill scratch for 4096-token chunks AND
      # the serving cushion. 1.5 GiB pairs with 20-33 (stress-clean, min free ~480 MiB
      # under 4x600K concurrent); 1 GiB paired with the old 20-25. 768 MiB OOMs a 31K prompt.
      B12X_DEFERRED_BANK_RESERVE_BYTES: "${RESERVE_BYTES:-1610612736}"
      # Promotion order: per-layer expert route counts (expert-routes-rank0-layer<L>.json,
      # "counts" = 384 ints); the loader relabels each parked layer so its most-routed experts
      # fill the promoted prefix. expert-order-blend/ = 0.6 real coding traffic (544K routed
      # tokens) + 0.4 prose (109K), measured +10% decode steps vs the image's baked order
      # (which came from an 18K-token sample and left ~15% of routes cold). Point at
      # /opt/ds41/expert-order to use the baked order. README: "Update 2026-10-30".
      B12X_EXPERT_ORDER_DIR: "${B12X_EXPERT_ORDER_DIR:-/opt/ds41/expert-order-blend}"
      # 1 = count expert routes on every layer and dump /cache/all-expert-routes-rank<r>-layer<L>.json
      # every 30 s (cumulative) to build your own order. ~0.5-1% decode while on. See README.
      B12X_EXPERT_ROUTES_ALL: "${B12X_EXPERT_ROUTES_ALL:-0}"
      # Cap-only adaptive verification (image .5, ds41-launch.yaml adaptive-verification: true): every DSpark
      # draft is verified while the decoding requests' total target rows (1 + drafts each) fit under this cap;
      # above it drafts are trimmed, best-confidence first across requests. 24 = 4 requests untouched, c6 +24%,
      # c8 +23% aggregate, c1/c4 unchanged (2026-10-31). 0 = upstream's cost-model adaptive (c1 -11%);
      # to verify every draft at any concurrency set adaptive-verification: false in ds41-launch.yaml.
      DS41_VERIFY_ROW_CAP: "${DS41_VERIFY_ROW_CAP:-24}"
      # 1 = log once per worker which BRRR TP3 kernels engaged (skinny GEMM, radix top-k, phased all-reduce). Image .6.
      BRRR_TP3_LOG: "${BRRR_TP3_LOG:-0}"
      # Image .7 bakes VLLM_DSPARK_SHARD_MARKOV_HEAD=1 / VLLM_DSPARK_REPLICATE_MARKOV_W1=1 (TP-sharded DSpark markov head,
      # needed by the vocab-split sampler). Set both to 0 here to get the stock replicated head; the split sampler then
      # disables itself (= image .6 behaviour, ~3-5% slower at c1).
      VLLM_DSPARK_SHARD_MARKOV_HEAD: "${VLLM_DSPARK_SHARD_MARKOV_HEAD:-1}"
      VLLM_DSPARK_REPLICATE_MARKOV_W1: "${VLLM_DSPARK_REPLICATE_MARKOV_W1:-1}"
      # Engram layers whose tables live in pinned host RAM (31.5 GiB per GPU worker each).
      # Both tables (1,14) do not fit next to the parked expert layers on 256 GB.
      DS41_ENGRAM_RAM_LAYERS: "${ENGRAM_RAM_LAYERS:-1}"
      # Host RAM row cache for the disk Engram table, in rows (256 B each): 8388608 = 2 GiB
      # per GPU worker. 0 = read every row from the checkpoint.
      DS41_ENGRAM_ROW_CACHE_ROWS: "${ENGRAM_ROW_CACHE_ROWS:-8388608}"
      # DSpark drafter vocabulary head in NVFP4 (drafts only; every token is still verified by
      # the full model, so outputs are unchanged): +3-4% decode. 0 = BF16 head.
      VLLM_DS41_DRAFT_NVFP4_HEAD: "${DRAFT_NVFP4_HEAD:-1}"
      # Mount a different lil launcher settings file here to change context, drafts or chunking.
      DS41_LAUNCH_SETTINGS: "${DS41_LAUNCH_SETTINGS:-/etc/ds41/launch.yaml}"
      # Persist vLLM/Inductor/Triton compile caches to the runtime-cache volume too (default
      # is the container's ephemeral layer, recompiled on every `recreate`). The heavy b12x +
      # FlashInfer JIT caches already live under /cache; this covers the rest.
      VLLM_CACHE_ROOT: "${VLLM_CACHE_ROOT:-/cache/vllm}"
      TORCHINDUCTOR_CACHE_DIR: "${TORCHINDUCTOR_CACHE_DIR:-/cache/inductor}"
      TRITON_CACHE_DIR: "${TRITON_CACHE_DIR:-/cache/triton}"
      HF_HUB_OFFLINE: "1"
volumes:
  runtime-cache:
```

## Credits

- **Model**: [DeepSeek](https://huggingface.co/deepseek-ai) — [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT).
- **Runtime, kernels and base image**: [Local Inference Lab](https://github.com/local-inference-lab) — the
  `karmic-kraken` vLLM fork and the [`ghcr.io/local-inference-lab/vllm`](https://github.com/local-inference-lab/blackwell-llm-docker)
  CUDA 13.4 runtime image this image is built from, the FlashInfer `b12x` attention/MoE/linear backends and DSpark
  speculative decoding, and [`llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench) (the
  benchmark, hotel-lights and lavd profiles used throughout).
- **Decode kernels in `patch-brrr/`**: [Fraser Price](https://huggingface.co/fraserprice) —
  [DeepSeek-4.1-Flash-1017-BRRR](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR) (Apache-2.0; the
  skinny FP8 GEMV, radix top-k, phased PCIe all-reduce, vocab-split sampling and the renderer tool-loop fix, written for
  TP4 and ported here to TP3; `LICENSE-brrr` and `NOTICE-brrr` are kept with the copies).
- **4-bit Engram tables and the NVFP4 checkpoint**: [NVIDIA](https://huggingface.co/nvidia) —
  [DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) (ModelOpt).
- **Adaptive verification** (the manager the cap-only patch builds on) and everything else under the hood:
  the [vLLM](https://github.com/vllm-project/vllm) and [FlashInfer](https://github.com/flashinfer-ai/flashinfer) projects.
- **Quality checks**: [LiveBench](https://github.com/LiveBench/LiveBench) (coding, release 2024-11-25).
- Ideas reviewed while building this (not shipped in the image): [kacper-daftcode/vLLM-Moet](https://github.com/kacper-daftcode/vLLM-Moet)
  (sm_120 SASS kernels, FP4 KV) and NVIDIA's [Flare](https://github.com/NVIDIA/flare-local-inference) (MTP verification budgets).
