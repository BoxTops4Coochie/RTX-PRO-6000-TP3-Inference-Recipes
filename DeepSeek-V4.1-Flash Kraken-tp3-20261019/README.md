# DeepSeek-V4.1-Flash-1017 Kraken TP3 — 20261019.3

DeepSeek-V4.1-Flash-1017 served at **tensor parallel 3 with the full 1M-token context** on three 96 GB RTX PRO 6000
Blackwell cards, from DeepSeek's **original weights** (FP8 dense layers, MXFP4 routed experts; nothing requantized).

Weights: [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @
`1bae0e1088993b9b82de08f59de5be5a34e42170` (MIT). Runtime: Local Inference Lab's vLLM "karmic-kraken" image
(`karmic-kraken-beta-20261016.2`) with FlashInfer `main` b12x and the TP3 changes in `src/`.

Image: **`azallaza/ds41-kraken-tp3:20261019.3`** (`sha256:c4c2de529da6d3762951e7f5c5b696866f20d740a5f16d9cdda0f13786352646`).
Same code as `20261019` (`sha256:54836f9d…`, 13.0 GB download, 45.1 GB unpacked); `20261019.2` added
`max-parallel-prefills: auto`, and `20261019.3` the NVFP4 drafter head and top-p 1.0 as the server default (see
[Update 20261019.3](#update-202610193-drafter-head-and-top-p-10)). Each shares all but three small layers with the
previous tag, so pulling it over an earlier one downloads ~120 KB.

```bash
docker pull azallaza/ds41-kraken-tp3:20261019.3
HF_CACHE_DIR=/path/to/huggingface docker compose up -d     # port 15015, model name DeepSeek-V4.1-Flash
```

The container arms expert promotion by itself once `/health` answers (log line
`ds41-serve: expert promotion armed and warmed`); serve traffic after that line for full speed.

Host settings: 350 W/GPU, +6000 memory offset, ~256 GB host RAM, the checkpoint on an NVMe drive (Engram layer 14 is read from it while serving).

## Update 2026-10-28: wider expert promotion (20-33) + 1.5 GiB reserve

Config-only change on the same image (`20261019.3`) — widen the parked/promoted expert range and raise the
promotion reserve. **llm-bench decode: 68.0 -> 72.3 engine steps/s @ 32K (+6.3%)**, no quality, KV, or prefill
cost. This supersedes the "parking 20-29 instead of 20-25 — no further gain" note under
[Measured and not adopted](#measured-and-not-adopted): that was a tok/s A/B, which can't resolve a ~6% gain through
spec-decode acceptance noise; engine-steps/s does.

| setting (compose `.env` / defaults) | was | now |
| --- | --- | --- |
| `DS41_OFFLOAD_TO` | `25` (layers 20-25, 6 layers) | **`33`** (layers 20-33, 14 layers) |
| `RESERVE_BYTES` | `1073741824` (1 GiB) | **`1610612736`** (1.5 GiB) |
| `restart` | `"no"` | **`"unless-stopped"`** (auto-start on host reboot) |
| vLLM/Inductor/Triton compile caches | ephemeral (recompiled each `recreate`) | **persisted to `/cache`** (`VLLM_CACHE_ROOT=/cache/vllm`, `TORCHINDUCTOR_CACHE_DIR=/cache/inductor`, `TRITON_CACHE_DIR=/cache/triton`) |

**Why it helps.** Spreading the fixed ~6 GiB of host-parked experts across 14 layers instead of 6 makes each
layer's cold tail shallower (now ~20 of 24 chunks promoted per layer vs ~14), cutting the cold-expert fetch rate
from ~6.7% to ~0.5% of routes. Free VRAM after promotion is now ~1.5 GiB (was 1 GiB).

**The 1.5 GiB reserve is required at this range.** At 1 GiB, 20-33 runs ~5 MiB free after warm-up and throws
repeated `expandable_segments: memory mapping failed with OOM` retries under concurrent long-context (the cuBLAS-crash
precursor). At 1.5 GiB it holds ~480 MiB free under a 4x600K-token prefill-during-decode stress with zero retries.
Run-to-run llm-bench noise is +/-1-2 steps (vLLM spec-decode batching is not bit-deterministic even at temperature
0), so +6.3% is the two-run mean.

**Limits.** Do not widen past ~34: offload 20-34/20-36 OOM at *load* because `DS41_VMM_COLD_FRACTION=1` host-maps
every wrapped layer's experts up front before promotion trims them. Keep the dspark draft layers 37-39 resident.
Lowering `DS41_VMM_COLD_FRACTION` below 1 to wrap more layers fits in memory but under-promotes (leaves VRAM unused)
and loses the gain (20-36/20-39 at cf=0.2 measured 69.7/70.3 steps, below 20-33). Host GPUs were at 450 W here.

## Update 20261019.3: drafter head and top-p 1.0

| Change | Effect |
| --- | --- |
| DSpark drafter vocabulary head in NVFP4 (`VLLM_DS41_DRAFT_NVFP4_HEAD=1`; compose `DRAFT_NVFP4_HEAD`, `0` = BF16) | Only the drafts change: the full model still verifies every token, so outputs keep the target model's distribution. A/B on the same image (top-p 0.95, fresh restart each): short 228.7 -> 235.3 tok/s, 32K C1 164 -> 169, 128K C1 160 -> 167, 128K C4 320 -> 329, 4 concurrent short 461 -> 475, coding 265 -> 268; prefill and needle unchanged |
| Server-default sampling `temperature 1.0, top_p 1.0` (`override-generation-config` in `ds41-launch.yaml`) | The model card's recommendation; the lil `ds41-flash` profile ships top_p 0.95 from the older V4.1 launcher. Requests that set their own values still win. Quality rerun at 1.0 below; the 0.95 results are kept in `qualification/topp-0.95-20261019.2/` |

Measured on `20261019.3` from a fresh restart (`bench-20261019.3/`): real code 10,646 / 10,228 / 11,354 tok/s prefill;
`llm_decode_bench` prefill 32K / 128K 13,486 / 12,245 tok/s; 983K needle pass at 7,193 tok/s (peak 97,319 MiB);
decode 32K C1 / C4 186.7 / 340.7 tok/s, 128K C1 / C4 179.2 / 307.1 tok/s; coding peak 271.8 tok/s mean; short
223-229 tok/s; 4 concurrent short 474 tok/s; 5/5 known answers. Draft acceptance at 1 user rose (2.71-2.77 vs
2.41-2.51 tokens/step at 0.95); the 128K C4 cell is the noisiest (307-349 tok/s across runs of the same image).

## Update 20261019.2: parallel prefills

| Change | Effect |
| --- | --- |
| `max-parallel-prefills: 1 -> auto` (vLLM `max_parallel_prefills`): up to 4 prefills share each step round-robin; a lone prefill keeps the full 4096-token budget | a short (89-token) request arriving 3 s into a ~420K-token prefill: TTFT **34.2-35.1 s -> 0.57-0.67 s**; the long prompt 37.2-38.1 -> 38.1-39.2 s |

Rechecked with `auto` (no other change): 2 x ~900K + 2 x ~14K prompts prefilling at once 4/4 OK, 0 preemptions,
peak KV 75.2%, min free 550 MiB; 983K-token needle pass at 7,467 tok/s; `llm_decode_bench` prefill 32K / 128K
13,502 / 12,206 tok/s (cold Engram row cache after the restart; 14,195 / 12,716 on `20261019`); decode 219-236 tok/s,
4 concurrent 490 tok/s, 5/5 known answers. Mount a settings file with `max-parallel-prefills: 1` to restore one
prefill at a time. Results in `qualification/parallel-prefills/`.

| | |
| --- | --- |
| GPUs | 3× RTX PRO 6000 Blackwell Workstation (sm_120, 96 GB, PCIe Gen5 x16), 350 W cap, +6000 memory offset |
| Host | Threadripper PRO 7965WX (24 cores), 256 GB RAM (247 GiB usable), driver 615.71.09 |
| Context / KV | 1,048,576-token window, **2,185,581-token KV pool (2.08 × 1M)**, up to 4 sequences |
| Prefill | 13,486-14,195 / 12,245-12,716 tok/s at 32K / 128K (`llm_decode_bench`); ~10,000-11,000 tok/s on real source code (100-260K tokens); 7,129-7,467 tok/s on a 983K-token prompt |
| Decode, 1 user | ~180-187 tok/s at 32K-128K context, 225-300 tok/s on code and explanations (DSpark, 5 drafts) |
| Decode, 4 users | 307-349 tok/s aggregate at 32K-128K context; ~475 tok/s on short prompts |

## Contents

1. [What it is](#what-it-is)
2. [How it fits in 3 × 96 GB](#how-it-fits-in-3--96-gb)
3. [Update 20261019.3](#update-202610193-drafter-head-and-top-p-10) · [Update 20261019.2](#update-202610192-parallel-prefills) · [Measurements](#measurements-on-this-image) · [Quality and long context](#quality-and-long-context)
4. [Memory](#memory)
5. [Measured and not adopted](#measured-and-not-adopted)
6. [Alternative weights](#alternative-weights) · [Known limitations](#known-limitations) · [Not tested](#not-tested)
7. [Files](#files) · [Variables](#variables) · [compose.yaml](#composeyaml)

## What it is

| Layer | Source |
| --- | --- |
| Base image | `ghcr.io/local-inference-lab/vllm@sha256:f5f9efe1…` (karmic-kraken-beta-20261016.2, vLLM `6d66eed35`) |
| b12x kernels | FlashInfer `main` @ `5b065abe` (124 commits past the base image's `79819b11`): compact-scale MoE decode/prefill, inline CSF scale decoding, MoE race fixes. Fetched by the Dockerfile; Python only, compiled at startup |
| TP3 changes | 16 vLLM files in `src/` (`source.patch` against the base image) |
| Weights | the original `deepseek-ai/DeepSeek-V4.1-Flash-1017` from a Hugging Face cache |

The base image already pads DS4.1's 64 attention heads to 72 (8 → 9 output groups) at TP3. What does not fit is
memory: the original checkpoint needs ~95 GiB per GPU before KV, caches and activations.

## How it fits in 3 × 96 GB

**Expert layers parked in host RAM, hottest experts promoted back to GPU.** Routed-expert layers 20-25 (of 40) start
in host-backed CUDA virtual memory (HOST_NUMA pages behind a device address range), so startup fits. Once the server
is healthy, `ds41-serve` creates `/cache/deferred-banks.arm` and sends a warm-up request; on that forward pass each
parked layer remaps 16-expert chunks of its weights to GPU memory **at the same virtual address** (captured CUDA graphs
stay valid) until 1 GiB per GPU remains free. The host copies of promoted chunks are released.

**Hot-first expert order.** At load, each parked layer's experts are relabeled most-routed-first (router rows and
expert weights together, so outputs are unchanged), using route counts recorded on 1017 (`expert-order/`). The
promoted prefix therefore holds the experts that carry most routes: routing is skewed (the top 256 of 384 experts
carry 98-99.9% of routes), and ~225 hot experts per parked layer end up on GPU.

**Compressed expert scales.** All routed-expert scales are compressed losslessly at load with FlashInfer's own
per-row CSF encoder (`scale_compression="csf"`) and expanded per call into one shared scratch (~141 MB per GPU).
This also covers the DSpark drafter's experts, whose scales the earlier fixed-base codec could not compress
(that cost ~2.1 ms per drafter MoE call, about a quarter of every decode step).

**Engram.** Two n-gram tables of 94 GiB each (layers 1 and 14). Layer 1's table is pinned in host RAM. Layer 14's
is read from the checkpoint with io_uring, with its E8M0 scales resident in host RAM and a 2 GiB-per-GPU host row
cache in front of the reader (`engram_row_cache.py`; exact, measured 62% row reuse). Both tables in RAM do not fit
next to the parked expert layers on 256 GB (see [Measured and not adopted](#measured-and-not-adopted)).

**Decode.** DSpark speculative decoding (the checkpoint's 3 `mtp.*` layers), 5 drafts, every draft verified.

## Measurements on this image

On `20261019` (one prefill at a time) from a fresh cache volume, 350 W, +6000 memory offset, one container; raw outputs in `bench/`.

| Benchmark | Result |
| --- | --- |
| Known-answer checks (5 prompts) | 5/5 |
| Needle at 28,553 tokens | pass, TTFT 2.32 s (**12,318 tok/s** prefill) |
| Needle at 982,982 tokens | pass, TTFT 137.9 s (**7,129 tok/s**), peak 97,337 of 97,887 MiB |
| Real source code (FlashInfer Python), 105K / 259K / 102K tokens | **10,594 / 9,928 / 11,086 tok/s** prefill |
| Decode, short prompt, 3 runs | 78.9-79.8 steps/s × 2.84-2.87 tokens/step = **226-227 tok/s** |
| 4 concurrent short requests | 458 tok/s aggregate |
| `llm_decode_bench` prefill 32K / 128K (32,252 / 128,511 tokens) | **14,195 / 12,716 tok/s** (TTFT 2.27 / 10.11 s) |
| `llm_decode_bench` 32K, C1 / C4 | 171.7 / 336.3 tok/s (67.8 / 138.3 engine steps/s) |
| `llm_decode_bench` 128K, C1 / C4 | 172.6 / 349.3 tok/s (67.1 / 138.5 engine steps/s) |
| `llm_decode_bench` coding peak (Sieve of Eratosthenes, 3 runs) | 254.4 tok/s mean |

Prefill on repetitive synthetic filler is faster than on real text (11-14K tok/s at 10-32K tokens) because the
Engram row cache hits more; the real-code rows are the representative numbers.

## Quality and long context

Same protocol as the GLM-5.3 Kraken TP3 releases (`qualification/run_qualification.py`): thinking on,
`reasoning_effort` max, temperature 1, on the published image without restarts. Top-p 1.0 on `20261019.3` (the
server default); the same checks at top-p 0.95 on `20261019.2` are in the right column.

| Test | top-p 1.0 (`20261019.3`) | top-p 0.95 (`20261019.2`) |
| --- | --- | --- |
| lavd (`llm_decode_bench --test-profile lavd`), 30 runs at 4 concurrent, seed 103 | **30 exact** / 0 wrong / 0 truncated | 30 exact / 0 / 0 |
| hotel-lights, 30 runs at 4 concurrent, seed 103 | **30 exact** / 0 wrong / 0 truncated | 30 exact / 0 / 0 |
| Retrieval of three exact markers at 128K / 900K tokens | **pass / pass** | pass / pass |
| Long generation after a captured 581,537-token conversation, seeds 201-204, natural EOS, 65,536-token cap | **4/4 clean**: 21,292-31,792 tokens, all ended with `stop`, worst 8-gram repeat 0.0005 | 4/4 clean: 16,323-23,274 tokens, worst 0.0013 |

Vision and stress tests ran on `20261019`/`.2` (vision without a top-p setting, the stress tests at 0.95).

| Vision and stress test | Result |
| --- | --- |
| Vision, basic set (60 images: OCR, 5-line read, count 3-12, 3x3 colors, tallest bar), T=0 / T=1 | **60/60, 60/60** |
| Vision, hard set (small low-contrast OCR, 10-line read, count 13-28, exact bar value, 12 similar shades), T=0 / T=1 | **58/60, 59/60** (117/120): one OCR spacing error, one 10-line read with inserted words; dense counting 24/24 |
| 4 concurrent requests x 10 photos at 6000x4000 (11.8 MB JPEGs) | **4/4 OK**, 0 preemptions, 9,831 prompt tokens each, min free 590 MiB |
| 4 concurrent: 2 x ~900K-token + 2 x ~14K-token prompts, 48K / 40K generated tokens each, `ignore_eos` | **4/4 OK** (all four running together), 0 preemptions, 1,827,825 prompt tokens, peak KV 74.0%, min free 590 MiB; with `auto` (all four prefilling at once, cold cache): 4/4 OK, 0 preemptions, peak KV 75.2%, min free 550 MiB, 818 s |

Vision scores use `vision/score_vision.py`, which also accepts a final "the answer is X" sentence: at T=1 the model
often ends with "Therefore, the answer is **green**." where the GLM scorer expects the bare word (strict scores:
basic 48/60, 48/60; hard 36/60, 50/60). The vision tower runs from host memory (`--cpu-offload-params vision`).

### Coding task

One prompt (`qualification/coding-task/PROMPT.md`): build a standard-library Python CLI, `bench_report.py`, that
validates a JSONL file of benchmark runs, groups them by model and concurrency, reports throughput statistics with
`--top`, `--min-acceptance`, `--model` and table/JSON/CSV output, plus a test file that runs both directly and
under pytest. The model wrote both files in one pass (462 + 323 lines, unedited in `coding-task/`).

| Check | Result |
| --- | --- |
| `python3 test_bench_report.py` / `pytest` | **18 passed** / **18 passed** |
| Sample file from the prompt (4 valid rows, 1 invalid) | table correct by hand calculation; the invalid row reported to stderr with its line number |
| `--top`, `--min-acceptance`, `--model`, `--format json` (numbers, not strings), `--format csv` (header row) | all as specified |
| Unreadable file / zero rows after filtering | exit code 1 with an error message, both |
| Standard library only, dataclasses, type hints, logic outside `main()` | yes: 2 frozen dataclasses, 17 functions |
| Generation speed (server log, 10 s windows, 1 user) | **~14,600 tokens in ~70 s, 191-281 tok/s** (rising as acceptance length climbed from 3.2 to 5.3 of 6); 60% of drafted tokens accepted |

Two choices left open by the prompt: the standard deviation is the population form, and token totals print as
floats (`3300.0`). Outputs and the commands run are in `coding-task/verification.txt`; the server log lines in
`coding-task/server-log.txt`.

For reference, GLM-5.3 Kraken TP3 20261015 measured lavd 30 exact, hotel-lights 27-28 exact, long generation
4/4 clean (worst 0.0017) and vision 117/120 on the hard set on the same tests.

## Memory

| Per GPU | |
| --- | --- |
| Weights after load | 79.66 GiB (expert layers 20-25 in host memory at this point) |
| Promoted to GPU after arming | ~1.3 GB per parked layer group pass (14 of 24 chunks per region) |
| KV | 2.25 GiB fixed = 2,185,581 tokens at 4096-token chunks |
| Free after promotion | 1 GiB (prefill scratch for 4096-token chunks) |
| Peak during a 983K-token prompt | 97,337 of 97,887 MiB |

**Host RAM:** ~150 GiB in use while serving, ~95 GiB available: layer 1's Engram table (94.4 GiB pinned), the
unpromoted parts of the parked expert layers (~16 GiB), layer 14's resident scales and row cache (~9 GiB), plus
the vLLM processes. With ~256 GB this leaves room for the OS and other services, not for the second Engram table.

## Measured and not adopted

| Change | Result |
| --- | --- |
| Both Engram tables in host RAM (at startup, or filled lazily after promotion) | Out of memory: 94.4 GiB needed with ~70-77 GiB free after promotion |
| Prefill chunks of 8192 tokens (1 GiB reserve) | Out of memory on a 31K prompt; KV would also drop to 1.84M tokens |
| Promotion reserve 768 MiB at 4096-token chunks | Out of memory on a 31K prompt |
| Adaptive verification on (the profile default) | 8-12% slower decode at short, 128K and 512K context; equal at 32K |
| PCIe one-shot all-reduce up to 6 MB, or DMA from 1 MB | No change in TTFT, prefill or decode |
| Prefill chunks of 4080 tokens so the attention all-reduce takes the b12x DMA path (4096 x 5120 elements is not a multiple of 24) | No change (real-code prefill 10,690 / 10,467 / 11,499 vs 11,189 / 10,560 / 11,692 tok/s); NCCL already moves a 40 MiB all-reduce at 45.8 GB/s bus bandwidth |
| 4 / 6 / 7 drafts instead of 5 | 4: 1-user decode at 32K-128K -23-28% (4 users +6-14%); 6: 1 user +1-4%, short prompts -5%, 4 users collapse to ~50 tok/s; 7: slower everywhere |
| Prefill indexer chunk 256 instead of 128 | 983K prefill 7,165 vs 7,277 tok/s; 4-user decode -3-7% (fewer experts promoted) |
| Tuning pinned to MoE only | Startup fails: ranks reach different tuning keys |
| Parking layers 20-29 instead of 20-25 | No further gain |
| `local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF` | Same weights; its container format is read by the base image's loader but not by FlashInfer `main` |

## Alternative weights

The recipe serves the original `deepseek-ai/DeepSeek-V4.1-Flash-1017` by default. Any checkpoint that keeps the
original FP8 layout (same tensor names, dtypes and shapes, same `model.safetensors.index.json`) is a drop-in: point
`MODEL` at it and it loads with no other change.

**Get it and run it** (example: the abliterated build below). Download the weights into the same Hugging Face cache
`HF_CACHE_DIR` points at, then set `MODEL` and start:

```bash
# 1. fetch the weights into the cache (skip the tiny GGUF LoRA; downloads all 48 FP8 shards,
#    ~510 GB. hf re-runs resume, so just rerun on a stall.)
HF_HUB_ENABLE_HF_TRANSFER=1 hf download mrfakename/DeepSeek-V4.1-Flash-1017-abliterated \
  --exclude '*.gguf' --cache-dir "$HF_CACHE_DIR/hub"

# 2. serve it — set MODEL inline, in compose.yaml, or in a .env next to it
MODEL=mrfakename/DeepSeek-V4.1-Flash-1017-abliterated docker compose up -d --force-recreate

# 3. wait for the arm line, same as the stock weights, then send traffic
docker logs -f ds41-kraken-tp3 2>&1 | grep -m1 'expert promotion armed and warmed'
```

`hf download` fetches the whole repo (~510 GB); it only skips files already downloaded *for this repo*, not the 8
shards that happen to be byte-identical to the stock model (those sit under a different cache folder). If you already
have the stock weights cached and want to save that ~214 GB, hardlink the shared shards into the abliterated repo's
`blobs/` by their LFS oid before step 1, then only the 40 edited shards (~296 GB) download.

Switch back by setting `MODEL=deepseek-ai/DeepSeek-V4.1-Flash-1017` (or removing the override) and recreating; both
copies stay cached, so flipping between them later is just the `MODEL` change. A persistent choice is cleanest in a
`.env` file beside `compose.yaml` (`MODEL=…`), which `docker compose` reads automatically.

The abliteration community re-publishes 1017 with refusal behaviour removed by a weight-space edit (directional
ablation of `attn.wo_b` and the shared-expert `w2`); the routed experts, Engram tables, MTP/DSpark layers and block
scales stay bit-identical to the original. We ran one such build (`mrfakename/DeepSeek-V4.1-Flash-1017-abliterated`,
48 FP8 shards, 8 of them byte-identical to the original) through this recipe on `20261019.3`:

| | Stock `1017` | Abliterated |
| --- | --- | --- |
| lavd / hotel-lights, 30 runs @ 4 concurrent | 30/30, 30/30 | **30/30, 30/30** |
| Retrieval 128K / 900K | pass / pass | **pass / pass** |
| Long generation, 4 seeds (worst 8-gram repeat) | 4/4 clean (0.0005) | **4/4 clean** (0.0019) |
| Real-code prefill / 983K-token needle | 10,646-11,354 / 7,193 tok/s | 10,133-11,203 / 7,054 tok/s |
| Decode engine steps/s (short, 32K, 4×32K) | 79.1-79.4 / 67.5 / 139.9 | 79.5-79.6 / 67.2 / 140.3 |
| Peak GPU memory | 97,319 MiB | 97,317 MiB |

Speed, context, memory and the quality checks are unchanged (tok/s moves only with run-to-run draft acceptance at
temperature 1). These are **third-party weights** — we did not produce them or audit them for safety, and the
uncensoring numbers are the publisher's, not ours. A checkpoint in a different container (CSF, NVFP4, EXL3, GGUF,
MLX) or with a changed tensor inventory (e.g. expert pruning) will **not** load here without conversion.

## Known limitations

- Promotion happens on the warm-up request after `/health`; requests that arrive before the
  `expert promotion armed and warmed` log line run with parked layers 20-25 in host memory (slower decode).
- Up to 4 concurrent sequences (`max-num-seqs: 4`, CUDA graphs up to 24 tokens).
- Time to first token is ~0.25 s even for short prompts (fixed per-request host overhead of the eager prefill pass).

## Not tested

Benchmark accuracy scores (GSM8K/MMLU), more than 4 concurrent requests (the configured limit), video inputs,
other GPU power limits.

## Files

- `Dockerfile`: base image + FlashInfer `main` @ `5b065abe` + `src/` + defaults; `ds41-serve` (entrypoint),
  `ds41-launch.yaml` (lil launcher settings, baked as `/etc/ds41/launch.yaml`), `expert-order/` (1017 route counts
  for layers 20-39), `compose.yaml`, `source.patch` (the 16 vLLM files against the base image), `build.log`,
  `image-inspect.json`.
- `qualification/`: `run_qualification.py` and the GLM-derived checks it runs (`long_retrieval.py`, `long_context.py`,
  `analyze_quality.py`), `vision/` (the GLM synthetic image sets, `run_vision.py`, `score_vision.py`), `img_stress.py`,
  `stress_kv.py`, `prefill_fairness.py` (short request behind a long prefill), `parallel-prefills/` (the `auto` rechecks),
  `coding-task/` (prompt, the model's two files, verification, server log),
  with `RESULTS.md` and the raw outputs. The 581K-token conversation used for long generation is not included.
- `bench-20261019.3/`: the `20261019.3` measurements (same scripts).
- `bench/`: the measurements above (`quick`, `prefill_real_code`, `llm_decode_bench`, `llm_decode_bench_prefill`,
  `needle_983k`) and the scripts that produced them: `bench.py` (quick), `prefill_real.py`, `bench_long.py` (needle).

## Variables

| Variable | Default | Values |
| --- | --- | --- |
| `HF_CACHE_DIR` | required | Hugging Face cache containing the weights repo below |
| `MODEL` | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | weights repo to serve; see [Alternative weights](#alternative-weights) |
| `HOST_PORT` | `15015` | host port for the OpenAI-compatible API |
| `GPU0`/`GPU1`/`GPU2` | `0`/`1`/`2` | GPU indices |
| `KV_CACHE_MEMORY_BYTES` | `2415919104` | KV bytes per GPU; more KV leaves less for promoted experts |
| `DS41_OFFLOAD_FROM` / `DS41_OFFLOAD_TO` | `20` / `25` | parked expert layers (ordinals) |
| `RESERVE_BYTES` | `1073741824` | GPU memory left free after promotion; 4096-token chunks need 1 GiB |
| `ENGRAM_RAM_LAYERS` | `1` | Engram layers held in pinned host RAM (empty = both read from disk) |
| `ENGRAM_ROW_CACHE_ROWS` | `8388608` | host row cache for disk Engram tables (rows of 256 B); `0` = off |
| `DRAFT_NVFP4_HEAD` | `1` | DSpark drafter vocabulary head in NVFP4; `0` = BF16 |
| `DS41_LAUNCH_SETTINGS` | `/etc/ds41/launch.yaml` | lil launcher settings (context, chunk size, drafts, Engram mode) |

## compose.yaml

```yaml
# DeepSeek-V4.1-Flash-1017 TP3 release, image azallaza/ds41-kraken-tp3:20261019.3
# (karmic-kraken-beta-20261016.2 + FlashInfer main 5b065abe b12x + the TP3 port in src/).
# Defaults: original deepseek-ai weights (MXFP4 experts, lossless), DSpark 5 drafts with
# every draft verified, 1M context, 4096-token prefill chunks, KV 2,185,581 tokens
# (2.08x 1M), expert layers 20-25 parked in host RAM with their hottest experts promoted
# to GPU after startup, Engram layer 1 in host RAM and layer 14 read from the checkpoint.
# Needs 3x 96 GB GPUs and ~256 GB host RAM (~150 GB used). Tested at 350 W/GPU.
services:
  ds41-kraken-tp3:
    image: "${DS41_IMAGE:-azallaza/ds41-kraken-tp3:20261019.3}"
    container_name: ds41-kraken-tp3
    restart: "no"
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
    environment:
      # Weights repo (from HF_CACHE_DIR). Any checkpoint in the original FP8 layout is a drop-in;
      # e.g. MODEL=mrfakename/DeepSeek-V4.1-Flash-1017-abliterated. See README "Alternative weights".
      MODEL: "${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash-1017}"
      # Fixed KV pool per GPU. 2.25 GiB = 2,185,581 tokens at 4096-token chunks.
      KV_CACHE_MEMORY_BYTES: "${KV_CACHE_MEMORY_BYTES:-2415919104}"
      # Expert layers (ordinals FROM..TO) kept in host RAM at startup; after the server is
      # healthy, their most-routed experts move to GPU memory until RESERVE bytes remain free.
      # 20-25 measured best (20-22: -5% decode, -10% at 4 users; 20-29: no further gain).
      DS41_OFFLOAD_FROM_ORDINAL: "${DS41_OFFLOAD_FROM:-20}"
      DS41_OFFLOAD_TO_ORDINAL: "${DS41_OFFLOAD_TO:-25}"
      # GPU memory left free after promotion: prefill scratch for 4096-token chunks.
      # 768 MiB runs out of memory on a 31K-token prompt.
      B12X_DEFERRED_BANK_RESERVE_BYTES: "${RESERVE_BYTES:-1073741824}"
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
      HF_HUB_OFFLINE: "1"
volumes:
  runtime-cache:
```

## Credits

The TP3 expert-parking design (host-backed VMM regions for the decoder-half expert layers, deferred promotion
behind an arm file) comes from [infernix](https://github.com/infernix)'s `ds41f-tp3` work on karmic-kraken; this
image replaces its slab banks and scale codec with in-place chunked promotion and FlashInfer's CSF compression.
Runtime and kernels: [Local Inference Lab](https://github.com/local-inference-lab) (vLLM fork, FlashInfer/b12x).
