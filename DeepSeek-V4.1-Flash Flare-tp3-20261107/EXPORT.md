# Exporting the package yourself

This guide builds the Flare package in this folder from scratch. It covers the toolchain, the patched Flare fork,
the checkpoint, the export, and the checks to run before serving. Budget **about 2 hours of mostly unattended time**
and **~900 GB of free disk**.

## 1. Requirements

| | Tested with | Notes |
| --- | --- | --- |
| GPUs | 3 × RTX PRO 6000 Blackwell, 96 GB (SM120), PCIe Gen5 x16 | the export tunes and captures on these exact cards. Tested at **450 W with +75 MHz core / +6000 MHz memory offset**; 350 W measured 10-14% slower. Kernel tuning measures your cards as they run, so export with the clocks and power limit you will serve with |
| Host RAM | 256 GB (tested: DDR5-6000, 8 channels) | export peaks around 160 GB. Serving uses ~148 GB with Engram in RAM, ~59 GB with Engram on disk |
| Disk | ~900 GB free | checkpoints ~412 GB, package ~380 GB, kernel tuning cache 60-80 GB. Tested on PCIe Gen5 NVMe (12 GB/s reads) |
| Driver | 615.71.09 | any driver supporting CUDA 13.1 |
| CUDA toolkit | 13.1 (`/usr/local/cuda-13.1`) | nvcc 12.x is too old for SM120. The export also compiles the BRRR layer kernel with it |
| Rust | 1.97 (rustup) | builds the `flare` binary and the Heliopause extension |
| uv | 0.11 | Flare's Python environment (Python 3.12, managed by uv) |
| Locked memory | `memlock unlimited` for your user | Flare pins host memory; for example `/etc/security/limits.d/90-memlock.conf`: `<user> - memlock unlimited`, then log in again |

```bash
export CUDA_HOME=/usr/local/cuda-13.1 PATH=/usr/local/cuda-13.1/bin:$PATH
ulimit -l        # must print "unlimited"
```

## 2. Build the Flare fork

The fork is upstream NVIDIA Flare at `972e28e4` plus `source.patch` from this folder.

```bash
git clone https://github.com/NVIDIA/flare-local-inference.git flare
cd flare
git checkout 972e28e44781d8d0c84d70ae434db92c36076a43
git apply /path/to/this-folder/source.patch           # 45 files (15 new, including brrr/); applies cleanly to 972e28e4

uv sync --locked --extra prepare --extra test         # Python environment in .venv (pulls FlashInfer @ 1f8ed8a3)
uv run --locked --extra prepare python -m ignition.patches   # Flare's torch/Triton patches; re-run after every uv sync
cargo build --release --manifest-path flare/Cargo.toml  # the flare binary: flare/target/release/flare
```

**Apply the FlashInfer patches** to the installed FlashInfer. They patch files inside `.venv`, so re-apply them
after every `uv sync`:

```bash
SP=$(.venv/bin/python -c "import flashinfer, pathlib; print(pathlib.Path(flashinfer.__file__).parents[1])")
for p in /path/to/this-folder/tp3-patches/*.patch; do patch -p1 -d "$SP" < "$p"; done
```

| Patch | Why |
| --- | --- |
| `flashinfer-tiny-decode-swiglu.patch` | the `tiny_decode` W4A8 MoE kernel ignored `swiglu_limit`; without it, outputs corrupt mid-reply at TP3 |
| `flashinfer-dcp-a2a-world3.patch`, `flashinfer-tuning-world3.patch` | PCIe all-to-all and its tuning tables for world size 3 |
| `flashinfer-engram-mxfp4.patch` | packed-FP4 path in the Engram lookup kernel, for NVIDIA's MXFP4 tables |

Check the build:

```bash
flare/target/release/flare --version
.venv/bin/ignition export --help | head -3
```

## 3. Build the checkpoint

The package is exported from a **checkpoint view**: DeepSeek's original shards 1-46 plus NVIDIA's shards 47-48,
which hold the two Engram tables in MXFP4 (97 GiB instead of 189 GiB in FP8). Download only the shards you need:

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --exclude "model-00047-of-00048.safetensors" "model-00048-of-00048.safetensors"          # ~307 GB
hf download nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 \
  --include "model-00047-of-00048.safetensors" "model-00048-of-00048.safetensors" LICENSE README.md   # ~105 GB
```

Then assemble the view (symlinks only, no copies) with this folder's `config.json` and
`model.safetensors.index.json`. The script checks that every tensor resolves and that the Engram tables are MXFP4:

```bash
/path/to/this-folder/make_checkpoint_view.sh \
  ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
  ~/.cache/huggingface/hub/models--nvidia--DeepSeek-V4.1-Flash-1017-NVFP4/snapshots/<revision> \
  ~/ds41-flare-ckpt
# expect: "96085 tensors in 48 shards, missing 0; layer-1 Engram table U8 [384006168, 128]"
```

The index in this folder is NVIDIA's mapping with a corrected `total_size`. Flare checks that field against the
actual tensor bytes; the uncorrected value makes the export stop at "index total_size does not match".

## 4. Export

From the fork checkout, with GPUs idle (stop any server first). **Export twice on a fresh machine:** with a cold
kernel tuning cache, tuning at `HOT=342` runs out of VRAM. A `HOT=300` export first fills the shared cache, and the
second export reuses it:

```bash
CKPT=~/ds41-flare-ckpt HOT=300 /path/to/this-folder/export.sh   # warms the tuning cache (and gives the 6.25M-KV variant)
CKPT=~/ds41-flare-ckpt /path/to/this-folder/export.sh           # the release build (HOT=342)
```

`export.sh` sets these and runs `ignition export`:

| Setting | Value | Meaning |
| --- | --- | --- |
| `FLARE_DS41_HOST_EXPERT_LAYERS` | `20-33` | routed-expert layers that keep part of their experts in host RAM |
| `FLARE_DS41_HOT_EXPERTS` (`HOT`) | `342` | experts per parked layer kept in VRAM (out of 384); the rest live in host RAM |
| `FLARE_DS41_MERGED_BANK` | `1` | one VRAM + host-RAM expert bank per parked layer (one MoE call) |
| `FLARE_DS41_EXPERT_COUNTS_DIR` | `expert-order/` | route counts that decide which experts are hot |
| `FLARE_DS41_BRRR_LAYERS` | `0-39` | layers that use the fused BRRR layer tail (compiled from `brrr/kernels/decoder.cu` at export) |
| `FLARE_DS41_BRRR_CKPT` | `$CKPT` | checkpoint the BRRR tail reads its weight layout from |
| `FLARE_DS41_BRRR_PARKED_MAX_ROWS` | `12` | parked layers use the BRRR tail only in graphs of ≤12 rows |
| `FLARE_DS41_EMBED_HOST`, `FLARE_DS41_VISION_HOST` | `1`, `1` | token embedding and vision tower in device-visible host RAM |
| `FLARE_DS41_AR_ROWS` | `1024` | size the prefill all-reduce runtime for 1024 rows (~600 MiB/GPU less than 4096) |
| `FLARE_DS41_TP3_VOCAB_PARALLEL` | `1` | vocabulary-parallel sampling at TP3 (fixes top_p at 1.0) |
| `FLARE_DS41_TP3_GRAPH_GIB` | `1` | the planner's graph-memory estimate for TP3 |
| `--prefill-chunk-size` (`PREFILL`) | `4096` | prefill chunk; 2048 gives more KV, less prefill speed |
| `--max-concurrency` | `8` | request slots |
| `--mtp-draft-tokens` (`DRAFTS`) | `5` | DSpark drafts, NVFP4 draft head, FP8 Markov head. 7 measured slower (MEASUREMENTS.md) |

What you'll see, roughly: weights load (~1 min) → BRRR kernel compile (~1 min, first export only) → kernel tuning
(**20-30 min with a cold cache**, seconds when cached) → compile (~3 min) → graph capture → weight payloads written
(~3 min) → publish (~8 min). The log ends with
`Exported cache entry: ~/.cache/flare/packages/models/<checkpoint name>/gb202-tp3-<timestamp>`.

### Variants: more KV for less speed

The hot-expert count is the main lever. Each hot expert per parked layer moved to host RAM frees ~79 MiB per GPU,
about **92K KV tokens**. Decode gets slower because more experts stream over PCIe.

| `HOT` | KV tokens | Measured (350 W, natural EOS, 32K c1 / c4 / c8) |
| --- | --- | --- |
| **342** | **2,373,760** | 288 / 634 / 838 tok/s (**this release**; 298-305 / 716-738 / 957-975 at 450 W, user command) |
| 300 | 6,247,680 | 269 / 610 / 800 tok/s: 4-7% slower, corruption check clean |

> **Disclaimer.** Other values were not measured on this fork. Expect KV ≈ 2.37M + (342 − HOT) × 92K tokens, and a
> decode cost that grows as more of your traffic routes to cold experts (roughly 1% per 6-10 experts in our
> measurements). Check the `Allocate cache` line and bench your own export before relying on it.

## 5. Check and serve

```bash
flare/target/release/flare models list                      # the new version appears as the default
FLARE=flare/target/release/flare PACKAGE=<package path> /path/to/this-folder/serve.sh
```

The server is ready when the log shows `Start server. http://127.0.0.1:15015`, about 55 s with the Engram tables
loading into RAM. The `Allocate cache` line reports the KV token count; it should match the table above within a
few thousand tokens.

**Less host RAM:** `ENGRAM_RAM=0 ./serve.sh` reads both Engram tables from disk with direct I/O. The server then
uses ~59 GiB of host RAM instead of ~148 GiB, with decode ~4% slower and prefill unchanged on Gen5 NVMe. See the
README for the numbers.

Then run the corruption check, which compares sampled decode tokens against a teacher-forced prefill and must
report `TRUE corruptions 0`, and a benchmark:

```bash
python3 /path/to/this-folder/tools/verify_fix.py 10
cd llm-inference-bench && .venv/bin/python llm_decode_bench.py --port 15015 --model DeepSeek-V4.1-Flash \
  --concurrency 1,4,8 --contexts 32k --coding-peak --output result.json
python3 /path/to/this-folder/tools/bench_steps.py result.json   # steps/s and tokens/step per cell
```

Optional: `tools/llm-bench-flare-steps.patch` (`git apply` inside `llm-inference-bench`, tested on v0.9.1) makes the
benchmark's own summary show the "MTP-normalized decode steps/s" table for Flare, computed from Flare's draft-round counters.

## 6. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Tuning fails with `OutOfMemoryError ... failed to prepare with configuration` | A **cold tuning cache** needs more free VRAM than serving. Export with `HOT=300` first, then the target: the tuning cache in `~/.cache/flare/kernels` is shared and the second export reuses it. A new draft count (`DRAFTS=7`) needs its own warm-up export |
| Export stops at `KeyError: 'FLARE_DS41_BRRR_CKPT'` | you ran `ignition export` directly; use `export.sh` or set `FLARE_DS41_BRRR_CKPT` to the checkpoint view |
| BRRR kernel compile fails (`nvcc`, `ninja`) | `CUDA_HOME` must point at CUDA 13.1 (`nvcc` with `sm_120a`); `ninja` comes with `uv sync`. Compiled kernels are cached in `.brrr-ext/` in the checkout |
| `index total_size does not match tensor payload bytes` | use this folder's `model.safetensors.index.json` (step 3) |
| Export dies on `Cannot patch imported module triton.runtime.build` | Flare's torch/Triton patches are missing: run `uv run --locked --extra prepare python -m ignition.patches` after `uv sync` |
| Chat requests return 400 for `thinking` / `clear_thinking` | Flare's thinking toggle is `chat_template_kwargs: {"enable_thinking": true}` (or `false`); vLLM's `thinking` / `clear_thinking` fields are rejected |
| The command does nothing and exits 0 | your shell's `env` is not `/usr/bin/env` (uv installs one at `~/.local/bin/env`); the scripts here call `/usr/bin/env` explicitly |
| Corruption check reports true corruptions | `flashinfer-tiny-decode-swiglu.patch` is not applied in `.venv` |
| First `flare run` of a new package asks to trust the publisher | packages are signed with your local development key (`~/.config/flare/`); approve it once in an interactive terminal |
| Disk fills up after several exports | remove old versions with `flare models remove <model>@<version>` then `flare models gc` (deleting folders by hand frees nothing: files are shared) |
