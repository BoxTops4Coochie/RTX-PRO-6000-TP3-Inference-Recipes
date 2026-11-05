# Exporting the package yourself

This walks through building the Flare package in this folder from scratch: toolchain, the patched Flare fork,
the checkpoint, the export, and checks before serving. Budget **about 1.5-2 hours of mostly unattended time** and
**~900 GB of free disk**. If you have the same hardware and only want to serve, download the published package
instead (see the main [README](README.md)).

## 1. Requirements

| | Tested with | Notes |
| --- | --- | --- |
| GPUs | 3 × RTX PRO 6000 Blackwell, 96 GB (SM120), PCIe Gen5 x16 | the export tunes and captures on these exact cards; tested at 350 W with a +6000 MHz memory offset. Kernel tuning measures your cards as they run, so export with the clocks and power limit you will serve with |
| Host RAM | 256 GB (tested: DDR5-6000, 8 channels) | export peaks around 160 GB; serving with Engram in RAM uses ~150 GB |
| Disk | ~900 GB free | checkpoints ~412 GB, package ~407 GB, kernel cache ~45 GB. Tested on two PCIe Gen5 NVMe drives (12 GB/s reads); slower drives only lengthen export, startup and disk-mode Engram |
| Driver | 615.71.09 | any driver supporting CUDA 13.1 |
| CUDA toolkit | 13.1 (`/usr/local/cuda-13.1`) | nvcc 12.x is too old for SM120 |
| Rust | 1.97 (rustup) | builds the `flare` binary and the Heliopause extension |
| uv | 0.11 | Flare's Python environment (Python 3.12, managed by uv) |
| Locked memory | `memlock unlimited` for your user | Flare pins host memory; e.g. `/etc/security/limits.d/90-memlock.conf`: `<user> - memlock unlimited`, then log in again |

```bash
export CUDA_HOME=/usr/local/cuda-13.1 PATH=/usr/local/cuda-13.1/bin:$PATH
ulimit -l        # must print "unlimited"
```

## 2. Build the Flare fork

The fork is upstream NVIDIA Flare at `78b18b1c` plus `source.patch` from this folder.

```bash
git clone https://github.com/NVIDIA/flare-local-inference.git flare
cd flare
git checkout 78b18b1c
git apply /path/to/this-folder/source.patch           # 32 files; applies cleanly to 78b18b1c

uv sync --locked --extra prepare --extra test         # Python environment in .venv (pulls FlashInfer @ ddbe93fc)
cargo build --release --manifest-path flare/Cargo.toml  # the flare binary: flare/target/release/flare
```

**Apply the FlashInfer patches** to the installed FlashInfer (they live inside `.venv`, so re-apply them after
every `uv sync`):

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

From the fork checkout, with GPUs idle (stop any server first):

```bash
CKPT=~/ds41-flare-ckpt /path/to/this-folder/export.sh
```

`export.sh` sets these and runs `ignition export`:

| Setting | Value | Meaning |
| --- | --- | --- |
| `FLARE_DS41_HOST_EXPERT_LAYERS` | `20-33` | routed-expert layers that keep part of their experts in host RAM |
| `FLARE_DS41_HOT_EXPERTS` (`HOT`) | `300` | experts per parked layer kept in VRAM (out of 384); the rest live in host RAM |
| `FLARE_DS41_MERGED_BANK` | `1` | one VRAM + host-RAM expert bank per parked layer (one MoE call) |
| `FLARE_DS41_EXPERT_COUNTS_DIR` | `expert-order/` | route counts that decide which experts are hot |
| `FLARE_DS41_TP3_VOCAB_PARALLEL` | `1` | vocabulary-parallel sampling at TP3 (fixes top_p at 1.0) |
| `FLARE_DS41_TP3_GRAPH_GIB` | `1` | the planner's graph-memory estimate for TP3 |
| `--prefill-chunk-size` (`PREFILL`) | `4096` | prefill chunk; 2048 gives more KV, less prefill speed |
| `--max-concurrency` | `8` | request slots |
| `--mtp-draft-tokens` | `5` | DSpark drafts, NVFP4 draft head, FP8 Markov head |

What you'll see, roughly: weights load (~1 min) → kernel tuning (**20-30 min with a cold cache**, seconds when
cached) → compile (~3 min) → graph capture → weight payloads written (~3 min) → publish. The log ends with
`Exported cache entry: ~/.cache/flare/packages/models/<checkpoint name>/gb202-tp3-<timestamp>`.

**Variants.** Every hot expert per parked layer costs about 93K KV tokens. Measured on this hardware:

| `HOT` | `PREFILL` | KV tokens | Notes |
| --- | --- | --- | --- |
| 300 | 4096 | 2.99M | **this release** |
| 310 | 4096 | 2.07M | decode within noise of 300 |
| 313 | 2048 | 3.47M | most KV; prefill ~15% slower |
| 326 | 2048 | 2.26M | |

## 5. Check and serve

```bash
flare/target/release/flare models list                      # the new version appears as the default
FLARE=flare/target/release/flare PACKAGE=<package path> /path/to/this-folder/serve.sh
```

The server is ready when the log shows `Start server. http://127.0.0.1:15015`, about 60 s with the Engram
tables loading into RAM. The `Allocate cache` line reports the KV token count; it should match the table above
within a few thousand tokens.

Then the corruption check (it compares sampled decode tokens against a teacher-forced prefill; it must report
`TRUE corruptions 0`) and a benchmark:

```bash
python3 /path/to/this-folder/tools/verify_fix.py 10
cd llm-inference-bench && .venv/bin/python llm_decode_bench.py --port 15015 --model DeepSeek-V4.1-Flash \
  --concurrency 1,4,8 --contexts 32k --duration 60 --output result.json
python3 /path/to/this-folder/tools/bench_steps.py result.json   # steps/s and tokens/step per cell
```

## 6. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Tuning fails with `OutOfMemoryError ... failed to prepare with configuration` | A **cold tuning cache** needs more free VRAM than serving. Export once with a lower `HOT` (for example 290), then the target `HOT`: the tuning cache in `~/.cache/flare/kernels` is shared and the second export reuses it |
| `index total_size does not match tensor payload bytes` | use this folder's `model.safetensors.index.json` (step 3) |
| The command does nothing and exits 0 | your shell's `env` is not `/usr/bin/env` (uv installs one at `~/.local/bin/env`); the scripts here call `/usr/bin/env` explicitly |
| Generation fails with `prepared model reported device error 6` | the package was exported without the top-p fix in `source.patch`, or a client asks for top_p < 1 on a package exported before it |
| `unsupported packed BF16 declaration options` at export | stale FlashInfer: run `uv sync --locked` (FlashInfer `ddbe93fc`) and re-apply `tp3-patches/` |
| Corruption check reports true corruptions | `flashinfer-tiny-decode-swiglu.patch` is not applied in `.venv` |
| First `flare run` of a new package asks to trust the publisher | packages are signed with your local development key (`~/.config/flare/`); approve it once in an interactive terminal |
| Disk fills up after several exports | remove old versions with `flare models remove <model>@<version>` then `flare models gc` (deleting folders by hand frees nothing: files are shared) |
