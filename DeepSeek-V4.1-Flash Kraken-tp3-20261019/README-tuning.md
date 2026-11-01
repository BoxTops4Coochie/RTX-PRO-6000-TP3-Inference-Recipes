# DS41 kraken TP3 — tuning notes (keeper config)

Image: `azallaza/ds41-kraken-tp3:20261019.3` (unchanged — all tuning below is
runtime env + mounted patches via `compose.override.yaml`, no image rebuild).

## Keeper config (shipped defaults in compose.yaml as of 2026-10-25)

| setting | value | why |
|---|---|---|
| `DS41_OFFLOAD_FROM/TO` | **20 / 33** | 14-layer offload+promote range. llm-bench **72.3 decode steps @32K vs 68.0 at the old 20-25 (+6.3%)**. Spreading the fixed ~6 GiB of host-parked experts across more layers makes each layer's cold tail shallower (cold-fetch rate ~6.7%→~0.5%). |
| `RESERVE_BYTES` | **1610612736 (1.5 GiB)** | Promotion free-VRAM floor. Required by 20-33: at 1 GiB it runs ~5 MiB free under load and throws OOM-mapping retries (crash precursor). 1.5 GiB → min free ~480 MiB under 4×600K concurrent long-context stress, zero retries. |
| launch `max-num-batched-tokens` | 4096 (ds41-launch.yaml mount) | Prefill throughput at 32-128K (2048 cost ~40% long-ctx prefill for ~+8% decode — rejected). |
| rejection sample | block (profile mount) | Lossless, neutral speed. |
| GPU power | 450 W (`nvidia-pl.service`) + core +100 / mem +6000 | Power is a **prefill** lever (~+10% at 450 W); does NOT help decode (clock-bound). |

## Hard limits found (don't cross)
- **Offload range ≤ ~34.** 20-34/36 OOM at *load* — cf=1 host-maps every wrapped
  layer's experts upfront (~110 GiB for 17 layers) before promotion trims it.
- **Keep draft layers 37-39 resident** (dspark hits them ~6×/step).
- **cf<1 (`DS41_VMM_COLD_FRACTION`)** lets you wrap more layers but under-promotes
  (leaves VRAM unused) → slower. 20-36/20-39 @ cf=0.2 = 69.7/70.3 steps (lose to 20-30).

## Dormant toggles (tried, measured neutral/negative, left OFF)
- `DS41_EP_B12X_ALLREDUCE=0` — ep:0 via b12x PCIe = neutral (not on steady decode path).
- `VLLM_MXFP8_LM_HEAD=0` — LM head→MXFP8: no speed (72.3→71.8), freed ~1 GiB unneeded
  headroom, small logit-precision cost. (Input embed has no quant hook — lookup.)
- `DS41_DRAFT_TP=` (empty = TP3) — draft-TP1 neutral for c1.
- `B12X_EXPERT_ORDER_DIR` default = general; set `/opt/ds41/expert-order-coding` for
  coding-heavy workloads (+21% c4 coding, ~-7% general).

## Not levers for decode (confirmed dead ends)
NCCL EP all-reduce, GPU power/clocks, the b12x MoE kernel (autotuned), porting MOET
kernels (kraken already faster), KV FP4 (capacity not speed; hurts long-ctx accuracy),
lowering dspark k (coding acceptance is high — keep k=5). BAR1 PCIe P2P driver setting
is the big one and is **already enabled** (`/etc/modprobe.d/nvidia-p2p-override.conf`).

## Expert promotion order from live traffic (2026-10-30) — KEEPER

**Result:** offloaded layers 20-33 were losing ~1.1 ms/step (~8%) to cold-expert reads
over PCIe because every shipped order (baked general, expert-order-codinglog) was built
from a ~17.8K-token sample. An order built from ~554K routed tokens of real traffic
cut the excess to 0.15 ms/step: MoE kernel avg 126 -> 104 us, decode step
14.1 -> 13.5 ms (70.7 -> 74.1 steps/s, +4.8%), offloaded layers now at resident speed.

Live config: `.env` `B12X_EXPERT_ORDER_DIR=/opt/ds41/expert-order-live`, dir
`expert-order-live/` mounted by compose.override.yaml. Comment the .env line out to
return to the baked order.

**How to rebuild/refine the order (procedure):**
1. `.env`: `B12X_EXPERT_ROUTES_ALL=1` (NOT `B12X_EXPERT_ROUTES` — that counter only
   exists in deferred-weight-banks mode, never constructed in our in-place VMM mode),
   restart. Every MoE layer then dumps `/cache/all-expert-routes-rank<r>-layer<L>.json`
   every 30 s (cumulative counts for the life of the container; ~0.5-1% step overhead).
2. Run real traffic. Counts = decode rows (all ~6 speculative candidates per step);
   prefill rows are not counted. Aim for >=500K routed tokens.
3. **PITFALL — id space:** the monitor records `topk_ids` AFTER the hot-first relabel,
   so for the offloaded layers (DS41_OFFLOAD_FROM..TO) the counts are indexed by the
   *slot* of whatever order was loaded at the time, not by expert id. Un-permute:
   `order = sorted(range(384), key=lambda e: (-applied_counts[e], e))` (the order file
   that was loaded), then `by_expert[order[s]] = live[s]`. Resident layers are not
   relabeled (counts already in expert-id space). Feeding slot-space counts straight in
   produced an ANTI-correlated order: 5.0 ms/step excess, 54 steps/s (observed).
4. Write `expert-routes-rank0-layer<L>.json` (`{"counts":[384 ints], ...}`) into a new
   dir, mount it, point `B12X_EXPERT_ORDER_DIR` at it, restart (loader relabels at load;
   log line `Hot-first expert order for layers [...]` confirms).
5. Verify with the profiler (`prof-analysis/profile_run.sh` + `step_overhead.py`):
   per-layer MoE kernel medians on 20-33 should sit at ~90-110 us like resident layers.

### UPDATE (2026-10-30, later): single-workload orders overfit — use the BLEND. KEEPER = `expert-order-blend`
The code-only live order (above) was fast on coding traffic but **collapsed on prose**:
llm-inference-bench 32k c1 fell to 54 steps/s (baseline 70-72) because the bench decodes
encyclopedic prose and code/prose share only ~285 of 320 hot experts per layer (cold rate
14-22% for the "other" workload). Fix: blend the normalized distributions
**0.6 code (544K tok real traffic) + 0.4 prose (109K tok of the bench)** -> ~1.5% cold on
BOTH. Results at KV 2.25 GiB (promotion 20/24,19/24): **bench 32k c1 = 75.3 steps/s
(207 tok/s), +4-7% vs baseline; coding MoE kernel avg 100 us = resident-layer speed.**
Live: `.env` `B12X_EXPERT_ORDER_DIR=/opt/ds41/expert-order-blend` (dir mounted).
Rule: an order must be built from a sample that covers EVERY workload you care about.
Re-blend when traffic mix changes (counts collected now are in blend-slot space for
layers 20-33 -> un-permute through expert-order-blend/).

**KV trade (tested with the code-only order):** KV 4.75 GiB (`KV_CACHE_MEMORY_BYTES=
5100273664`) -> 4.61M tokens KV (2.1x) at promotion 17/24 cost ~-2.6% decode on code
(74.1 -> 72.2 steps/s). Under the blend the shallower depth costs more (~3-3.5% cold
in-sample); NOT adopted yet — re-test if more KV is wanted. Line left commented in .env.
