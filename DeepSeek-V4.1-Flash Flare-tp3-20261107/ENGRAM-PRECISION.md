# MXFP4 vs FP8 Engram tables: what we measured

DeepSeek-V4.1-Flash-1017 has two Engram n-gram tables (layers 1 and 14). DeepSeek ships them in FP8 (94.4 GiB each,
189 GiB total). NVIDIA's [DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4)
release carries them in MXFP4 (48.6 GiB each, 97 GiB total). That format is lossy: packed E2M1 values with E8M0 block
scales recomputed per 32 values). This page collects every measurement we have of the two side by side: a direct
A/B on one build, then the scores of each release (vLLM Kraken `20261019.4` and later, served with the
`engram-nvidia` weights, and both Flare releases use MXFP4). The release scores compare different builds, so read
them as a record, not a controlled study.

**Summary.** MXFP4 Engram is lossy, and the loss is measurable where Engram matters most: on fact-dense encyclopedia
text, perplexity rises 2.6% and the predicted distributions move about twice as far from FP8 as FP8 moves from
itself between runs. On code there is no measurable change. None of it shows up in answers: TriviaQA, MMLU-Pro,
LiveBench and the reasoning checks all stay within run-to-run noise.

## Direct A/B on one build (2026-11-07)

Same engine, same day, only the Engram tables changed: vLLM Kraken TP3 `.11` served the stock checkpoint (FP8 tables;
layer 1 in RAM, layer 14 from disk) and then the `engram-nvidia` variant (NVIDIA's MXFP4 tables, both in RAM). FP8 ran
twice to measure the engine's own run-to-run noise. vLLM is not bit-reproducible on this stack (fused kernels sum
with float atomics), so two FP8 runs already disagree slightly, and MXFP4 is judged against that floor.

**Teacher-forced next-token probabilities** on 197 fixed 1024-token passages (117 WikiText-103 articles, 80 source
files; 201,531 scored positions), top-20 prompt logprobs at every position. KL is estimated over the reference top-20
plus a residual bucket.

| | FP8 run 1 vs FP8 run 2 (noise floor) | FP8 vs MXFP4 |
| --- | --- | --- |
| Encyclopedia text: top-1 agreement | 95.5% | **93.8%** |
| Encyclopedia text: mean KL (nats/token) | 0.024 | **0.055** |
| Encyclopedia text: perplexity | 2.735 → 2.737 (+0.07%) | 2.735 → **2.805 (+2.6%)** |
| Code: top-1 agreement | 96.7% | 96.2% |
| Code: mean KL | 0.010 | 0.013 |
| Code: perplexity | 2.911 → 2.913 (+0.05%) | 2.911 → 2.907 (−0.2%) |

**Benchmarks**, thinking off, greedy (temperature 0) on vLLM. TriviaQA: 1,000 rc.nocontext validation questions,
normalized exact match against the answer aliases ("contains": an alias appears in the answer). MMLU-Pro: 1,000
questions stratified over its 14 categories, answer letter taken from the reply (up to 2,048 tokens).

| | FP8 run 1 | FP8 run 2 | MXFP4 | Flare 20261107 run 1 | Flare 20261107 run 2 |
| --- | --- | --- | --- | --- | --- |
| TriviaQA exact / contains (of 1,000) | 836 / 881 | 844 / 886 | **838 / 882** | 838 / 880 | 838 / 882 |
| MMLU-Pro (of 1,000) | 774 | 774 | **768** | 776 | 766 |

Flare samples DeepSeek at temperature 1.0 (it does not allow 0), so its runs show the MXFP4 release build with sampling
noise rather than a greedy A/B. Question-level churn: the two FP8 runs both scored 774 on MMLU-Pro while 26 questions
flipped each way; FP8 against MXFP4 flipped 33 one way and 27 the other. On the 980 distinct TriviaQA questions, FP8
run 1 vs run 2 differed by 1 / 9 and FP8 vs MXFP4 by 7 / 10 and 11 / 6. Both benchmark gaps sit inside that churn.

## How different the tables are

| Check | Result |
| --- | --- |
| Value error of MXFP4 against the FP8 tables (first 262K rows of layer 1) | relative RMS 0.120 |
| Our own round-to-nearest MXFP4 conversion of the FP8 tables vs NVIDIA's | block scales identical on 100% of blocks, packed bytes 89% identical (rounding ties) |
| Packed-FP4 lookup kernel vs a reference decode on real rows | bit-exact |
| 43-question greedy eval: answers token-identical to FP8 (vLLM `.4`) | 41/43. One differs only in wording (bellhop riddle). In the other, MXFP4 answered 30 (correct) where FP8 answered 60 |

Greedy open-ended generations diverge from FP8 after the first token whose probability shifts slightly. This is
the expected fingerprint of a lossy table. In the checks we read, the content stayed equivalent.

## Release-by-release scores

Same harness and settings across releases: thinking on, `reasoning_effort` max, temperature 1, top-p 1.0, seed 103
for the profile runs, unless noted. LiveBench is coding, release 2024-11-25, 32K budget, temperature 1.

| Test | FP8 Engram | MXFP4 Engram |
| --- | --- | --- |
| 43-question eval, thinking off | 41-42/43 (vLLM stock, `.2`, `.3`) | 42/43 (vLLM `.4`-`.6`, `.9`, `.11`), 43/43 (vLLM `.8`, `.10`, our converted tables, Flare 20261105, Flare 20261107), 41/43 (vLLM `.7`) |
| LiveBench coding | **81.3** (vLLM stock) | **81.9** (vLLM `.4`/`.5`); 81.5-84.2 across vLLM `.6`-`.10` (7 runs); **82.9** (Flare 20261105); **82.2** (Flare 20261107) |
| hotel-lights, 30 runs | 30/30 (vLLM `.3`, c4) | 30/30 on every vLLM tag (c4); 30/30 Flare 20261105 (c4), 30/30 Flare 20261107 (c8) |
| lavd, 30 runs | 30/30 (vLLM `.3`, c4) | 30/30 Flare 20261105 (c4), 30/30 Flare 20261107 (c8) |
| Long retrieval, 3 markers at 128K / 900K | pass / pass (vLLM `.3`) | pass / pass (Flare 20261105, Flare 20261107) |
| Long generation after a 581K-token conversation, 4 seeds | 4/4, worst 8-gram repeat 0.0005 (vLLM `.3`) | 4/4, worst 0.0013 (Flare 20261105) and 0.0011 (Flare 20261107) |
| Vision, basic / hard set (60 + 60 images, T=0 / T=1) | 60/60, 58-59/60 (vLLM `20261019`) | 60/60, 59/60 (Flare, pre-rebase fork) |
| Tool calls, 60 prompts | not run | 60/60 (Flare 20261105, Flare 20261107) |

LiveBench at temperature 1 moves by a question or two between runs. Six vLLM runs spanned 81.5-83.2, and every
miss was a complete wrong answer on mostly the same hard questions. The single FP8 run (81.3) sits at the bottom of
that range, and the MXFP4 runs span 81.5-84.2: no difference beyond run-to-run noise.

The vLLM tags also differ in kernels and features (BRRR fused decoder from `.6`, sparse attention in `.10`), and the
Flare releases are a different runtime. Several MXFP4 rows therefore compare different builds; the like-for-like rows
are vLLM stock / `.3` against `.4` / `.5`, which changed only the Engram tables.

## Why MXFP4: memory and speed

| | FP8 Engram | MXFP4 Engram |
| --- | --- | --- |
| Table size (both layers) | 189 GiB | 97 GiB |
| vLLM Kraken TP3 (256 GB host) | only layer 1 fits in RAM, layer 14 read from disk: 79.5 steps/s at 32K c1 | both in RAM: **84.4-84.5 steps/s (+6%)**, ~150 GB host RAM |
| Flare TP3 (256 GB host) | both in RAM leaves ~14 GiB free; one in RAM is 9% slower (the other table loses the page cache) | both in RAM with ~100 GiB free, or both on disk at ~59 GiB server RAM and ~4% slower decode |

## Not measured

- An FP8 Engram build on Flare with the full qualification suite; Flare FP8 builds only had corruption checks. (The
  tables are the same files on both engines; the direct A/B above ran on vLLM because it returns prompt logprobs and
  allows greedy decoding.)
- Long-tail factual recall beyond TriviaQA (rare entities, exact dates and numbers), where the 2.6% perplexity rise
  on encyclopedia text would most plausibly turn into wrong answers.
- A same-build A/B of FP8 vs MXFP4 on LiveBench with several runs per side.
