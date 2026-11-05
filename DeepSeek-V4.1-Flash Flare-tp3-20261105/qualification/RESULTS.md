# Qualification results — Flare TP3 20261105 (H300 + MXFP4 Engram in RAM + prefill 4096)

Run on the running package without restarts, 350 W, KV 2,991,488 tokens. Protocol follows the vLLM Kraken TP3
releases; Flare deltas are noted in `qual_all.sh` and the scripts (no `/tokenize` or `/detokenize`: prompts are
counted with `/v1/messages/count_tokens`; thinking via `reasoning_effort`; idle checks read `flare_*` metrics).

| Check | Result | vLLM `.11` |
| --- | --- | --- |
| eval-43 (thinking off; Flare clamps temperature to 1.0) | 43/43 | 42/43 |
| Tool calls, 60 at c4, argument validity + second turn | 60/60 | — |
| lavd, 30 runs at c4, effort max, natural EOS | 30/30 exact | 30/30 |
| hotel-lights, 30 runs at c4 | 30/30 exact | 30/30 |
| Long retrieval, 3 markers at 128K / 900K tokens (127,986 / 899,988 actual) | pass / pass | pass / pass |
| Long generation after a 581,537-token conversation, seeds 201-204 | 4/4, natural stop, 17.6K-22.5K tokens, 8-gram repetition ≤ 0.0013 | 4/4 |
| Decode bench with natural EOS, 32K c1/c4/c8 | 250.0 / 587.8 / 793.7 tok/s, 0 loops | — |
| LiveBench coding (2024-11-25 release, 128 questions) | 82.9 (LCB 71.8, completion 94.0) | 81.9 |
| KV stress: 2 waves × 8 × ~530K unique prompts + pinger | PASS: 16/16, pinger 137/137, 9.5M prompt tokens, 17.2 min, 0 error log lines | PASS |

Raw outputs are in `results/` (profile runs keep their full texts).
