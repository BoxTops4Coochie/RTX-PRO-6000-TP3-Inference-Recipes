# Qualification results — Flare TP3 20261107 (2x build: H342 + BRRR layer tail + MXFP4 Engram in RAM)

Run on the running package without restarts, 450 W, KV 2,373,760 tokens, serve flags as in `../serve.sh` (alpha 0.8,
balanced scheduler). Protocol follows the vLLM Kraken TP3 releases, but runs at **concurrency 8** where the protocol
allows (eval, tool calls, lavd, hotel-lights, LiveBench; the 20261105 release used 4). The whole suite takes about 1 hour
instead of 2.5. Flare deltas are noted in `qual_all.sh` and the scripts: Flare has no `/tokenize` or `/detokenize`,
so prompts are counted with `/v1/messages/count_tokens`; thinking is set via `reasoning_effort`; idle checks read
`flare_*` metrics.

| Check | Result | 20261105 (H300) | vLLM `.11` |
| --- | --- | --- | --- |
| eval-43 (thinking off; Flare clamps temperature to 1.0) | 43/43 | 43/43 | 42/43 |
| Tool calls, 60 at c8, argument validity + second turn | 60/60 | 60/60 (c4) | — |
| lavd, 30 runs at c8, effort max, natural EOS | 30/30 exact | 30/30 (c4) | 30/30 |
| hotel-lights, 30 runs at c8 | 30/30 exact | 30/30 (c4) | 30/30 |
| Long retrieval, 3 markers at 128K / 900K tokens | pass / pass | pass / pass | pass / pass |
| Long generation after a 581,537-token conversation, seeds 201-204 | 4/4, natural stop, 16.7K-28.1K tokens, 8-gram repetition ≤ 0.0011 | 4/4 | 4/4 |
| Decode bench with natural EOS, 32K c1/c4/c8 | 293.8 / 707.7 / 964.3 tok/s, 0 loops | 250.0 / 587.8 / 793.7 (350 W) | — |
| LiveBench coding (2024-11-25 release, 128 questions) | 82.2 (LCB 74.4, completion 90.0) | 82.9 | 81.9 |
| KV stress: 2 waves × 8 × ~530K unique prompts + pinger | PASS: 16/16, pinger 136/136, 9.5M prompt tokens, 14.8 min, min free VRAM 82 MiB, 0 error log lines | PASS | PASS |

LiveBench runs at temperature 1.0, so scores vary by about ±1 between runs. Raw outputs are in `results/` (profile
runs keep their full texts).

Running it: start the server, then `FLARE_LOG=<server log> CKPT=<checkpoint view> ./qual_all.sh` (or name stages:
`./qual_all.sh eval tools`). LiveBench and `llm-inference-bench` are expected in `~/LiveBench` and `~/llm-inference-bench`.
