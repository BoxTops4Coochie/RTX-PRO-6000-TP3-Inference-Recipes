"""Print steps/s and tokens/step per cell from llm_decode_bench (>=0.9) JSON results for a Flare server."""
import json, sys
for path in sys.argv[1:]:
    d = json.load(open(path))
    print(path.rsplit("/", 1)[-1])
    print(f"  {'ctx':>6} {'conc':>4} {'tok/s':>7} {'steps/s':>8} {'tok/step':>8} {'accept':>6} {'GPU busy':>8} {'engram wait':>11}")
    for r in d["results"]:
        f = r.get("server_flare_metrics") or {}
        if not f.get("valid"):
            continue
        c, s = f["counter_delta"], f["measurement_seconds"]
        steps = f["verification_graphs_per_second"]
        tps = r.get("aggregate_tps", 0)
        print(f"  {r.get('context_label', r.get('context', '?')):>6} {r.get('concurrency', '?'):>4} {tps:7.1f} {steps:8.1f} "
              f"{c['raw_emitted_tokens'] / c['verification_graphs']:8.2f} {f['raw_draft_acceptance']:6.2f} "
              f"{c['verification_gpu_nanoseconds'] / 1e9 / s * 100:7.0f}% {c['embedding_readiness_nanoseconds'] / 1e9 / s * 100:10.1f}%")
