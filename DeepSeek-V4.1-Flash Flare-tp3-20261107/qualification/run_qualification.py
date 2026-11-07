#!/usr/bin/env python3
"""Quality and long-context qualification of the running DS4.1 TP3 server.

Same protocol as the GLM-5.3 Kraken TP3 releases, adapted to DeepSeek-V4.1-Flash
(thinking on, reasoning_effort max, temperature 1, top-p QUAL_TOP_P, default 1.0 as on
the model card and the server):
  - llm_decode_bench profiles lavd and hotel-lights: 30 runs each, seed 103,
    100,000-token cap, at concurrency QC (default 8, the server's sequence limit)
  - long retrieval: three exact markers in 128K- and 900K-token prompts
  - long generation after the captured 581,537-token conversation, seeds 201-204,
    natural EOS, 65,536-token cap, scored for 8-gram repetition
Writes RESULTS.md and the raw outputs next to this file.
"""
import datetime, json, os, pathlib, subprocess, sys
D = pathlib.Path(__file__).resolve().parent
# llm-inference-bench checkout (https://github.com/local-inference-lab/llm-inference-bench).
BENCH = os.environ.get('LLM_BENCH', str(pathlib.Path.home() / 'llm-inference-bench/llm_decode_bench.py'))
PY = sys.executable
TOP_P = os.environ.get('QUAL_TOP_P', '1.0')

def note(m):
    print(m, flush=True)
    with (D / 'RESULTS.md').open('a') as f:
        f.write(datetime.datetime.now(datetime.timezone.utc).isoformat()[:19] + ' — ' + m + '\n')

def profile(name, conc):
    tag = f'{name}-c{conc}'
    cmd = [PY, BENCH, '--host', 'localhost', '--port', '15015', '--model', 'DeepSeek-V4.1-Flash',
           '--test-profile', name, '--profile-runs', '30', '--profile-concurrency', str(conc),
           '--max-tokens', '100000', '--reasoning-effort', 'max',
           '--completion-stats-top-p', TOP_P, '--completion-stats-temperature', '1',
           '--completion-stats-seed', '103', '--completion-stats-save-text',
           '--completion-stats-no-prefill-scout', '--respect-eos', '--display-mode', 'plain',
           '--output', str(D / f'{tag}.json')]
    note(f'START {tag}')
    with (D / f'{tag}.log').open('w') as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, input=b'n\n').returncode
    runs = json.load(open(D / f'{tag}.json'))['runs']
    wrong = [(x['run_index'], x['score_detail'], x['completion_tokens']) for x in runs
             if not x.get('correct') and not x.get('hit_max_tokens')]
    note(f'DONE {tag} rc={rc} | exact {sum(1 for x in runs if x.get("correct"))} / wrong {len(wrong)}'
         f' / truncated {sum(1 for x in runs if x.get("hit_max_tokens"))} | {wrong}')

def judge(r):
    q = r['quality']['content']; c = r['content']
    return dict(empty=not c.strip(), repeat=q['repeat_8gram'] > 0.05,
                thin=len(c) < 2000 and r['usage']['completion_tokens'] > 1500)

note(f'top-p {TOP_P}')
stages = sys.argv[1:] or ['lavd', 'hotel-lights', 'retrieval', 'longgen']
(D / 'runtime.json').write_bytes(json.dumps({'server': 'flare', 'cmdline': subprocess.check_output(['bash','-c',"ps -eo args | grep 'target/release/[f]lare run' | head -1"], text=True).strip()}).encode())
if 'lavd' in stages:
    profile('lavd', int(os.environ.get('QC', '8')))
if 'hotel-lights' in stages:
    profile('hotel-lights', int(os.environ.get('QC', '8')))
if 'retrieval' in stages:
    note('START long retrieval 128K/900K')
    with (D / 'long-retrieval.log').open('w') as f:
        rc = subprocess.run([PY, str(D / 'long_retrieval.py'), '--label', 'release', '--top-p', TOP_P], stdout=f,
                            stderr=subprocess.STDOUT, timeout=7200).returncode
    note(f'DONE long retrieval rc={rc}: ' + ' '.join(l.split(' {')[0] + (' pass' if '"passed": true' in l else ' FAIL')
         for l in (D / 'long-retrieval.log').read_text().splitlines() if l.startswith('DONE')))
if 'longgen' in stages:
    rows = []
    for seed in (201, 202, 203, 204):
        tag = f'longgen-{seed}'
        with (D / f'{tag}.log').open('w') as f:
            rc = subprocess.run([PY, str(D / 'long_context.py'), '--label', tag, '--natural-eos',
                                 '--top-p', TOP_P, '--seed', str(seed), '--tokens', '65536'],
                                stdout=f, stderr=subprocess.STDOUT, timeout=5400).returncode
        if rc:
            note(f'{tag} RUNNER rc={rc}'); continue
        r = json.loads((D / 'outputs' / tag / 'result.json').read_text())
        v = judge(r)
        row = dict(seed=seed, content_chars=len(r['content']), completion=r['usage']['completion_tokens'],
                   finish=r.get('finish_reasons'), repeat_8gram=r['quality']['content']['repeat_8gram'],
                   ttr=r['quality']['content']['worst_window_ttr'], fail=any(v.values()), **v)
        rows.append(row)
        (D / 'long-generation.json').write_text(json.dumps(rows, indent=2))
        note(json.dumps(row))
note(f'done (top-p {TOP_P})')
