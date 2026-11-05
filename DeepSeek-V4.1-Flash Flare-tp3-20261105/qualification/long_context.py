"""Natural-EOS qualification; preserve every SSE event and both channels."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import urllib.request

HERE = Path(__file__).resolve().parent / 'outputs'
HERE.mkdir(exist_ok=True)
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "quality", ROOT / "analyze_quality.py")
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def get(path):
    with urllib.request.urlopen("http://127.0.0.1:15015" + path, timeout=10) as r:
        return r.read().decode()


def idle():
    streak = 0
    while streak < 3:
        gpu = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits"], text=True)
        rows = [s for s in get('/metrics').splitlines() if s.startswith(
            ('flare_active_requests', 'flare_pending_requests'))]
        clean = len(rows) >= 2 and all(float(s.rsplit(' ', 1)[1]) == 0 for s in rows)
        clean &= all(float(v) == 0 for s in gpu.splitlines() for v in s.split(',')[1:])
        streak = streak + 1 if clean else 0
        if streak < 3:
            time.sleep(5)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--label', required=True)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=73)
    p.add_argument('--context', choices=('full', 'none'), default='full')
    p.add_argument('--tokens', type=int, default=65536)
    p.add_argument('--prompt', choices=('retrospective', 'audit'), default='retrospective')
    p.add_argument('--natural-eos', action='store_true')
    p.add_argument('--trace', action='store_true')
    p.add_argument('--ban-tool-start', action='store_true')
    p.add_argument('--clear-thinking', action='store_true')
    p.add_argument('--payload', type=Path)
    a = p.parse_args()
    assert a.natural_eos and not a.ban_tool_start and not a.trace
    out = HERE / a.label
    out.mkdir(exist_ok=False)
    cap = a.payload or ROOT / 'captured-request.san.json'
    body = json.loads(cap.read_text())
    body.pop('tools', None)
    body.pop('tool_choice', None)
    ask = ('Ignore any pending tool work. Using the full conversation above, write '
           'an exhaustive technical retrospective of at least 6000 words. Cover: '
           '(1) every distinct problem investigated and its resolution, '
           '(2) the systems and components involved and how they interact, '
           '(3) a chronological narrative of how understanding evolved, '
           '(4) unresolved issues and concrete next steps, and '
           '(5) lessons learned. Use detailed prose with headed sections. Be '
           'thorough and specific; do not summarise briefly.')
    if a.prompt == 'audit':
        ask = json.loads((ROOT.parent / 'r30-port/results/long-context/manifest.json')
                         .read_text())['appended_instruction']
    body['messages'] = (body['messages'] if a.context == 'full' else []) + [
        {'role': 'user', 'content': ask}]
    body.update(model='DeepSeek-V4.1-Flash', temperature=a.temperature, top_p=a.top_p,
                seed=a.seed, max_completion_tokens=a.tokens,
                min_tokens=0, ignore_eos=False, repetition_penalty=1.0,
                tool_choice='none', stream=True,
                stream_options={'include_usage': True, 'continuous_usage_stats': True},
                cache_salt='ds41-qualification-' + a.label,
                reasoning_effort='max')
    if a.ban_tool_start:
        body['bad_words'] = ['<tool_call>']
    body['return_token_ids'] = True
    encoded = json.dumps(body, ensure_ascii=False).encode()
    canonical = {k:v for k,v in body.items() if k not in ('cache_salt', 'seed')}
    canonical_sha256 = hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    idle()
    powers = subprocess.check_output(['nvidia-smi', '--query-gpu=power.limit',
                                     '--format=csv,noheader,nounits'], text=True)
    assert all(float(v) <= 400 for v in powers.splitlines()), powers
    runtime = {'server': 'flare', 'cmdline': subprocess.check_output(['bash', '-c', "ps -eo args | grep 'target/release/[f]lare run' | head -1"], text=True).strip()}
    (out / 'manifest.json').write_text(json.dumps({
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        'request_sha256': hashlib.sha256(encoded).hexdigest(),
        'canonical_without_seed_or_cache_salt_sha256': canonical_sha256,
        'capture_sha256': hashlib.sha256(cap.read_bytes()).hexdigest(),
        'settings': {k:v for k,v in body.items() if k != 'messages'},
        'appended_prompt': ask, 'runtime': runtime, 'power_limits': powers}, indent=2))
    (out / 'before.metrics').write_text(get('/metrics'))
    if a.trace:
        request = urllib.request.Request('http://127.0.0.1:15015/collective_rpc',
            data=json.dumps({'method':'reset_kda_trace'}).encode(),
            headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=30) as response:
            reset = json.load(response)
        (out/'trace-reset.json').write_text(json.dumps(reset, indent=2))
        assert all(r['layers'] for r in reset['results']), reset
    print('START', a.label, flush=True)
    start = time.monotonic()
    last = start
    first = None
    channels = {'content': [], 'reasoning': []}
    usage = None
    finishes = []
    output_ids = []
    stop_reasons = []
    tool_calls = {}
    first_tool_completion = None
    done = False
    request = urllib.request.Request('http://127.0.0.1:15015/v1/chat/completions',
        data=encoded, headers={'Content-Type': 'application/json'})
    with (out / 'stream.sse').open('wb') as rawlog:
        with urllib.request.urlopen(request, timeout=1800) as response:
            for raw in response:
                rawlog.write(raw)
                if not raw.startswith(b'data: '):
                    continue
                if raw.strip() == b'data: [DONE]':
                    done = True
                    continue
                event = json.loads(raw[6:])
                if event.get('error'):
                    raise RuntimeError(event['error'])
                usage = event.get('usage') or usage
                for choice in event.get('choices', []):
                    output_ids.extend(choice.get('token_ids') or [])
                    if choice.get('stop_reason') is not None:stop_reasons.append(choice['stop_reason'])
                    delta = choice.get('delta') or {}
                    for tc in delta.get('tool_calls') or []:
                        if first_tool_completion is None and usage:
                            first_tool_completion = usage.get('completion_tokens')
                        entry = tool_calls.setdefault(tc['index'], {'name': '', 'arguments': ''})
                        for field in ('name', 'arguments'):
                            entry[field] += (tc.get('function') or {}).get(field) or ''
                    for key, value in [('content', delta.get('content')), ('reasoning',
                            delta.get('reasoning') or delta.get('reasoning_content'))]:
                        if value:
                            channels[key].append(value)
                            if first is None:
                                first = time.monotonic()
                    if choice.get('finish_reason'):
                        finishes.append(choice['finish_reason'])
                if time.monotonic() - last > 30:
                    rawlog.flush()
                    print('PROGRESS', a.label, usage, 'chars',
                        {k: sum(map(len,v)) for k,v in channels.items()}, flush=True)
                    last = time.monotonic()
    record = {k: ''.join(v) for k,v in channels.items()}
    record.update(usage=usage, finish_reasons=finishes, saw_done=done,
        wall_s=time.monotonic()-start, first_visible_s=None if first is None else first-start)
    record.update(tool_calls=tool_calls, first_tool_completion=first_tool_completion, output_token_ids=output_ids, stop_reasons=stop_reasons)
    if output_ids:
        # Flare has no /detokenize: decode locally with the checkpoint tokenizer (artifact only; scoring uses the stream).
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file('/m2-2/flare-ckpt/ds41-1017-engram-mxfp4/tokenizer.json')
            raw_text = {'prompt': tok.decode(output_ids, skip_special_tokens=False)}
        except Exception as error:  # noqa: BLE001
            raw_text = {'unavailable': f'{type(error).__name__}: {error}'}
        (out/'raw-detokenized.json').write_text(json.dumps(raw_text,ensure_ascii=False,indent=2))
    record['quality'] = {k: quality.analyze(record[k]) for k in channels}
    record['flags'] = {k: quality.flags(record['quality'][k]) for k in channels}
    record['qualifying_length'] = bool(usage and usage.get('completion_tokens',0) >= 25000
        and sum(len(record[k]) for k in channels) >= 25000
        and (first_tool_completion is None or first_tool_completion >= 25000))
    (out/'result.json').write_text(json.dumps(record, indent=2))
    (out/'after.metrics').write_text(get('/metrics'))
    if a.trace:
        request = urllib.request.Request('http://127.0.0.1:15015/collective_rpc',
            data=json.dumps({'method':'get_kda_trace'}).encode(),
            headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=60) as response:
            (out/'state-trace.json').write_bytes(response.read())
    current = {'server': 'flare', 'cmdline': subprocess.check_output(['bash', '-c', "ps -eo args | grep 'target/release/[f]lare run' | head -1"], text=True).strip()}
    assert current == runtime, 'Service changed during trial'
    print('DONE', a.label, json.dumps({k:v for k,v in record.items()
        if k not in (*channels, 'tool_calls', 'output_token_ids')}), 'tool_count', len(tool_calls), flush=True)
    assert done and usage, 'Incomplete stream'


if __name__ == '__main__':
    main()
