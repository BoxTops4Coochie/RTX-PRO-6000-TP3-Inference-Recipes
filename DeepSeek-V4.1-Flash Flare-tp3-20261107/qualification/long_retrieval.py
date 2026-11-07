"""Natural-EOS long-context retrieval with exact known answers at three positions."""
import hashlib
import argparse
import json
import re
from pathlib import Path
import subprocess
import urllib.request

HERE = Path(__file__).resolve().parent
OUT = HERE / 'long-retrieval'
OUT.mkdir(exist_ok=True)
URL = 'http://127.0.0.1:15015'
unit = 'This ordinary archived entry contains no requested marker.\n'
expected = {'A': 'ALPHA-R34-7K2M9Q', 'B': 'BETA-R34-8P6V3X', 'C': 'GAMMA-R34-4T9N2Z'}


def count_tokens(msgs):
    """Flare: exact prompt token count via /v1/messages/count_tokens (no vLLM /tokenize)."""
    import urllib.request
    body = dict(model='DeepSeek-V4.1-Flash', max_tokens=1, messages=msgs)
    req = urllib.request.Request(URL + '/v1/messages/count_tokens', json.dumps(body).encode(),
        {'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)['input_tokens']

def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.load(response)


def wait_idle():
    """Three consecutive 5 s samples with no running or waiting requests."""
    import time
    streak = 0
    while streak < 3:
        with urllib.request.urlopen(URL + '/metrics', timeout=10) as r:
            rows = [s for s in r.read().decode().splitlines()
                    if s.startswith(('flare_active_requests', 'flare_pending_requests'))]
        clean = len(rows) >= 2 and all(float(s.rsplit(' ', 1)[1]) == 0 for s in rows)
        streak = streak + 1 if clean else 0
        time.sleep(5)


def messages(n):
    text = ('Read this archive and retrieve its three marker values.\n'
            'Marker A = ' + expected['A'] + '\n' + unit * (n//2)
            + 'Marker B = ' + expected['B'] + '\n' + unit * (n-n//2)
            + 'Marker C = ' + expected['C'] + '\n'
            'Return only a JSON object with keys A, B, C and their exact marker values.')
    return [{'role': 'user', 'content': text}]


parser = argparse.ArgumentParser()
parser.add_argument('--label', required=True)
parser.add_argument('--top-p', type=float, default=1.0)
parser.add_argument('--targets', nargs='+', type=int, default=[128000, 900000])
args = parser.parse_args()
OUT = OUT / args.label
OUT.mkdir(exist_ok=False)
for target in args.targets:
    wait_idle()
    n = target // 10
    attempts = []
    for _ in range(8):
        msgs = messages(n)
        count = count_tokens(msgs)
        attempts.append({'records': n, 'tokens': count})
        if abs(count-target) < 100:
            break
        n = max(1, round(n * target/count))
    assert abs(count-target) < 100, attempts
    body = dict(model='DeepSeek-V4.1-Flash', messages=msgs, temperature=1, top_p=args.top_p,
        seed=target, min_tokens=0, ignore_eos=False, max_completion_tokens=2048,
        repetition_penalty=1, tool_choice='none', cache_salt=f'ds41-{args.label}-retrieval-{target}',
        reasoning_effort='max')
    request_bytes = json.dumps(body).encode()
    out = OUT / str(target)
    out.mkdir(exist_ok=False)
    (out/'manifest.json').write_text(json.dumps(dict(target=target,
        request_sha256=hashlib.sha256(request_bytes).hexdigest(), attempts=attempts,
        settings={k:v for k,v in body.items() if k!='messages'}, expected=expected,
        runtime={'server': 'flare', 'cmdline': subprocess.check_output(['bash', '-c', "ps -eo args | grep 'target/release/[f]lare run' | head -1"], text=True).strip()}), indent=2))
    print('START',target,'actual prompt',count,flush=True)
    response = post('/v1/chat/completions', body)
    (out/'response.json').write_text(json.dumps(response,indent=2))
    choice = response['choices'][0]
    text = choice['message'].get('content') or ''
    strict_json = True
    try:
        actual = json.loads(text)
    except ValueError:
        strict_json = False
        fenced = re.fullmatch(r'```(?:json)?\s*\n(.*)\n```', text.strip(), re.DOTALL)
        try:
            actual = json.loads(fenced.group(1)) if fenced else None
        except ValueError:
            actual = None
    passed = (actual == expected and choice['finish_reason']=='stop'
              and not choice['message'].get('tool_calls')
              and response['usage']['prompt_tokens']==count)
    receipt = dict(passed=passed,strict_json=strict_json,actual=actual,expected=expected,usage=response['usage'],
                   finish_reason=choice['finish_reason'])
    (out/'result.json').write_text(json.dumps(receipt,indent=2))
    print('DONE',target,json.dumps(receipt),flush=True)
    assert passed, receipt
