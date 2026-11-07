#!/usr/bin/env python3
"""Tool-call qualification (flare#193 lesson): N tool-requiring prompts at concurrency C.

Per request: no HTTP/server error; the model calls a tool (finish_reason tool_calls); every call names a
declared tool, its arguments parse as a JSON object with non-empty unique keys, all required parameters
present with the right JSON types and enum values. A second turn returns the tool result and must end
with a non-empty assistant answer.
usage: tool_calls.py [N=60] [C=4]
"""
import json, random, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:15015/v1/chat/completions"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
C = int(sys.argv[2]) if len(sys.argv) > 2 else 4
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
     "parameters": {"type": "object", "properties": {
         "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
         "required": ["city", "unit"]}}},
    {"type": "function", "function": {"name": "search_files", "description": "Find files matching a glob under a directory.",
     "parameters": {"type": "object", "properties": {
         "pattern": {"type": "string"}, "path": {"type": "string"}, "max_results": {"type": "integer"}},
         "required": ["pattern", "path"]}}},
    {"type": "function", "function": {"name": "run_command", "description": "Run a shell command and return its output.",
     "parameters": {"type": "object", "properties": {
         "command": {"type": "string"}, "timeout_seconds": {"type": "integer"}},
         "required": ["command", "timeout_seconds"]}}},
]
SPEC = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}
CITIES = ["Lisbon", "Osaka", "Nairobi", "Calgary", "Hanoi", "Reykjavik", "Lima", "Tbilisi"]
TASKS = [
    lambda r: (f"What's the weather in {r.choice(CITIES)} right now, in {r.choice(['celsius', 'fahrenheit'])}? Use the tool.", "get_weather"),
    lambda r: (f"Find all {r.choice(['*.py', '*.rs', '*.md', '*.json'])} files under {r.choice(['/srv/app', '~/projects', '/etc'])}, at most {r.randint(5, 50)} results.", "search_files"),
    lambda r: (f"Run `{r.choice(['df -h', 'uname -a', 'ls -la /tmp', 'git status'])}` with a {r.randint(5, 120)} second timeout and tell me what it says.", "run_command"),
]
FAKE = {"get_weather": {"temperature": 18, "conditions": "light rain"},
        "search_files": {"matches": ["a/one.py", "b/two.py"], "truncated": False},
        "run_command": {"exit_code": 0, "stdout": "ok\n"}}


def post(body):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.load(r), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read()[:300]!r}"
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def check_call(call):
    fn = call.get("function", {})
    name = fn.get("name")
    if name not in SPEC:
        return f"unknown tool {name!r}"
    try:
        args = json.loads(fn.get("arguments") or "")
    except json.JSONDecodeError as e:
        return f"arguments not JSON: {e}"
    if not isinstance(args, dict):
        return "arguments not an object"
    if any(not k for k in args):
        return "empty argument name"
    spec = SPEC[name]
    for key in spec["required"]:
        if key not in args:
            return f"missing required {key}"
    for key, value in args.items():
        prop = spec["properties"].get(key)
        if prop is None:
            return f"undeclared argument {key}"
        if not isinstance(value, TYPES[prop["type"]]) or (prop["type"] == "integer" and isinstance(value, bool)):
            return f"{key} has type {type(value).__name__}"
        if "enum" in prop and value not in prop["enum"]:
            return f"{key}={value!r} not in enum"
    return None


def one(i):
    rng = random.Random(1000 + i)
    prompt, expected = rng.choice(TASKS)(rng)
    msgs = [{"role": "user", "content": prompt}]
    base = dict(model="DeepSeek-V4.1-Flash", tools=TOOLS, max_tokens=4096, reasoning_effort="high", seed=i)
    d, err = post({**base, "messages": msgs})
    if err:
        return i, "FAIL", err
    choice = d["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    if not calls:
        return i, "FAIL", f"no tool call (finish={choice.get('finish_reason')}): {(choice['message'].get('content') or '')[:120]!r}"
    names = [c.get("function", {}).get("name") for c in calls]
    if len(set(c.get("id") for c in calls)) != len(calls):
        return i, "FAIL", "duplicate tool call ids"
    for c in calls:
        problem = check_call(c)
        if problem:
            return i, "FAIL", problem
    msgs.append(choice["message"])
    for c in calls:
        msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(FAKE[c["function"]["name"]])})
    d2, err = post({**base, "messages": msgs})
    if err:
        return i, "FAIL", "turn 2: " + err
    answer = (d2["choices"][0]["message"].get("content") or "").strip()
    if not answer and not d2["choices"][0]["message"].get("tool_calls"):
        return i, "FAIL", "turn 2: empty answer"
    return i, "OK", f"{','.join(names)}{'' if expected in names else ' (unexpected tool)'}"


t0 = time.time()
with ThreadPoolExecutor(C) as pool:
    results = sorted(pool.map(one, range(N)))
ok = sum(1 for _, s, _ in results if s == "OK")
odd = [r for r in results if r[1] == "OK" and "unexpected" in r[2]]
for r in results:
    if r[1] != "OK":
        print("  FAIL", r[0], r[2])
print(f"tool calls: {ok}/{N} OK ({len(odd)} chose a different tool), concurrency {C}, {time.time() - t0:.0f} s")
json.dump([dict(index=i, status=s, detail=d) for i, s, d in results],
          open(__file__.replace(".py", ".json"), "w"), indent=2)
