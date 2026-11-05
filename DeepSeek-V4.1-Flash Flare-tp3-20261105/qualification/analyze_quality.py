"""Stricter degeneration analysis than the shipped soup detector.

The bundle's detector uses a *global* soup ratio plus a repeated-character run.
Manual review found degraded reasoning in cold-01/05/07 that it labelled `ok`,
so this adds the two things a global ratio misses:

  * WINDOWED analysis - a clean 5k-token preamble dilutes a corrupt tail, so
    every metric is also computed over sliding windows and reported at its worst.
  * LOOPING - verbatim n-gram repetition, which is semantic collapse that
    contains no unusual punctuation at all and is therefore invisible to a
    punctuation-soup detector.

Reports per-file metrics and a FLAG when any window looks degenerate. This is a
screening aid, not ground truth; excerpts are printed so a human can judge.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter

WORD_RE = re.compile(r"\S+")
RUN_RE = re.compile(r"(.)\1{19,}", re.DOTALL)          # >=20 identical chars
PUNCT_RE = re.compile(r"[^\w\s]")
WIN_CHARS = 2000
STEP = 1000


def windows(text: str, size: int = WIN_CHARS, step: int = STEP):
    if len(text) <= size:
        yield 0, text
        return
    for i in range(0, len(text) - size + 1, step):
        yield i, text[i:i + size]


def punct_ratio(s: str) -> float:
    return (len(PUNCT_RE.findall(s)) / len(s)) if s else 0.0


def repeat_ngram_ratio(text: str, n: int = 8) -> float:
    """Fraction of n-grams that are duplicates -> verbatim looping."""
    w = WORD_RE.findall(text)
    if len(w) < n * 2:
        return 0.0
    grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    c = Counter(grams)
    dup = sum(v - 1 for v in c.values() if v > 1)
    return dup / len(grams)


def type_token_ratio(text: str) -> float:
    w = [x.lower() for x in WORD_RE.findall(text)]
    return (len(set(w)) / len(w)) if w else 1.0


def analyze(text: str) -> dict:
    if not text:
        return {"chars": 0, "empty": True}
    longest_word = max((len(x) for x in WORD_RE.findall(text)), default=0)
    run = RUN_RE.search(text)
    worst_punct = 0.0
    worst_punct_at = 0
    worst_ttr = 1.0
    worst_ttr_at = 0
    for off, win in windows(text):
        p = punct_ratio(win)
        if p > worst_punct:
            worst_punct, worst_punct_at = p, off
        t = type_token_ratio(win)
        if t < worst_ttr:
            worst_ttr, worst_ttr_at = t, off
    # tail vs head: degradation usually accumulates
    half = len(text) // 2
    return {
        "chars": len(text),
        "longest_word": longest_word,
        "char_run": len(run.group(0)) if run else 0,
        "global_punct": round(punct_ratio(text), 4),
        "worst_window_punct": round(worst_punct, 4),
        "worst_window_punct_at": worst_punct_at,
        "worst_window_ttr": round(worst_ttr, 3),
        "worst_window_ttr_at": worst_ttr_at,
        "repeat_8gram": round(repeat_ngram_ratio(text), 4),
        "head_punct": round(punct_ratio(text[:half]), 4),
        "tail_punct": round(punct_ratio(text[half:]), 4),
    }


def flags(m: dict) -> list[str]:
    f = []
    if m.get("empty"):
        return ["EMPTY"]
    if m["longest_word"] >= 200:
        f.append(f"longword={m['longest_word']}")
    if m["char_run"] >= 20:
        f.append(f"charrun={m['char_run']}")
    if m["worst_window_punct"] >= 0.25:
        f.append(f"punct_window={m['worst_window_punct']}@{m['worst_window_punct_at']}")
    if m["repeat_8gram"] >= 0.15:
        f.append(f"loop8={m['repeat_8gram']}")
    if m["worst_window_ttr"] <= 0.25:
        f.append(f"lowTTR={m['worst_window_ttr']}@{m['worst_window_ttr_at']}")
    if m["tail_punct"] >= 2.5 * max(m["head_punct"], 0.01) and m["tail_punct"] >= 0.2:
        f.append(f"tail_degrade {m['head_punct']}->{m['tail_punct']}")
    return f


def main(paths):
    print(f"{'file':22}{'verdict':8}{'compl':>7}{'chars':>8}{'lword':>7}{'run':>5}"
          f"{'gpunct':>8}{'wpunct':>8}{'loop8':>7}{'wTTR':>6}  FLAGS")
    for p in paths:
        j = json.load(open(p))
        text = (j.get("reasoning") or "") + "\n" + (j.get("content") or "")
        m = analyze(text.strip())
        u = j.get("usage") or {}
        fl = flags(m)
        name = p.split("/")[-2] if "/" in p else p
        if m.get("empty"):
            print(f"{name:22}{'-':8}{u.get('completion_tokens',0):>7}{0:>8}  EMPTY")
            continue
        print(f"{name:22}{('CORRUPT' if j.get('corrupt') else 'ok'):8}"
              f"{u.get('completion_tokens',0):>7}{m['chars']:>8}{m['longest_word']:>7}"
              f"{m['char_run']:>5}{m['global_punct']:>8}{m['worst_window_punct']:>8}"
              f"{m['repeat_8gram']:>7}{m['worst_window_ttr']:>6}  {' '.join(fl) if fl else '-'}")


if __name__ == "__main__":
    main(sys.argv[1:])
