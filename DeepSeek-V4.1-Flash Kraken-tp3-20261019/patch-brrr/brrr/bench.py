import asyncio
import itertools
import operator
import secrets
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import Annotated, Self

import httpx
import numpy as np
from cyclopts import App, Parameter, validators
from pydantic import BaseModel

MAX_TOKENS = 2048

CODE_TASK = """I have a FastAPI service that receives webhook events from a payment provider and stores them in Postgres with asyncpg. The current handler:

```python
@app.post("/webhooks/payments")
async def payments_webhook(request: Request) -> dict[str, str]:
    payload = await request.json()
    if payload["type"] == "invoice.paid":
        await pool.execute(
            "INSERT INTO invoices (id, amount) VALUES ($1, $2)",
            payload["data"]["id"], payload["data"]["amount"],
        )
    return {"status": "ok"}
```

Problems: the `X-Signature` header is never verified, retried deliveries insert the same invoice twice, and any exception returns a 500 so the provider retries forever.

Rewrite this, returning a single python file: verify the signature with HMAC-SHA256 over the raw body and reject stale timestamps, make inserts idempotent on the event id, dispatch event types through a registry instead of if-chains, and return 2xx for event types we don't handle. Use pydantic models, full type hints, and no bare except. Include the SQL for the idempotency table and a pytest test for the signature check."""
CODE_SUFFIX = (
    "Sorry, the task above is repeated. It is a single "
    "task, so answer it exactly once. Output only well-typed code (no comments or header/footer text) with "
    "no prose before, between, or after them. Your full output will be directly inserted into a code file, so your full response MUST be code only. No prose/comments whatsoever; pure, single python file as output which I will paste in."
)
COLUMNS = (
    "prompt",
    "conc",
    "TTFT p50",
    *(
        f"{phase} {stat}"
        for phase in ("PP", "TG")
        for stat in ("p1", "p50", "p99", "total")
    ),
    "TG window",
    "accept",
)

app = App(
    help="Reproduce the DeepSeek-4.1-Flash-1017-BRRR benchmark against a running server."
)


class PromptDetails(BaseModel, frozen=True):
    cached_tokens: int


class Usage(BaseModel, frozen=True):
    prompt_tokens: int
    completion_tokens: int = 0
    prompt_tokens_details: PromptDetails | None = None


class Chunk(BaseModel, frozen=True):
    usage: Usage | None = None


class Tokenized(BaseModel, frozen=True):
    count: int


class ServedModel(BaseModel, frozen=True):
    id: str


class ServedModels(BaseModel, frozen=True):
    data: tuple[ServedModel, ...]


class Arrival(BaseModel, frozen=True):
    usage: Usage
    at: float

    @property
    def tokens(self) -> int:
        return self.usage.completion_tokens


class Sample(BaseModel, frozen=True):
    prompt_tokens: int
    cached_tokens: int
    started: float
    arrivals: tuple[Arrival, ...]

    @property
    def ttft(self) -> float:
        return self.arrivals[0].at - self.started

    def tokens_at(self, at: float) -> int:
        return max((a.tokens for a in self.arrivals if a.at <= at), default=0)


class Window(BaseModel, frozen=True):
    tokens: tuple[int, ...]
    seconds: float


class Spec(BaseModel, frozen=True):
    drafts: float = 0
    draft_tokens: float = 0
    accepted: float = 0

    def combine(self, other: Self, op: Callable[[float, float], float]) -> Self:
        return type(self)(
            drafts=op(self.drafts, other.drafts),
            draft_tokens=op(self.draft_tokens, other.draft_tokens),
            accepted=op(self.accepted, other.accepted),
        )

    def __add__(self, other: Self) -> Self:
        return self.combine(other, operator.add)

    def __sub__(self, other: Self) -> Self:
        return self.combine(other, operator.sub)


class Round(BaseModel, frozen=True):
    prefill: tuple[Sample, ...]
    window: Window
    spec: Spec


def coding_prompt(nonce: str, repeats: int) -> str:
    return f"{nonce} " + "\n\n".join([CODE_TASK] * repeats) + f"\n\n{CODE_SUFFIX}"


async def calibrated_prompt(http: httpx.AsyncClient, model: str, size: int) -> str:
    nonce = secrets.token_hex(8)
    guess = max(1, round(size * 3.6 / (len(CODE_TASK) + 2)))
    response = await http.post(
        "/tokenize", json={"model": model, "prompt": coding_prompt(nonce, guess)}
    )
    measured = Tokenized.model_validate_json(response.raise_for_status().content).count
    return coding_prompt(nonce, max(1, round(guess * size / measured)))


async def stream(
    http: httpx.AsyncClient,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    thinking: int,
    max_tokens: int | None,
) -> Sample:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "chat_template_kwargs": {"reasoning_effort": thinking}
        if thinking
        else {"thinking": False, "enable_thinking": False},
        **({"max_tokens": max_tokens} if max_tokens else {}),
    }
    start = time.perf_counter()
    async with http.stream("POST", "/v1/chat/completions", json=body) as response:
        response.raise_for_status()
        reports = [
            Arrival(usage=usage, at=time.perf_counter())
            async for line in response.aiter_lines()
            if line.startswith("data: ") and line != "data: [DONE]"
            if (usage := Chunk.model_validate_json(line.removeprefix("data: ")).usage)
        ]
    peaks = itertools.accumulate((r.tokens for r in reports), max, initial=0)
    arrivals = tuple(r for r, peak in zip(reports, peaks) if r.tokens > peak)
    if not arrivals:
        raise RuntimeError("no tokens generated before the request ended")
    final = reports[-1].usage
    if final.prompt_tokens_details is None:
        raise RuntimeError("server reported no prompt_tokens_details")
    return Sample(
        prompt_tokens=final.prompt_tokens,
        cached_tokens=final.prompt_tokens_details.cached_tokens,
        started=start,
        arrivals=arrivals,
    )


async def spec_counters(http: httpx.AsyncClient) -> Spec:
    named = [
        (
            line.partition("{")[0].partition(" ")[0].removesuffix("_total"),
            float(line.rsplit(" ", 1)[1]),
        )
        for line in (await http.get("/metrics")).text.splitlines()
        if not line.startswith("#")
    ]

    def total(counter: str) -> float:
        return sum(v for name, v in named if name == f"vllm:spec_decode_{counter}")

    return Spec(
        drafts=total("num_drafts"),
        draft_tokens=total("num_draft_tokens"),
        accepted=total("num_accepted_tokens"),
    )


async def cool(limit: int) -> None:
    query = ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"]
    while max(map(int, subprocess.check_output(query, text=True).split())) > limit:
        await asyncio.sleep(2)


def aligned(cells: Iterable[str]) -> str:
    return "".join(f"{cell:>11}" for cell in cells)


def rates(values: Sequence[float], total: float) -> tuple[str, ...]:
    return *(f"{np.percentile(values, q):,.0f}" for q in (99, 50, 1)), f"{total:,.0f}"


def decode_window(batch: Sequence[Sample]) -> Window:
    start = max(s.arrivals[0].at for s in batch)
    end = min(s.arrivals[-1].at for s in batch)
    gaps = np.concatenate(
        [np.diff([a.at for a in s.arrivals if start <= a.at <= end]) for s in batch]
    )
    if end <= start or not gaps.size:
        raise RuntimeError("a stream finished before every stream started decoding")
    edge = float(np.median(gaps)) / 2
    return Window(
        tokens=tuple(
            s.tokens_at(end + edge) - s.tokens_at(start + edge) for s in batch
        ),
        seconds=end - start,
    )


def row(concurrency: int, rounds: Sequence[Round]) -> tuple[str, ...]:
    samples = [s for r in rounds for s in r.prefill]
    tokens = [n for r in rounds for n in r.window.tokens]
    spec = sum((r.spec for r in rounds), Spec())
    return (
        f"{round(np.mean([s.prompt_tokens for s in samples])):,}",
        str(concurrency),
        f"{np.percentile([s.ttft for s in samples], 50) * 1000:,.0f} ms",
        *rates(
            [s.prompt_tokens / s.ttft for s in samples],
            sum(s.prompt_tokens for s in samples)
            / sum(max(s.ttft for s in r.prefill) for r in rounds),
        ),
        *rates(
            [n / r.window.seconds for r in rounds for n in r.window.tokens],
            sum(tokens) / sum(r.window.seconds for r in rounds),
        ),
        f"{np.median(tokens):,.0f}",
        f"{1 + spec.accepted / spec.drafts:.2f} ({spec.accepted / spec.draft_tokens:.0%})"
        if spec.drafts
        else "—",
    )


@app.default
async def bench(
    *,
    sizes: Annotated[
        tuple[int, ...],
        Parameter(
            name=("--sizes", "-s"),
            help="Prompt sizes in tokens.",
            negative="",
            consume_multiple=True,
        ),
    ] = (2048, 16384, 65528),
    concurrency: Annotated[
        tuple[int, ...],
        Parameter(
            name=("--concurrency", "-c"),
            help="Parallel request counts.",
            negative="",
            consume_multiple=True,
        ),
    ] = (1, 4, 8, 16),
    repeats: Annotated[
        int, Parameter(name=("--repeats", "-r"), help="Measured rounds per cell.")
    ] = 5,
    cool_to: Annotated[
        int,
        Parameter(
            help="Before every round, wait until the hottest GPU is at or below this °C."
        ),
    ] = 70,
    temperature: Annotated[
        float, Parameter(name=("--temperature", "-t"), help="Sampling temperature.")
    ] = 1.0,
    top_p: Annotated[float, Parameter(help="Top-p.")] = 1.0,
    thinking: Annotated[
        int,
        Parameter(
            name=("--thinking", "-k"),
            help="Thinking level 1-100 (DeepSeek V4.1: low = 50, high = 75, max = 100); 0 turns thinking off. With thinking on, requests run uncapped to natural EOS.",
            validator=validators.Number(gte=0, lte=100),
        ),
    ] = 0,
    port: Annotated[
        int, Parameter(name=("--port", "-p"), help="Port of the running server.")
    ] = 8000,
) -> None:
    async with httpx.AsyncClient(
        base_url=f"http://localhost:{port}", timeout=None
    ) as http:
        models = (await http.get("/v1/models")).content
        model = ServedModels.model_validate_json(models).data[0].id
        send = partial(
            stream, http, model, temperature=temperature, top_p=top_p, thinking=thinking
        )

        async def measure(size: int, c: int) -> Round:
            await cool(cool_to)
            prompts = await asyncio.gather(
                *(calibrated_prompt(http, model, size) for _ in range(c))
            )
            prefill = await asyncio.gather(
                *(send(prompt=p, max_tokens=1) for p in prompts)
            )
            if any(s.cached_tokens for s in prefill):
                raise RuntimeError("prefill pass hit the prefix cache")
            before = await spec_counters(http)
            decode = await asyncio.gather(
                *(
                    send(prompt=p, max_tokens=None if thinking else MAX_TOKENS)
                    for p in prompts
                )
            )
            spec = await spec_counters(http) - before
            return Round(
                prefill=tuple(prefill), window=decode_window(decode), spec=spec
            )

        budget = (
            f"thinking {thinking} · uncapped to natural EOS"
            if thinking
            else f"thinking off · gen ≤{MAX_TOKENS} tok to natural EOS"
        )
        print(
            f"{model} · code prompt · {budget} · repeats={repeats} · cool to {cool_to}°C"
        )
        print(
            "tok/s: p1/p50/p99 = fastest/median/slowest stream · PP from a max_tokens=1 pass · TG from a cached second pass, timed while all streams decode and none has finished"
        )
        print(aligned(COLUMNS))
        for size, c in itertools.product(sizes, concurrency):
            print(aligned(row(c, [await measure(size, c) for _ in range(repeats)])))
