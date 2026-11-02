"""LLM token accounting, budgets and limits.

Ollama Cloud is billed by plan (usage limits per 5-hour session and per
week), not per token, so tokens are what we count. Money is an estimate:
tokens x the per-token price the same open-weight models cost on
pay-per-token hosts (LLM_PRICES), which is also the bill if we moved to one.

Every LLM call (llm.py) goes through here:
- a per-replica concurrency cap (LLM_MAX_CONCURRENCY), so a burst of claims
  doesn't fire dozens of calls at once and trip the provider's rate limit;
- a per-run token budget (LLM_RUN_TOKEN_BUDGET): a prompt-injection loop or
  a model that rambles can't burn tokens without end;
- a daily budget across all replicas (LLM_DAILY_TOKEN_BUDGET, in Redis): the
  cost circuit breaker. Off (0) by default.

A refused call raises LLMError, the same as an LLM outage, so the node falls
back to its code path and the claim still reaches the adjuster (the
graceful-degradation path every node already has). Budgets are checked
before a call, so a run can go over by the calls already in flight (intake
and photo_review run in parallel); output is capped (LLM_MAX_OUTPUT_TOKENS in
llm.py), which bounds each one. Seen on kind with a 1,500-token budget: the
first three calls ran, the fourth (Opinion) was refused and fell back.

What each claim used is kept in Redis (llm:usage:<claim>, summed over all its
runs) and returned by GET /claims/{id} for adjusters; Prometheus gets the same
counts per model and call (prometheus-client is optional: services/eval
imports llm.py without it).
"""

import asyncio
import contextvars
import json
import logging
import os
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime

log = logging.getLogger("agent.usage")

RUN_TOKEN_BUDGET = int(os.getenv("LLM_RUN_TOKEN_BUDGET", "20000"))
DAILY_TOKEN_BUDGET = int(os.getenv("LLM_DAILY_TOKEN_BUDGET", "0"))  # 0 = no cap
MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "4"))
# USD per million tokens, [input, output]. Reference rates for the same
# open-weight models on pay-per-token hosts, checked 2026-11-02 on price
# trackers (gpt-oss-120b $0.04-0.18 in / $0.17-0.80 out; Gemma 4 31B $0.12-0.14
# in / $0.35-0.40 out). The defaults are the upper end, for budgeting.
DEFAULT_PRICES = {"gpt-oss:120b": [0.15, 0.60], "gemma4:31b": [0.14, 0.40]}
PRICES = json.loads(os.getenv("LLM_PRICES", "") or json.dumps(DEFAULT_PRICES))
CLAIM_TTL_S = 90 * 24 * 3600
DAY_TTL_S = 3 * 24 * 3600

try:
    from prometheus_client import Counter, Histogram

    TOKENS = Counter(
        "llm_tokens_total", "LLM tokens used", ["model", "call", "direction"]
    )
    CALLS = Counter(
        "llm_calls_total",
        "LLM calls by outcome (ok, error, rate_limited, budget)",
        ["model", "call", "outcome"],
    )
    LATENCY = Histogram(
        "llm_call_duration_seconds",
        "Time an LLM call took, retries included",
        ["model", "call"],
        buckets=(0.5, 1, 2, 3, 5, 8, 13, 20, 30, 60, 120),
    )
    COST = Counter(
        "llm_estimated_cost_usd_total", "Estimated LLM cost at LLM_PRICES", ["model"]
    )
except ImportError:  # services/eval: no metrics
    TOKENS = CALLS = LATENCY = COST = None


def cost(model, input_tokens, output_tokens):
    """Estimated USD at LLM_PRICES; None for a model without a price."""
    price = PRICES.get(model)
    if not price:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1e6


def observe(model, call, outcome, seconds, input_tokens=0, output_tokens=0):
    """Prometheus, for every call (also outside a claim run)."""
    if CALLS is None:
        return
    CALLS.labels(model, call, outcome).inc()
    LATENCY.labels(model, call).observe(seconds)
    if input_tokens or output_tokens:
        TOKENS.labels(model, call, "input").inc(input_tokens)
        TOKENS.labels(model, call, "output").inc(output_tokens)
        usd = cost(model, input_tokens, output_tokens)
        if usd:
            COST.labels(model).inc(usd)


_slots = weakref.WeakKeyDictionary()  # event loop -> Semaphore


def slot():
    """The concurrency cap: `async with usage.slot():` around a call. One
    semaphore per event loop (tests run several loops)."""
    loop = asyncio.get_running_loop()
    if loop not in _slots:
        _slots[loop] = asyncio.Semaphore(MAX_CONCURRENCY)
    return _slots[loop]


def _usage_key(claim_id):
    return f"llm:usage:{claim_id}"


def _day_key(day=None):
    return f"llm:tokens:{day or datetime.now(UTC).date().isoformat()}"


class Ledger:
    """Token counts in Redis, shared by every replica: per claim (all its runs
    together) and per UTC day (for the daily budget)."""

    def __init__(self, redis):
        self.r = redis  # redis.asyncio client with decode_responses=True

    async def record(self, claim_id, model, input_tokens, output_tokens):
        key = _usage_key(claim_id)
        pipe = self.r.pipeline()
        for prefix in ("", f"{model}|"):
            pipe.hincrby(key, prefix + "calls", 1)
            pipe.hincrby(key, prefix + "input", input_tokens)
            pipe.hincrby(key, prefix + "output", output_tokens)
        pipe.expire(key, CLAIM_TTL_S)
        pipe.incrby(_day_key(), input_tokens + output_tokens)
        pipe.expire(_day_key(), DAY_TTL_S)
        await pipe.execute()

    async def today(self):
        return int(await self.r.get(_day_key()) or 0)

    async def claim(self, claim_id):
        """What the claim's LLM calls used so far, or None if it made none."""
        return summarize(await self.r.hgetall(_usage_key(claim_id)))


def summarize(raw):
    """A claim's usage hash ({"calls", "input", "output", "<model>|input", ...})
    as the API shows it, with estimated costs."""
    if not raw:
        return None
    by_model = {}
    for name, value in raw.items():
        if "|" in name:
            model, what = name.rsplit("|", 1)
            by_model.setdefault(model, {})[what] = int(value)
    models = {
        model: {
            "calls": v.get("calls", 0),
            "input_tokens": v.get("input", 0),
            "output_tokens": v.get("output", 0),
            "est_cost_usd": cost(model, v.get("input", 0), v.get("output", 0)),
        }
        for model, v in by_model.items()
    }
    costs = [m["est_cost_usd"] for m in models.values()]
    return {
        "calls": int(raw.get("calls", 0)),
        "input_tokens": int(raw.get("input", 0)),
        "output_tokens": int(raw.get("output", 0)),
        # None if any model has no price: a partial sum would read as the total.
        "est_cost_usd": round(sum(costs), 8) if costs and None not in costs else None,
        "by_model": models,
    }


@dataclass
class Meter:
    """One claim run's LLM usage (start, retake, approval or retry)."""

    claim_id: str
    ledger: Ledger | None = None
    run_budget: int = RUN_TOKEN_BUDGET
    daily_budget: int = DAILY_TOKEN_BUDGET
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    refused: list = field(default_factory=list)

    @property
    def tokens(self):
        return self.input_tokens + self.output_tokens

    async def refusal(self):
        """Why the next call may not run, or None."""
        if self.run_budget and self.tokens >= self.run_budget:
            return f"run token budget spent ({self.tokens}/{self.run_budget})"
        if self.daily_budget and self.ledger is not None:
            try:
                used = await self.ledger.today()
            except Exception:  # Redis blip: don't block claims on the counter
                log.warning("daily usage not read", exc_info=True)
                used = 0
            if used >= self.daily_budget:
                return f"daily token budget spent ({used}/{self.daily_budget})"
        return None

    async def add(self, model, input_tokens, output_tokens):
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if self.ledger is not None:
            try:
                await self.ledger.record(
                    self.claim_id, model, input_tokens, output_tokens
                )
            except Exception:  # accounting must never fail a claim
                log.exception("claim %s: usage not recorded", self.claim_id)


_current = contextvars.ContextVar("llm_meter", default=None)


def current():
    """The running claim's Meter, or None outside a run (eval, tests)."""
    return _current.get()


@contextmanager
def metering(claim_id, ledger=None, **budgets):
    """Meter the LLM calls made inside the block. LangGraph runs nodes in
    tasks that copy this context, so they all add to the same Meter."""
    meter = Meter(claim_id, ledger, **budgets)
    token = _current.set(meter)
    try:
        yield meter
    finally:
        _current.reset(token)
