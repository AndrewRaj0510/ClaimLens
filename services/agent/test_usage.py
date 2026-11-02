"""LLM metering (usage.py, llm.py): budgets, retries, the concurrency cap,
per-claim usage and metrics. The Ledger itself runs against a real Redis
when TEST_REDIS_URL is set (as in test_runstate.py)."""

import asyncio
import os
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import fakes
import llm
import usage
from graph import build_graph
from test_tracing import FakeOllama, ollama_llm

pytestmark = pytest.mark.anyio

CLAIM = "e" * 32
KEY, JOB = f"claims/{CLAIM}/0", "6" * 32
TEST_REDIS = os.getenv("TEST_REDIS_URL")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(llm, "RETRY_BACKOFF_S", (0.0, 0.0))
    monkeypatch.setattr(llm.random, "random", lambda: 0.0)


def claim_state():
    return {
        "claim_id": CLAIM,
        "customer_id": "C00001",
        "policy_id": "P00001",
        "incident_date": "2026-09-01",
        "reported_date": "2026-09-03",
        "description": "I was rear-ended at a red light on Main Street.",
        "claimed_amount": None,
        "photos": [{"key": KEY, "job_id": JOB}],
        "retakes": 0,
        "trace": [],
    }


def graph_with(gpt, gemma):
    d = fakes.deps(
        llm=gpt,
        vlm=gemma,
        photos={KEY: fakes.photo_bytes(0)},
        vision={JOB: fakes.vision_result(("dent", 2.0))},
    )
    return build_graph(d, InMemorySaver()), {"configurable": {"thread_id": CLAIM}}


async def test_a_claim_run_is_metered_per_model():
    ledger = fakes.FakeLedger()
    graph, config = graph_with(ollama_llm("gpt-oss:120b"), ollama_llm("gemma4:31b"))
    with usage.metering(CLAIM, ledger) as meter:
        await graph.ainvoke(claim_state(), config)
    # Intake, PhotoReview, the clause-search chat, Opinion: 120 in / 30 out each.
    assert meter.calls == 4
    assert (meter.input_tokens, meter.output_tokens) == (480, 120)
    decision = {"action": "approve", "amount": None, "adjuster": "sam", "notes": ""}
    with usage.metering(CLAIM, ledger) as second:
        await graph.ainvoke(Command(resume=decision), config)
    assert second.calls == 1  # the report

    used = await ledger.claim(CLAIM)
    assert used["calls"] == 5
    assert used["by_model"]["gemma4:31b"]["calls"] == 1
    assert used["by_model"]["gpt-oss:120b"] == {
        "calls": 4,
        "input_tokens": 480,
        "output_tokens": 120,
        "est_cost_usd": usage.cost("gpt-oss:120b", 480, 120),
    }
    assert used["est_cost_usd"] == pytest.approx(
        usage.cost("gpt-oss:120b", 480, 120) + usage.cost("gemma4:31b", 120, 30)
    )


async def test_over_the_run_budget_the_llm_steps_fall_back():
    gpt, gemma = ollama_llm("gpt-oss:120b"), ollama_llm("gemma4:31b")
    graph, config = graph_with(gpt, gemma)
    # 150 tokens a call: the intake call runs, then the budget is spent.
    with usage.metering(CLAIM, run_budget=150) as meter:
        await graph.ainvoke(claim_state(), config)
    assert meter.calls == 1
    snap = await graph.aget_state(config)
    sources = {s["node"]: s.get("source") for s in snap.values["trace"]}
    assert sources["intake"] == "llm"
    assert sources["photo_review"] == "fallback"
    assert snap.values["opinion"]["source"] == "fallback"
    # The claim still reaches the adjuster: a budget never decides a claim.
    assert snap.next == ("human_approval",)
    assert len(gpt.client.calls) + len(gemma.client.calls) == 1


async def test_the_daily_budget_is_shared_through_the_ledger():
    ledger = fakes.FakeLedger(today=1_000)
    gpt = ollama_llm("gpt-oss:120b")
    with (
        usage.metering(CLAIM, ledger, daily_budget=1_000),
        pytest.raises(llm.LLMError, match="daily token budget"),
    ):
        await gpt.chat([{"role": "user", "content": "hi"}], tools=[])
    assert gpt.client.calls == []
    with usage.metering(CLAIM, ledger, daily_budget=0):  # 0 = off
        await gpt.chat([{"role": "user", "content": "hi"}], tools=[])
    assert len(gpt.client.calls) == 1


async def test_outside_a_run_calls_are_not_budgeted():
    """services/eval calls the same client with no claim around it."""
    assert usage.current() is None
    await ollama_llm("gpt-oss:120b").chat([], tools=[])


class Flaky(FakeOllama):
    def __init__(self, *statuses):
        super().__init__()
        self.statuses = list(statuses)
        self.options = []

    async def chat(self, model, messages, think, options, **kwargs):
        self.options.append(options)
        if self.statuses:
            from ollama import ResponseError

            raise ResponseError("nope", self.statuses.pop(0))
        return await super().chat(model, messages, think, options, **kwargs)


@pytest.mark.parametrize(
    ("statuses", "calls", "ok"),
    [
        ((429,), 2, True),  # rate limited, then fine
        ((503, 502), 3, True),
        ((429, 429, 429), 3, False),  # gives up after two retries
        ((401,), 1, False),  # a bad key isn't retried
        ((400,), 1, False),
    ],
)
async def test_rate_limits_and_server_errors_are_retried(statuses, calls, ok):
    gpt = ollama_llm("gpt-oss:120b")
    gpt.client = Flaky(*statuses)
    with usage.metering(CLAIM) as meter:
        if ok:
            await gpt.chat([], tools=[])
        else:
            with pytest.raises(llm.LLMError):
                await gpt.chat([], tools=[])
    assert len(gpt.client.options) == calls
    assert meter.calls == (1 if ok else 0)  # failed calls cost nothing


async def test_output_is_capped():
    gpt = ollama_llm("gpt-oss:120b")
    gpt.client = Flaky()
    await gpt.chat([], tools=[])
    assert gpt.client.options[0] == {"temperature": 0, "num_predict": 4096}


async def test_concurrent_calls_are_capped(monkeypatch):
    monkeypatch.setattr(usage, "MAX_CONCURRENCY", 2)
    monkeypatch.setattr(usage, "_slots", usage.weakref.WeakKeyDictionary())
    active = peak = 0

    class Slow(FakeOllama):
        async def chat(self, *args, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return SimpleNamespace(
                message=SimpleNamespace(content="", tool_calls=None),
                prompt_eval_count=1,
                eval_count=1,
            )

    gpt = ollama_llm("gpt-oss:120b")
    gpt.client = Slow()
    await asyncio.gather(*(gpt.chat([], tools=[]) for _ in range(6)))
    assert peak == 2


async def test_a_broken_ledger_never_fails_a_call():
    class Broken(fakes.FakeLedger):
        async def record(self, *args):
            raise ConnectionError("redis down")

        async def today(self):
            raise ConnectionError("redis down")

    with usage.metering(CLAIM, Broken(), daily_budget=10) as meter:
        await ollama_llm("gpt-oss:120b").chat([], tools=[])
    assert meter.calls == 1


def test_cost_and_unpriced_models():
    assert usage.cost("gpt-oss:120b", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert usage.cost("unknown", 10, 10) is None
    raw = {"calls": "1", "input": "10", "output": "5", "unknown|calls": "1"}
    # A partial sum would look like the total, so there is none.
    assert usage.summarize(raw)["est_cost_usd"] is None
    assert usage.summarize({}) is None


async def test_metrics_count_tokens_and_outcomes():
    from prometheus_client import generate_latest

    gpt = ollama_llm("gpt-oss:120b")
    await gpt.chat([], tools=[])
    gpt.client = Flaky(401)
    with pytest.raises(llm.LLMError):
        await gpt.chat([], tools=[])
    text = generate_latest().decode()
    assert (
        'llm_tokens_total{call="chat",direction="input",model="gpt-oss:120b"}' in text
    )
    assert 'llm_calls_total{call="chat",model="gpt-oss:120b",outcome="error"}' in text
    assert 'llm_estimated_cost_usd_total{model="gpt-oss:120b"}' in text


@pytest.mark.skipif(not TEST_REDIS, reason="set TEST_REDIS_URL to run")
async def test_ledger_on_redis():
    import redis.asyncio as redis

    r = redis.Redis.from_url(TEST_REDIS, decode_responses=True)
    await r.flushdb()
    try:
        ledger = usage.Ledger(r)
        await ledger.record(CLAIM, "gpt-oss:120b", 2_000, 150)
        await ledger.record(CLAIM, "gemma4:31b", 1_500, 100)
        await ledger.record("f" * 32, "gpt-oss:120b", 10, 10)
        used = await ledger.claim(CLAIM)
        assert (used["calls"], used["input_tokens"], used["output_tokens"]) == (
            2,
            3_500,
            250,
        )
        assert used["by_model"]["gemma4:31b"]["input_tokens"] == 1_500
        assert await ledger.today() == 3_770
        assert 0 < await r.ttl(f"llm:usage:{CLAIM}") <= usage.CLAIM_TTL_S
        assert await ledger.claim("0" * 32) is None
    finally:
        await r.flushdb()
        await r.aclose()
