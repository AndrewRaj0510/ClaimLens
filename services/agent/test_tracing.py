"""Langfuse tracing: off by default, and when on, one trace per claim with a
span per run, per node and per LLM call. Spans go to an in-memory OpenTelemetry
exporter instead of a server (the Langfuse SDK is built on OpenTelemetry)."""

import json
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import fakes
import tracing
from graph import build_graph
from llm import OllamaLLM

pytestmark = pytest.mark.anyio

CLAIM = "c" * 32
KEY, JOB = f"claims/{CLAIM}/0", "5" * 32

REPLIES = {
    "Intake": {
        "peril": "collision",
        "summary": "Rear-ended at a light.",
        "mentioned_date": None,
        "other_vehicle_involved": True,
        "injuries_mentioned": False,
    },
    "Opinion": {
        "assessment": "covered",
        "reasoning": "Part D covers it.",
        "citations": ["C1"],
        "concerns": [],
    },
    "Report": {"customer_message": "Approved.", "adjuster_summary": "Approved."},
    "PhotoReview": {
        "matches_description": "yes",
        "damage_seen": ["dent"],
        "concerns": [],
        "summary": "A dent.",
    },
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeOllama:
    """Stands in for ollama.AsyncClient: answers by the schema it was given."""

    def __init__(self):
        self.calls = []

    async def chat(self, model, messages, think, options, format=None, tools=None):
        self.calls.append(messages)
        content = (
            json.dumps(REPLIES[format["title"]]) if format else "No search needed."
        )
        return SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=None),
            prompt_eval_count=120,
            eval_count=30,
        )


def ollama_llm(model):
    llm = OllamaLLM(host="http://unused", model=model, api_key="", think="")
    llm.client = FakeOllama()
    return llm


@pytest.fixture
def spans(monkeypatch):
    from langfuse import Langfuse
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    lf = Langfuse(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        base_url="http://langfuse.invalid",
        span_exporter=exporter,
    )
    monkeypatch.setattr(tracing, "ENABLED", True)
    monkeypatch.setattr(tracing, "_client", lf)

    def finished():
        lf.flush()
        return exporter.get_finished_spans()

    yield finished
    lf.shutdown()


def test_off_without_keys_and_yields_no_callbacks():
    assert tracing.client() is None or tracing.ENABLED
    with tracing.claim_run(CLAIM, "start"), tracing.node("intake") as span:
        span.update(output={"ignored": True})
    with tracing.generation("x", "m", [], {}) as gen:
        gen.update(output="ignored")


def test_photos_are_never_logged():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u", "images": [b"\xff\xd8jpeg", b"\xff\xd8"]},
    ]
    safe = tracing.without_images(messages)
    assert safe[1]["images"] == "[2 redacted photo(s), not logged]"
    assert messages[1]["images"][0] == b"\xff\xd8jpeg"  # the call itself is untouched


async def test_a_claim_is_one_trace_with_runs_nodes_and_generations(spans):
    d = fakes.deps(
        llm=ollama_llm("gpt-oss:120b"),
        vlm=ollama_llm("gemma4:31b"),
        photos={KEY: fakes.photo_bytes(0)},
        vision={JOB: fakes.vision_result(("dent", 2.0))},
    )
    graph = build_graph(d, InMemorySaver())
    config = {"configurable": {"thread_id": CLAIM}}
    state = {
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
    # Two runs, like production: up to the approval interrupt, then the decision.
    with tracing.claim_run(CLAIM, "start", "C00001"):
        await graph.ainvoke(state, config)
    decision = {"action": "approve", "amount": None, "adjuster": "sam", "notes": ""}
    with tracing.claim_run(CLAIM, "approval"):
        await graph.ainvoke(Command(resume=decision), config)

    finished = spans()
    by_name = {}
    for s in finished:
        by_name.setdefault(s.name, []).append(s)

    # One trace for the whole claim, across both runs.
    assert {s.context.trace_id for s in finished} == {
        int(tracing.client().create_trace_id(seed=CLAIM), 16)
    }
    assert set(by_name) >= {"claim start", "claim approval"}
    for node in ("intake", "photo_review", "policy_lookup", "final_report"):
        assert node in by_name, node

    # Each LLM call is a generation nested under the node that made it.
    ids = {s.context.span_id: s for s in finished}

    def ancestors(span):
        names = []
        while span.parent is not None and span.parent.span_id in ids:
            span = ids[span.parent.span_id]
            names.append(span.name)
        return names

    gens = [
        s
        for s in finished
        if s.attributes.get("langfuse.observation.type") == "generation"
    ]
    assert {g.name for g in gens} >= {"Intake", "PhotoReview", "Opinion", "Report"}
    where = {g.name: ancestors(g) for g in gens}
    assert "intake" in where["Intake"] and "claim start" in where["Intake"]
    assert "photo_review" in where["PhotoReview"]
    assert "final_report" in where["Report"] and "claim approval" in where["Report"]

    # The approval interrupt is a pause, not an error.
    first_approval = min(by_name["human_approval"], key=lambda s: s.start_time)
    assert first_approval.status.status_code.name != "ERROR"
    assert ancestors(first_approval) == ["claim start"]

    photo_gen = next(g for g in gens if g.name == "PhotoReview")
    logged = photo_gen.attributes["langfuse.observation.input"]
    assert "redacted photo(s), not logged" in logged
    assert "\\xff" not in logged and "/9j/" not in logged  # no JPEG bytes or base64
    usage = json.loads(photo_gen.attributes["langfuse.observation.usage_details"])
    assert usage == {"input": 120, "output": 30}
    assert photo_gen.attributes["langfuse.observation.model.name"] == "gemma4:31b"


class BrokenSpan:
    def update(self, **kwargs):
        raise RuntimeError("update failed")


class BrokenLangfuse:
    """Every call fails: opening spans or updating them."""

    def __init__(self, fail_open):
        self.fail_open = fail_open

    def create_trace_id(self, seed):
        return "0" * 32

    def start_as_current_observation(self, **kwargs):
        if self.fail_open:
            raise RuntimeError("exporter down")
        from contextlib import nullcontext

        return nullcontext(BrokenSpan())


@pytest.mark.parametrize("fail_open", [True, False])
async def test_a_broken_tracer_never_fails_a_claim(monkeypatch, fail_open):
    monkeypatch.setattr(tracing, "ENABLED", True)
    monkeypatch.setattr(tracing, "_client", BrokenLangfuse(fail_open))
    d = fakes.deps(
        llm=ollama_llm("gpt-oss:120b"),
        vlm=ollama_llm("gemma4:31b"),
        photos={KEY: fakes.photo_bytes(0)},
        vision={JOB: fakes.vision_result(("dent", 2.0))},
    )
    graph = build_graph(d, InMemorySaver())
    config = {"configurable": {"thread_id": CLAIM}}
    state = {
        "claim_id": CLAIM,
        "policy_id": "P00001",
        "incident_date": "2026-09-01",
        "reported_date": "2026-09-03",
        "description": "I was rear-ended at a red light on Main Street.",
        "claimed_amount": None,
        "photos": [{"key": KEY, "job_id": JOB}],
        "retakes": 0,
        "trace": [],
    }
    with tracing.claim_run(CLAIM, "start"):
        await graph.ainvoke(state, config)
    decision = {"action": "approve", "amount": None, "adjuster": "sam", "notes": ""}
    with tracing.claim_run(CLAIM, "approval"):
        await graph.ainvoke(Command(resume=decision), config)
    snap = await graph.aget_state(config)
    assert snap.values["report"]["customer_message"] == "Approved."
    assert {s["source"] for s in snap.values["trace"] if s.get("source")} == {"llm"}
