import time
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

import fakes
from graph import build_graph
from main import create_app

CLAIM = "b" * 32
KEY, JOB = f"claims/{CLAIM}/0", "3" * 32


@pytest.fixture
def client():
    d = fakes.deps(
        photos={KEY: fakes.photo_bytes(0), f"claims/{CLAIM}/1": fakes.photo_bytes(1)},
        vision={
            JOB: fakes.vision_result(("dent", 2.0)),
            "4" * 32: fakes.vision_result(("dent", 2.0)),
        },
    )

    @asynccontextmanager
    async def lifespan(app):
        app.state.graph = build_graph(d, InMemorySaver())
        yield

    with TestClient(create_app(lifespan=lifespan)) as c:
        c.deps = d
        yield c


def new_claim(**overrides):
    return {
        "claim_id": CLAIM,
        "policy_id": "P00001",
        "incident_date": "2026-09-01",
        "description": "Rear-ended at a red light on Main Street.",
        "photos": [{"key": KEY, "job_id": JOB}],
        **overrides,
    }


def wait_for(client, status, claim_id=CLAIM, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/claims/{claim_id}").json()
        if body["status"] == status:
            return body
        time.sleep(0.05)
    raise AssertionError(
        f"claim never reached {status}: {body['status']} {body.get('error')}"
    )


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "service": "agent"}


def test_claim_lifecycle(client):
    r = client.post("/claims", json=new_claim())
    assert r.status_code == 202
    body = wait_for(client, "awaiting_approval")
    assert body["question"]["recommendation"]["action"] == "approve"
    assert body["state"]["payable"]["expected"] == 50.0
    assert [t["node"] for t in body["trace"]][-1] == "fraud_checks"

    r = client.post(
        f"/claims/{CLAIM}/decision",
        json={"action": "edit", "amount": 40, "adjuster": "sam"},
    )
    assert r.status_code == 202
    body = wait_for(client, "completed")
    assert body["state"]["decision"]["amount"] == 40
    assert body["question"] is None


def test_decision_only_when_awaiting_approval(client):
    client.post("/claims", json=new_claim())
    wait_for(client, "awaiting_approval")
    client.post(
        f"/claims/{CLAIM}/decision", json={"action": "reject", "adjuster": "sam"}
    )
    wait_for(client, "completed")
    r = client.post(
        f"/claims/{CLAIM}/decision", json={"action": "approve", "adjuster": "eve"}
    )
    assert r.status_code == 409  # decided once; can't be re-decided


def test_retake_flow(client):
    client.deps.photos.objects[KEY] = fakes.photo_bytes(0, blur=6)
    client.post("/claims", json=new_claim())
    body = wait_for(client, "awaiting_retake")
    assert body["question"]["photos"][0]["photo_index"] == 0
    # Can't decide a claim that's waiting for photos.
    assert (
        client.post(
            f"/claims/{CLAIM}/decision", json={"action": "approve", "adjuster": "x"}
        ).status_code
        == 409
    )

    r = client.post(
        f"/claims/{CLAIM}/photos",
        json={"photos": [{"key": f"claims/{CLAIM}/1", "job_id": "4" * 32}]},
    )
    assert r.status_code == 202
    wait_for(client, "awaiting_approval")


def test_photo_keys_must_belong_to_the_claim(client):
    other = f"claims/{'c' * 32}/0"
    r = client.post("/claims", json=new_claim(photos=[{"key": other, "job_id": JOB}]))
    assert r.status_code == 422
    r = client.post(
        "/claims", json=new_claim(photos=[{"key": "../../etc/passwd", "job_id": JOB}])
    )
    assert r.status_code == 422


def test_duplicate_claim_id_is_refused(client):
    client.post("/claims", json=new_claim())
    wait_for(client, "awaiting_approval")
    assert client.post("/claims", json=new_claim()).status_code == 409


@pytest.mark.parametrize(
    "bad",
    [
        {"policy_id": "x"},
        {"description": "short"},
        {"photos": []},
        {"claimed_amount": -5},
        {"incident_date": "yesterday"},
        {"incident_date": "2099-01-01"},  # in the future
    ],
)
def test_invalid_claims_are_rejected(client, bad):
    assert client.post("/claims", json=new_claim(**bad)).status_code == 422


def test_invalid_decisions_are_rejected_before_resuming(client):
    client.post("/claims", json=new_claim())
    wait_for(client, "awaiting_approval")
    for bad in (
        {"action": "pay_double", "adjuster": "x"},
        {"action": "edit", "adjuster": "x"},  # edit needs an amount
        {"action": "approve", "adjuster": ""},
        {"action": "approve", "adjuster": "x", "amount": -1},
    ):
        assert client.post(f"/claims/{CLAIM}/decision", json=bad).status_code == 422
    assert client.get(f"/claims/{CLAIM}").json()["status"] == "awaiting_approval"


def test_unknown_claim_is_404(client):
    assert client.get(f"/claims/{'d' * 32}").status_code == 404
    assert client.get("/claims/not-an-id").status_code == 422


def test_failed_run_is_reported_and_can_be_retried(client):
    tools = client.deps.tools

    async def broken(node, server, tool, args):
        raise RuntimeError("pricing server down")

    original = tools.call
    tools.call = broken
    client.post("/claims", json=new_claim())
    body = wait_for(client, "failed")
    assert "pricing server down" in body["error"]

    tools.call = original
    assert client.post(f"/claims/{CLAIM}/retry").status_code == 202
    wait_for(client, "awaiting_approval")


class Gate:
    """Holds a coroutine until released (from the test thread)."""

    def __init__(self):
        self.open = False

    async def wait(self):
        import asyncio

        while not self.open:
            await asyncio.sleep(0.01)


def test_claim_is_visible_before_its_first_checkpoint(client):
    gate, graph = Gate(), client.app.state.graph
    real = graph.ainvoke

    async def delayed(*args, **kwargs):
        await gate.wait()
        return await real(*args, **kwargs)

    graph.ainvoke = delayed
    client.post("/claims", json=new_claim())
    r = client.get(f"/claims/{CLAIM}")  # no checkpoint exists yet
    assert r.status_code == 200
    assert r.json()["status"] == "processing"
    gate.open = True
    wait_for(client, "awaiting_approval")


def test_run_cut_off_by_a_restart_shows_interrupted_and_can_be_retried(client):
    gate = Gate()
    vision = client.deps.vision
    real = vision.result

    async def blocked(job_id, timeout_s):
        await gate.wait()
        return await real(job_id, timeout_s)

    vision.result = blocked
    client.post("/claims", json=new_claim())
    deadline = time.time() + 5
    while not client.get(f"/claims/{CLAIM}").json()["state"].get("quality"):
        assert time.time() < deadline
        time.sleep(0.02)  # wait until it's stuck in damage_analysis

    # "Restart": the in-memory run is lost; only the checkpoint survives.
    task = client.app.state.runs.pop(CLAIM)
    client.portal.call(task.cancel)
    body = wait_for(client, "interrupted")
    assert "POST /retry" in body["error"]

    gate.open = True
    assert client.post(f"/claims/{CLAIM}/retry").status_code == 202
    wait_for(client, "awaiting_approval")


def test_retry_refused_for_a_healthy_claim(client):
    client.post("/claims", json=new_claim())
    wait_for(client, "awaiting_approval")
    assert client.post(f"/claims/{CLAIM}/retry").status_code == 409
