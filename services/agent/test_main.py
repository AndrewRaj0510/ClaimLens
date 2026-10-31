import time
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

import fakes
import runstate
from graph import build_graph
from main import create_app

CLAIM = "b" * 32
KEY, JOB = f"claims/{CLAIM}/0", "3" * 32


def make_deps():
    return fakes.deps(
        photos={KEY: fakes.photo_bytes(0), f"claims/{CLAIM}/1": fakes.photo_bytes(1)},
        vision={
            JOB: fakes.vision_result(("dent", 2.0)),
            "4" * 32: fakes.vision_result(("dent", 2.0)),
        },
    )


def replica(d, saver, index, rs, **kwargs):
    """An app wired like one pod. Pass the same saver/index/rs to two of
    them and they share Postgres and Redis, like two replicas."""

    @asynccontextmanager
    async def lifespan(app):
        app.state.graph = build_graph(d, saver)
        app.state.index = index
        app.state.runstate = rs
        yield

    return create_app(lifespan=lifespan, **kwargs)


@pytest.fixture
def client():
    d = make_deps()
    app = replica(d, InMemorySaver(), fakes.FakeIndex(), fakes.FakeRunState())
    with TestClient(app) as c:
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
    client.post("/claims", json=new_claim(customer_id="C00001"))
    r = client.get(f"/claims/{CLAIM}")  # no checkpoint exists yet
    assert r.status_code == 200
    assert r.json()["status"] == "processing"
    # The owner is known already: the gateway authorizes the customer on it.
    # (Found end to end: the gateway 404'd customers on their own new claim.)
    assert r.json()["customer_id"] == "C00001"
    assert r.json()["state"] == {}
    gate.open = True
    wait_for(client, "awaiting_approval")


def block_vision(client):
    """Holds the run in damage_analysis until the returned gate opens."""
    gate = Gate()
    vision = client.deps.vision
    real = vision.result

    async def blocked(job_id, timeout_s):
        await gate.wait()
        return await real(job_id, timeout_s)

    vision.result = blocked
    return gate


def wait_until_in_damage_analysis(client):
    deadline = time.time() + 5
    while not client.get(f"/claims/{CLAIM}").json()["state"].get("quality"):
        assert time.time() < deadline
        time.sleep(0.02)


def test_run_cut_off_by_a_dead_pod_shows_interrupted_and_can_be_retried(client):
    gate = block_vision(client)
    client.post("/claims", json=new_claim())
    wait_until_in_damage_analysis(client)

    # The pod is killed (no shutdown, so no release): only the checkpoint and
    # the lease survive, and the lease still says "processing"...
    rs = client.app.state.runstate

    async def no_release(claim_id, token):
        pass

    rs.release = no_release
    client.portal.call(client.app.state.runs[CLAIM].cancel)
    time.sleep(0.1)
    assert client.get(f"/claims/{CLAIM}").json()["status"] == "processing"
    # ...until its TTL runs out.
    rs.expire(CLAIM)
    del rs.release  # the next pod releases normally
    body = wait_for(client, "interrupted")
    assert "POST /retry" in body["error"]

    gate.open = True
    assert client.post(f"/claims/{CLAIM}/retry").status_code == 202
    wait_for(client, "awaiting_approval")


def test_retry_refused_for_a_healthy_claim(client):
    client.post("/claims", json=new_claim())
    wait_for(client, "awaiting_approval")
    assert client.post(f"/claims/{CLAIM}/retry").status_code == 409


def test_claims_are_listed_with_status_and_summary(client):
    client.post("/claims", json=new_claim(customer_id="C00001"))
    wait_for(client, "awaiting_approval")
    deadline = time.time() + 5
    while True:  # the index is updated just after the run stops
        rows = client.get("/claims").json()["claims"]
        if rows[0]["status"] == "awaiting_approval" or time.time() > deadline:
            break
        time.sleep(0.02)
    row = rows[0]
    assert (row["claim_id"], row["customer_id"]) == (CLAIM, "C00001")
    assert row["status"] == "awaiting_approval"
    assert row["summary"]["recommendation"] == "approve"
    assert row["summary"]["payable_expected"] == 50.0
    assert row["summary"]["damages"] == ["dent (moderate)"]

    assert client.get("/claims?customer_id=C00001").json()["claims"]
    assert client.get("/claims?customer_id=C00002").json()["claims"] == []
    assert client.get("/claims?status=completed").json()["claims"] == []
    assert client.get("/claims?status=bogus").status_code == 422


def test_a_stale_processing_row_is_answered_from_the_checkpoint(client):
    client.post("/claims", json=new_claim())
    wait_for(client, "awaiting_approval")
    # As if the pod died before the row was updated.
    client.app.state.index.rows[CLAIM]["status"] = "processing"
    rows = client.get("/claims").json()["claims"]
    assert rows[0]["status"] == "awaiting_approval"


def test_claim_on_another_customers_policy_is_not_found(client):
    # The fake policy P00001 belongs to C00001.
    client.post("/claims", json=new_claim(customer_id="C00002"))
    body = wait_for(client, "awaiting_approval")
    assert body["state"]["policy"] is None
    assert body["state"]["coverage"]["reasons"] == ["Policy not found."]
    assert body["question"]["recommendation"]["action"] == "deny"


def test_customer_id_is_validated(client):
    assert client.post("/claims", json=new_claim(customer_id="x")).status_code == 422


def test_any_replica_answers_for_a_run_on_another():
    d, saver, index, rs = (
        make_deps(),
        InMemorySaver(),
        fakes.FakeIndex(),
        fakes.FakeRunState(),
    )
    with (
        TestClient(replica(d, saver, index, rs)) as a,
        TestClient(replica(d, saver, index, rs)) as b,
    ):
        a.deps = d
        gate = block_vision(a)
        a.post("/claims", json=new_claim(customer_id="C00001"))
        # Seen from the other pod: running, owned by C00001, and it can't be
        # started again there.
        body = b.get(f"/claims/{CLAIM}").json()
        assert (body["status"], body["customer_id"]) == ("processing", "C00001")
        assert b.post(f"/claims/{CLAIM}/retry").status_code == 409
        assert b.get("/claims").json()["claims"][0]["status"] == "processing"
        wait_until_in_damage_analysis(b)
        assert b.post(f"/claims/{CLAIM}/retry").status_code == 409

        gate.open = True
        wait_for(b, "awaiting_approval")
        # The decision lands on B, which resumes the run A paused.
        r = b.post(
            f"/claims/{CLAIM}/decision", json={"action": "approve", "adjuster": "sam"}
        )
        assert r.status_code == 202
        wait_for(a, "completed")


def test_a_failure_on_one_replica_is_seen_on_the_other():
    d, saver, index, rs = (
        make_deps(),
        InMemorySaver(),
        fakes.FakeIndex(),
        fakes.FakeRunState(),
    )

    async def broken(node, server, tool, args):
        raise RuntimeError("pricing server down")

    d.tools.call = broken
    with (
        TestClient(replica(d, saver, index, rs)) as a,
        TestClient(replica(d, saver, index, rs)) as b,
    ):
        a.post("/claims", json=new_claim())
        body = wait_for(b, "failed")
        assert "pricing server down" in body["error"]


def test_shutdown_drains_runs_then_cancels_the_rest():
    d, saver, index, rs = (
        make_deps(),
        InMemorySaver(),
        fakes.FakeIndex(),
        fakes.FakeRunState(),
    )
    with TestClient(replica(d, saver, index, rs, drain_s=0.2)) as a:
        a.deps = d
        block_vision(a)  # never opened: outlasts the drain
        a.post("/claims", json=new_claim())
        wait_until_in_damage_analysis(a)
    # The pod has shut down: the run was cancelled and released its lease, so
    # another replica says "interrupted" at once, without waiting for the TTL.
    assert rs.leases == {}
    with TestClient(replica(make_deps(), saver, index, rs)) as b:
        assert b.get(f"/claims/{CLAIM}").json()["status"] == "interrupted"
        assert b.post(f"/claims/{CLAIM}/retry").status_code == 202
        wait_for(b, "awaiting_approval")


def test_a_run_that_lost_its_lease_stops(client, monkeypatch):
    monkeypatch.setattr(runstate, "LEASE_S", 0.06)  # renew every 20 ms
    block_vision(client)
    client.post("/claims", json=new_claim())
    wait_until_in_damage_analysis(client)
    task = client.app.state.runs[CLAIM]
    # The lease expired and another replica took the claim over.
    client.app.state.runstate.leases[CLAIM] = "another replica"
    deadline = time.time() + 5
    while not task.done():
        assert time.time() < deadline
        time.sleep(0.02)
    assert task.cancelled()
    # Its lease isn't touched: it belongs to the other replica.
    assert client.app.state.runstate.leases[CLAIM] == "another replica"


def test_metrics_are_exposed(client):
    client.get(f"/claims/{'d' * 32}")
    text = client.get("/metrics").text
    assert (
        'http_requests_total{method="GET",route="/claims/{claim_id}",status="404"}'
        in text
    )
