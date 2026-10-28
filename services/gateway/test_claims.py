import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import claims
import fakes
from auth import COOKIE, rate_limiter, sessions, user_store
from common import agent_client, photo_store, vision_jobs
from main import app

CLAIM = "e" * 32
FORM = {
    "policy_id": "P00001",
    "incident_date": "2026-09-01",
    "description": "Rear-ended at a red light on Main Street.",
}


def full_claim(customer_id="C00001", status="awaiting_approval", **state):
    """What the agent returns for GET /claims/{id}."""
    return {
        "claim_id": CLAIM,
        "customer_id": customer_id,
        "status": status,
        "question": {
            "type": "approval",
            "recommendation": {"action": "approve", "reasons": []},
            "fraud": {"score": 0.1, "level": "low"},
        },
        "error": None,
        "state": {
            "customer_id": customer_id,
            "policy_id": "P00001",
            "incident_date": "2026-09-01",
            "description": "Rear-ended.",
            "photos": [{"key": f"claims/{CLAIM}/abc", "job_id": "1" * 32}],
            "fraud": {"score": 0.1, "level": "low", "signals": []},
            "recommendation": {"action": "approve"},
            **state,
        },
        "trace": [{"node": "intake"}],
    }


class FakeJobs:
    def __init__(self, fail=False):
        self.submitted = []
        self.fail = fail

    async def submit(self, data, content_type, key=None):
        if self.fail:
            raise ConnectionError("storage down")
        self.submitted.append((key, data, content_type))
        return f"{len(self.submitted):032x}"


class FakeStore:
    def __init__(self):
        self.objects = {f"claims/{CLAIM}/abc": (b"\xff\xd8jpeg", "image/jpeg")}

    def get(self, key):
        return self.objects[key]


class Env:
    def __init__(self):
        self.jobs, self.store = FakeJobs(), FakeStore()
        self.users, self.sessions = fakes.FakeUsers(), fakes.FakeSessions()
        self.limiter = fakes.FakeLimiter()
        self.seen, self.replies = [], {}
        self.client = TestClient(app, base_url="https://testserver")

    def handler(self, request):
        body = json.loads(request.content) if request.content else None
        self.seen.append(
            (request.method, request.url.path, body, dict(request.url.params))
        )
        status, payload = self.replies.get(
            (request.method, request.url.path), (202, {"status": "processing"})
        )
        return httpx.Response(status, json=payload)

    def login(self, user):
        token = asyncio.run(self.sessions.create(user))
        self.client.cookies.set(COOKIE, token)
        return self.client

    def requests(self):
        return [(m, p) for m, p, _, _ in self.seen]


@pytest.fixture
def env():
    e = Env()
    agent = httpx.AsyncClient(
        transport=httpx.MockTransport(e.handler), base_url="http://agent"
    )
    app.dependency_overrides.update(
        {
            vision_jobs: lambda: e.jobs,
            agent_client: lambda: agent,
            photo_store: lambda: e.store,
            user_store: lambda: e.users,
            sessions: lambda: e.sessions,
            rate_limiter: lambda: e.limiter,
        }
    )
    yield e
    app.dependency_overrides.clear()


def photos(n=1, ctype="image/jpeg", size=10):
    return [("photos", (f"car{i}.jpg", b"x" * size, ctype)) for i in range(n)]


# ---- filing a claim ----


def test_claim_photos_are_stored_queued_then_the_agent_starts(env):
    client = env.login(fakes.CUSTOMER)
    r = client.post("/claims", data={**FORM, "claimed_amount": "900"}, files=photos(2))
    assert r.status_code == 202

    keys = [k for k, _, _ in env.jobs.submitted]
    assert len(keys) == 2 and all(k.startswith("claims/") for k in keys)
    claim_id = keys[0].split("/")[1]
    assert all(k.split("/")[1] == claim_id for k in keys)  # one folder per claim
    assert not any("car0" in k for k in keys)  # user filenames never used

    method, path, body, _ = env.seen[0]
    assert (method, path) == ("POST", "/claims")
    assert body["claim_id"] == claim_id
    assert body["customer_id"] == "C00001"  # from the session, not the form
    assert body["claimed_amount"] == 900
    assert [p["key"] for p in body["photos"]] == keys


@pytest.mark.parametrize(
    ("form", "files", "code"),
    [
        ({**FORM, "policy_id": "nope"}, photos(), 422),
        ({**FORM, "policy_id": "P00002"}, photos(), 422),  # someone else's policy
        (FORM, photos(11), 422),  # too many
        (FORM, photos(ctype="application/pdf"), 415),
        ({**FORM, "description": "short"}, photos(), 422),
        ({**FORM, "incident_date": "2099-01-01"}, photos(), 422),  # future
    ],
)
def test_bad_claims_are_rejected_before_anything_is_stored(env, form, files, code):
    client = env.login(fakes.CUSTOMER)
    assert client.post("/claims", data=form, files=files).status_code == code
    assert env.jobs.submitted == [] and env.seen == []


def test_someone_elses_policy_looks_like_an_unknown_one(env):
    client = env.login(fakes.CUSTOMER)
    theirs = client.post(
        "/claims", data={**FORM, "policy_id": "P00002"}, files=photos()
    )
    missing = client.post(
        "/claims", data={**FORM, "policy_id": "P99999"}, files=photos()
    )
    assert theirs.json() == missing.json() == {"detail": "Unknown policy"}


def test_one_bad_photo_rejects_the_whole_upload(env):
    client = env.login(fakes.CUSTOMER)
    files = [*photos(1), ("photos", ("x.txt", b"text", "text/plain"))]
    assert client.post("/claims", data=FORM, files=files).status_code == 415
    assert env.jobs.submitted == []  # validated all before storing any


def test_only_customers_file_claims(env):
    client = env.login(fakes.ADJUSTER)
    assert client.post("/claims", data=FORM, files=photos()).status_code == 403
    assert env.jobs.submitted == []


def test_submissions_are_rate_limited_per_user(env):
    client = env.login(fakes.CUSTOMER)
    env.limiter.counts["submit:daniel.brown1@example.com"] = claims.SUBMISSIONS_PER_HOUR
    r = client.post("/claims", data=FORM, files=photos())
    assert r.status_code == 429
    assert "retry-after" in r.headers
    assert env.jobs.submitted == []


def test_storage_down_is_503(env):
    client = env.login(fakes.CUSTOMER)
    env.jobs.fail = True
    assert client.post("/claims", data=FORM, files=photos()).status_code == 503
    assert env.seen == []


def test_my_policies(env):
    client = env.login(fakes.CUSTOMER)
    assert [p["policy_id"] for p in client.get("/me/policies").json()["policies"]] == [
        "P00001"
    ]
    assert env.login(fakes.ADJUSTER).get("/me/policies").status_code == 403


# ---- who sees what ----


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/claims"),
        ("GET", f"/claims/{CLAIM}"),
        ("GET", f"/claims/{CLAIM}/photos/0"),
        ("POST", f"/claims/{CLAIM}/decision"),
        ("POST", f"/claims/{CLAIM}/retry"),
        ("GET", "/me/policies"),
        ("POST", "/detect"),
        ("GET", f"/jobs/{CLAIM}"),
    ],
)
def test_everything_needs_a_login(env, method, path):
    r = env.client.request(method, path)
    assert r.status_code == 401
    assert env.seen == []


def test_customer_can_see_their_claim_before_its_first_checkpoint(env):
    # The agent's answer while the run hasn't saved anything yet.
    early = {
        "claim_id": CLAIM,
        "customer_id": "C00001",
        "status": "processing",
        "question": None,
        "error": None,
        "state": {},
        "trace": [],
    }
    env.replies[("GET", f"/claims/{CLAIM}")] = (200, early)
    r = env.login(fakes.CUSTOMER).get(f"/claims/{CLAIM}")
    assert r.status_code == 200
    assert r.json()["status"] == "processing"
    assert r.json()["photo_count"] == 0


def test_customer_view_leaves_out_internal_assessment(env):
    env.replies[("GET", f"/claims/{CLAIM}")] = (200, full_claim())
    body = env.login(fakes.CUSTOMER).get(f"/claims/{CLAIM}").json()
    assert body["status"] == "awaiting_approval"
    assert body["question"] is None  # the approval question carries the fraud score
    assert body["photo_count"] == 1
    text = json.dumps(body)
    for internal in ("fraud", "recommendation", "trace", "signals", "customer_id"):
        assert internal not in text


def test_customer_sees_retake_requests_and_the_outcome(env):
    retake = full_claim(status="awaiting_retake")
    retake["question"] = {
        "type": "retake",
        "photos": [{"photo_index": 0, "problems": ["Blurry"]}],
    }
    env.replies[("GET", f"/claims/{CLAIM}")] = (200, retake)
    client = env.login(fakes.CUSTOMER)
    assert client.get(f"/claims/{CLAIM}").json()["question"] == retake["question"]

    done = full_claim(
        status="completed",
        decision={"action": "edit", "amount": 400.0, "adjuster": "sam", "notes": "n"},
        report={
            "customer_message": "Approved for $400.",
            "adjuster_summary": "internal",
        },
    )
    env.replies[("GET", f"/claims/{CLAIM}")] = (200, done)
    body = client.get(f"/claims/{CLAIM}").json()
    assert body["outcome"] == {"action": "edit", "amount": 400.0}
    assert body["message"] == "Approved for $400."
    assert "internal" not in json.dumps(body) and "sam" not in json.dumps(body)


def test_adjuster_sees_everything(env):
    env.replies[("GET", f"/claims/{CLAIM}")] = (200, full_claim())
    body = env.login(fakes.ADJUSTER).get(f"/claims/{CLAIM}").json()
    assert body == full_claim()


@pytest.mark.parametrize("owner", ["C00002", None])
def test_other_peoples_claims_are_404(env, owner):
    env.replies[("GET", f"/claims/{CLAIM}")] = (200, full_claim(customer_id=owner))
    client = env.login(fakes.CUSTOMER)
    for r in (
        client.get(f"/claims/{CLAIM}"),
        client.get(f"/claims/{CLAIM}/photos/0"),
        client.post(f"/claims/{CLAIM}/photos", files=photos()),
    ):
        assert r.status_code == 404
        assert r.json()["detail"] == "Unknown claim"
    assert env.jobs.submitted == []


def test_customer_list_is_scoped_and_reduced(env):
    row = {
        "claim_id": CLAIM,
        "customer_id": "C00001",
        "policy_id": "P00001",
        "status": "completed",
        "created_at": "2026-10-27T10:00:00+00:00",
        "updated_at": "2026-10-27T10:05:00+00:00",
        "summary": {
            "incident_date": "2026-09-01",
            "damages": ["dent (minor)"],
            "fraud_level": "high",
            "recommendation": "deny",
            "decision": {"action": "approve", "amount": 100.0},
        },
    }
    env.replies[("GET", "/claims")] = (200, {"claims": [row]})
    rows = env.login(fakes.CUSTOMER).get("/claims?status=completed").json()["claims"]
    assert env.seen[-1][3] == {
        "limit": "50",
        "status": "completed",
        "customer_id": "C00001",
    }
    assert rows[0]["damages"] == ["dent (minor)"]
    assert "high" not in json.dumps(rows) and "deny" not in json.dumps(rows)

    rows = env.login(fakes.ADJUSTER).get("/claims").json()["claims"]
    assert env.seen[-1][3] == {"limit": "50"}  # every customer
    assert rows == [row]


def test_list_filter_is_validated(env):
    assert env.login(fakes.ADJUSTER).get("/claims?status=bogus").status_code == 422
    assert env.seen == []


# ---- photos ----


def test_photo_is_streamed_with_safe_headers(env):
    env.replies[("GET", f"/claims/{CLAIM}")] = (200, full_claim())
    r = env.login(fakes.CUSTOMER).get(f"/claims/{CLAIM}/photos/0")
    assert r.status_code == 200
    assert r.content == b"\xff\xd8jpeg"
    assert r.headers["content-type"] == "image/jpeg"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["cache-control"].startswith("private")


def test_photo_with_an_unexpected_stored_type_is_not_served_as_that_type(env):
    env.replies[("GET", f"/claims/{CLAIM}")] = (200, full_claim())
    env.store.objects[f"claims/{CLAIM}/abc"] = (b"<script>", "text/html")
    r = env.login(fakes.ADJUSTER).get(f"/claims/{CLAIM}/photos/0")
    assert r.headers["content-type"] == "application/octet-stream"


def test_missing_photos_are_404(env):
    env.replies[("GET", f"/claims/{CLAIM}")] = (200, full_claim())
    client = env.login(fakes.ADJUSTER)
    assert client.get(f"/claims/{CLAIM}/photos/1").status_code == 404  # index
    assert client.get(f"/claims/{CLAIM}/photos/99").status_code == 422
    env.store.objects.clear()
    assert client.get(f"/claims/{CLAIM}/photos/0").status_code == 404  # object


# ---- retakes, decisions, retries ----


def test_retake_only_when_the_claim_asks_for_photos(env):
    client = env.login(fakes.CUSTOMER)
    env.replies[("GET", f"/claims/{CLAIM}")] = (200, full_claim())
    assert client.post(f"/claims/{CLAIM}/photos", files=photos()).status_code == 409
    assert env.jobs.submitted == []

    env.replies[("GET", f"/claims/{CLAIM}")] = (
        200,
        full_claim(status="awaiting_retake"),
    )
    assert client.post(f"/claims/{CLAIM}/photos", files=photos()).status_code == 202
    assert env.jobs.submitted[0][0].startswith(f"claims/{CLAIM}/")
    assert env.requests()[-1] == ("POST", f"/claims/{CLAIM}/photos")


def test_customers_cannot_decide_or_retry(env):
    client = env.login(fakes.CUSTOMER)
    for path in (f"/claims/{CLAIM}/decision", f"/claims/{CLAIM}/retry"):
        r = client.post(path, json={"action": "approve"})
        assert r.status_code == 403
    assert env.seen == []  # never reached the agent


def test_decision_records_the_logged_in_adjuster(env):
    client = env.login(fakes.ADJUSTER)
    r = client.post(
        f"/claims/{CLAIM}/decision",
        json={"action": "approve", "adjuster": "someone-else", "notes": "ok"},
    )
    assert r.status_code == 202
    assert env.seen[-1][:3] == (
        "POST",
        f"/claims/{CLAIM}/decision",
        {"action": "approve", "adjuster": "sam", "notes": "ok"},
    )


def test_retry_is_forwarded_for_adjusters(env):
    assert env.login(fakes.ADJUSTER).post(f"/claims/{CLAIM}/retry").status_code == 202
    assert env.requests() == [("POST", f"/claims/{CLAIM}/retry")]


def test_agent_errors_are_mapped(env):
    client = env.login(fakes.ADJUSTER)
    env.replies[("GET", f"/claims/{CLAIM}")] = (404, {"detail": "Unknown claim"})
    r = client.get(f"/claims/{CLAIM}")
    assert r.status_code == 404 and r.json()["detail"] == "Unknown claim"
    env.replies[("GET", f"/claims/{CLAIM}")] = (500, {"detail": "Traceback..."})
    r = client.get(f"/claims/{CLAIM}")
    assert r.status_code == 502 and "Traceback" not in r.text  # no internals leaked
