import json

import httpx
import pytest
from fastapi.testclient import TestClient

import claims
from common import agent_client, vision_jobs
from main import app

client = TestClient(app)
CLAIM = "e" * 32
FORM = {
    "policy_id": "P00001",
    "incident_date": "2026-09-01",
    "description": "Rear-ended at a red light on Main Street.",
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


@pytest.fixture
def env(monkeypatch):
    """Fake queue + a mock agent that records requests and answers from `replies`."""
    jobs = FakeJobs()
    seen = []
    replies = {}

    def handler(request):
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, body))
        status, payload = replies.get(
            (request.method, request.url.path), (202, {"status": "processing"})
        )
        return httpx.Response(status, json=payload)

    agent = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://agent"
    )
    app.dependency_overrides[vision_jobs] = lambda: jobs
    app.dependency_overrides[agent_client] = lambda: agent
    monkeypatch.setattr(claims, "ADJUSTER_API_KEY", "s3cret-adjuster-key")
    yield jobs, seen, replies
    app.dependency_overrides.clear()


def photos(n=1, ctype="image/jpeg", size=10):
    return [("photos", (f"car{i}.jpg", b"x" * size, ctype)) for i in range(n)]


def test_claim_photos_are_stored_queued_then_the_agent_starts(env):
    jobs, seen, _ = env
    r = client.post("/claims", data={**FORM, "claimed_amount": "900"}, files=photos(2))
    assert r.status_code == 202

    keys = [k for k, _, _ in jobs.submitted]
    assert len(keys) == 2 and all(k.startswith("claims/") for k in keys)
    claim_id = keys[0].split("/")[1]
    assert all(k.split("/")[1] == claim_id for k in keys)  # one folder per claim
    assert not any("car0" in k for k in keys)  # user filenames never used

    method, path, body = seen[0]
    assert (method, path) == ("POST", "/claims")
    assert body["claim_id"] == claim_id
    assert body["claimed_amount"] == 900
    assert [p["key"] for p in body["photos"]] == keys


@pytest.mark.parametrize(
    ("form", "files", "code"),
    [
        ({**FORM, "policy_id": "nope"}, photos(), 422),
        (FORM, photos(11), 422),  # too many
        (FORM, photos(ctype="application/pdf"), 415),
        ({**FORM, "description": "short"}, photos(), 422),
    ],
)
def test_bad_claims_are_rejected_before_anything_is_stored(env, form, files, code):
    jobs, seen, _ = env
    assert client.post("/claims", data=form, files=files).status_code == code
    assert jobs.submitted == [] and seen == []


def test_one_bad_photo_rejects_the_whole_upload(env):
    jobs, _, _ = env
    files = [*photos(1), ("photos", ("x.txt", b"text", "text/plain"))]
    assert client.post("/claims", data=FORM, files=files).status_code == 415
    assert jobs.submitted == []  # validated all before storing any


def test_agent_errors_are_mapped(env):
    _, _, replies = env
    replies[("GET", f"/claims/{CLAIM}")] = (404, {"detail": "Unknown claim"})
    r = client.get(f"/claims/{CLAIM}")
    assert r.status_code == 404 and r.json()["detail"] == "Unknown claim"
    replies[("GET", f"/claims/{CLAIM}")] = (500, {"detail": "Traceback..."})
    r = client.get(f"/claims/{CLAIM}")
    assert r.status_code == 502 and "Traceback" not in r.text  # no internals leaked


def test_retake_only_when_the_claim_asks_for_photos(env):
    jobs, seen, replies = env
    replies[("GET", f"/claims/{CLAIM}")] = (200, {"status": "awaiting_approval"})
    assert client.post(f"/claims/{CLAIM}/photos", files=photos()).status_code == 409
    assert jobs.submitted == []

    replies[("GET", f"/claims/{CLAIM}")] = (200, {"status": "awaiting_retake"})
    assert client.post(f"/claims/{CLAIM}/photos", files=photos()).status_code == 202
    assert jobs.submitted[0][0].startswith(f"claims/{CLAIM}/")
    assert seen[-1][1] == f"/claims/{CLAIM}/photos"


@pytest.mark.parametrize("key", [None, "", "wrong-key", "s3cret-adjuster-ke"])
def test_decision_needs_the_adjuster_key(env, key):
    _, seen, _ = env
    headers = {"X-Adjuster-Key": key} if key is not None else {}
    r = client.post(
        f"/claims/{CLAIM}/decision",
        json={"action": "approve", "adjuster": "sam"},
        headers=headers,
    )
    assert r.status_code == 401
    assert seen == []  # never reached the agent


def test_decision_with_the_key_is_forwarded(env):
    _, seen, _ = env
    decision = {"action": "approve", "adjuster": "sam"}
    r = client.post(
        f"/claims/{CLAIM}/decision",
        json=decision,
        headers={"X-Adjuster-Key": "s3cret-adjuster-key"},
    )
    assert r.status_code == 202
    assert seen == [("POST", f"/claims/{CLAIM}/decision", decision)]


def test_unset_adjuster_key_locks_decisions(env, monkeypatch):
    monkeypatch.setattr(claims, "ADJUSTER_API_KEY", "")
    r = client.post(
        f"/claims/{CLAIM}/decision",
        json={"action": "approve", "adjuster": "x"},
        headers={"X-Adjuster-Key": ""},
    )
    assert r.status_code == 401


def test_storage_down_is_503(env):
    jobs, seen, _ = env
    jobs.fail = True
    assert client.post("/claims", data=FORM, files=photos()).status_code == 503
    assert seen == []


def test_template_placeholder_key_does_not_work(env, monkeypatch):
    # Someone copied .env.example's old "change-me" without editing it.
    monkeypatch.setattr(claims, "ADJUSTER_API_KEY", "change-me")
    r = client.post(
        f"/claims/{CLAIM}/decision",
        json={"action": "approve", "adjuster": "x"},
        headers={"X-Adjuster-Key": "change-me"},
    )
    assert r.status_code == 401


def test_non_ascii_key_is_401_not_500(env):
    _, seen, _ = env
    r = client.post(
        f"/claims/{CLAIM}/decision",
        json={"action": "approve", "adjuster": "x"},
        headers={"X-Adjuster-Key": "s3cret-adjuster-kéy".encode("latin-1")},
    )
    assert r.status_code == 401
    assert seen == []


def test_future_incident_is_rejected_before_storing_photos(env):
    jobs, seen, _ = env
    r = client.post(
        "/claims", data={**FORM, "incident_date": "2099-01-01"}, files=photos()
    )
    assert r.status_code == 422
    assert jobs.submitted == [] and seen == []
