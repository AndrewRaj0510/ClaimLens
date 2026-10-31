import json
import os

import pytest
from fastapi.testclient import TestClient

import common
import fakes
from auth import current_user
from common import vision_jobs
from main import app
from vision_jobs import STREAM, VisionJobs

# Not entered as a context manager, so the lifespan (Redis, S3) doesn't run;
# the job queue is replaced per test via dependency_overrides.
client = TestClient(app)

RESULT = {"width": 10, "height": 10, "detections": [], "damage_area_pct": 0.0}
JOB_ID = "0123456789abcdef0123456789abcdef"


class FakeJobs:
    """Stands in for VisionJobs: records submissions, returns a canned result."""

    def __init__(self, result=None, status=None, fail=False):
        self.submitted = []
        self.result = result
        self._status = status
        self.fail = fail

    async def submit(self, data, content_type):
        if self.fail:
            raise ConnectionError("redis down")
        self.submitted.append((data, content_type))
        return JOB_ID

    async def wait(self, job_id, timeout_s):
        return self.result

    async def status(self, job_id):
        return self._status


@pytest.fixture
def jobs():
    def use(fake):
        app.dependency_overrides[vision_jobs] = lambda: fake
        return fake

    # Logged in: the login itself is tested in test_auth.py.
    app.dependency_overrides[current_user] = lambda: fakes.ADJUSTER

    yield use
    app.dependency_overrides.clear()


def upload(data=b"jpegbytes", ctype="image/jpeg"):
    return client.post("/detect", files={"file": ("car.jpg", data, ctype)})


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "gateway"}


def test_detect_enqueues_and_returns_the_result(jobs):
    fake = jobs(FakeJobs(result={"status": "done", "result": RESULT}))
    r = upload(b"jpegbytes")
    assert r.status_code == 200
    assert r.json() == RESULT
    assert fake.submitted == [(b"jpegbytes", "image/jpeg")]


def test_slow_job_returns_202_with_job_id(jobs):
    jobs(FakeJobs(result=None))
    r = upload()
    assert r.status_code == 202
    assert r.json() == {"job_id": JOB_ID, "status": "pending"}


def test_non_image_type_is_rejected_before_enqueueing(jobs):
    fake = jobs(FakeJobs())
    assert upload(ctype="text/plain").status_code == 415
    assert fake.submitted == []


def test_oversized_upload_is_rejected(jobs, monkeypatch):
    fake = jobs(FakeJobs())
    monkeypatch.setattr(common, "MAX_UPLOAD_MB", 0.001)  # ~1 KB
    assert upload(b"x" * 2000).status_code == 413
    assert fake.submitted == []


def test_queue_down_is_503(jobs):
    jobs(FakeJobs(fail=True))
    assert upload().status_code == 503


@pytest.mark.parametrize(
    ("worker_code", "expected"),
    [(400, 400), (404, 404), (413, 413), (503, 503), (500, 502)],
)
def test_worker_errors_are_mapped(jobs, worker_code, expected):
    jobs(FakeJobs(result={"status": "error", "code": worker_code, "detail": "nope"}))
    r = upload()
    assert r.status_code == expected
    assert r.json()["detail"] == ("nope" if expected != 502 else "Vision service error")


@pytest.mark.parametrize(
    ("status", "code"),
    [
        ({"status": "done", "result": RESULT}, 200),
        ({"status": "pending"}, 202),
        (None, 404),
    ],
)
def test_get_job(jobs, status, code):
    jobs(FakeJobs(status=status))
    assert client.get(f"/jobs/{JOB_ID}").status_code == code


def test_job_id_must_be_a_job_id(jobs):
    jobs(FakeJobs(status={"status": "pending"}))
    assert client.get("/jobs/..%2F..%2Fetc").status_code in (404, 422)
    assert client.get("/jobs/NOT-A-JOB").status_code == 422


# ---- VisionJobs against a real Redis (fake object store) ----

TEST_REDIS = os.getenv("TEST_REDIS_URL")


class MemoryStore:
    def __init__(self):
        self.objects = {}

    def put(self, key, data, content_type):
        self.objects[key] = (data, content_type)


@pytest.mark.skipif(not TEST_REDIS, reason="set TEST_REDIS_URL to run")
@pytest.mark.anyio
async def test_vision_jobs_round_trip():
    import redis.asyncio as redis

    r = redis.Redis.from_url(TEST_REDIS, decode_responses=True)
    await r.flushdb()
    store = MemoryStore()
    jobs = VisionJobs(r, store)
    try:
        job_id = await jobs.submit(b"img", "image/png")
        # The image is stored before the job is visible on the stream.
        [(_, fields)] = await r.xrange(STREAM)
        assert fields == {"job_id": job_id, "image_key": f"uploads/{job_id}"}
        assert store.objects[f"uploads/{job_id}"] == (b"img", "image/png")
        assert await jobs.status(job_id) == {"status": "pending"}
        assert await jobs.wait(job_id, 0.1) is None  # nothing published yet

        # Simulate the worker publishing (same keys as worker.publish).
        done = json.dumps({"status": "done", "result": RESULT})
        await r.set(f"vision:result:{job_id}", done)
        await r.rpush(f"vision:done:{job_id}", done)
        assert await jobs.wait(job_id, 1) == {"status": "done", "result": RESULT}
        assert (await jobs.status(job_id))["status"] == "done"
        assert await jobs.status("f" * 32) is None
    finally:
        await r.flushdb()
        await r.aclose()


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_metrics_count_requests_by_route_template(jobs):
    jobs(FakeJobs(status={"status": "pending"}))
    client.get("/health")
    client.get(f"/jobs/{JOB_ID}")
    client.get("/no-such-page")
    text = client.get("/metrics").text
    assert 'http_requests_total{method="GET",route="/health",status="200"}' in text
    # The template, never the ID: one series per route, however many jobs.
    assert 'route="/jobs/{job_id}"' in text and JOB_ID not in text
    assert 'route="unmatched",status="404"' in text
    assert 'route="/metrics"' not in text  # scrapes aren't counted
    assert "http_request_duration_seconds_bucket" in text


class BlockingRedis:
    """BLPOP that holds until released, counting the waiters (connections)."""

    def __init__(self):
        import asyncio

        self.release = asyncio.Event()
        self.waiting = 0

    async def blpop(self, key, timeout):
        self.waiting += 1
        await self.release.wait()
        return key, json.dumps({"status": "done", "result": RESULT})


def test_waiters_are_capped_so_redis_connections_are_left_for_logins():
    # Found in the load test: 100+ uploads waiting in BLPOP took every
    # connection in the pool, and logins failed with them.
    import asyncio

    async def scenario():
        r = BlockingRedis()
        jobs = VisionJobs(r, store=None, max_waiters=2)
        waiting = [asyncio.create_task(jobs.wait(f"job{i}", 30)) for i in range(2)]
        while r.waiting < 2:
            await asyncio.sleep(0)
        # The third gets no connection: None at once, i.e. 202 + job ID to poll.
        assert await asyncio.wait_for(jobs.wait("job3", 30), timeout=1) is None
        assert r.waiting == 2
        r.release.set()
        assert [(await t)["status"] for t in waiting] == ["done", "done"]

    asyncio.run(scenario())
