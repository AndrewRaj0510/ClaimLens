import io
import json
import os

import numpy as np
import pytest
from PIL import Image

import worker
from detector import Detection
from storage import ObjectNotFound


class FakeDetector:
    def __init__(self):
        self.names = {0: "dent"}

    def __call__(self, image, conf=0.25):
        w, h = image.size
        mask = np.zeros((h, w), dtype=bool)
        mask[:, : w // 4] = True
        return [Detection(0, 0.9, (0.0, 0.0, w / 4, h), mask)]


class CrashingDetector:
    def __init__(self):
        self.names = {0: "dent"}

    def __call__(self, image, conf=0.25):
        raise RuntimeError("onnxruntime blew up")


class FakeStore:
    def __init__(self, objects):
        self.objects = objects

    def get(self, key, max_bytes):
        if key not in self.objects:
            raise ObjectNotFound(key)
        return self.objects[key][: max_bytes + 1]


def jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (200, 100), (90, 90, 90)).save(buf, "JPEG")
    return buf.getvalue()


STORE = FakeStore({"uploads/ok": jpeg(), "uploads/junk": b"not an image"})


def test_job_result_for_a_good_image():
    out = worker.run_job({"image_key": "uploads/ok"}, FakeDetector(), STORE)
    assert out["status"] == "done"
    assert out["result"]["detections"][0]["class_name"] == "dent"
    assert out["result"]["damage_area_pct"] == 25.0


@pytest.mark.parametrize(
    ("key", "detector", "code"),
    [
        ("uploads/junk", FakeDetector(), 400),
        ("uploads/missing", FakeDetector(), 404),
        ("uploads/ok", None, 503),
    ],
)
def test_expected_failures_become_error_results(key, detector, code):
    out = worker.run_job({"image_key": key}, detector, STORE)
    assert out == {"status": "error", "code": code, "detail": out["detail"]}


def test_oversized_object_is_rejected(monkeypatch):
    import analysis

    monkeypatch.setattr(analysis, "MAX_UPLOAD_MB", 0.0001)  # ~100 bytes
    out = worker.run_job({"image_key": "uploads/ok"}, FakeDetector(), STORE)
    assert out["code"] == 413


def test_unexpected_errors_raise_so_the_job_is_retried():
    with pytest.raises(RuntimeError):
        worker.run_job({"image_key": "uploads/ok"}, CrashingDetector(), STORE)


# ---- against a real Redis ----
# Uses database 15 and flushes it: point TEST_REDIS_URL at a Redis you don't mind
# emptying, e.g. redis://:<pw>@127.0.0.1:6380/15

TEST_REDIS = os.getenv("TEST_REDIS_URL")
needs_redis = pytest.mark.skipif(not TEST_REDIS, reason="set TEST_REDIS_URL to run")


@pytest.fixture
def r(tmp_path, monkeypatch):
    import redis

    monkeypatch.setattr(worker, "HEARTBEAT", tmp_path / "heartbeat")
    monkeypatch.setattr(worker, "BLOCK_MS", 100)
    client = redis.Redis.from_url(TEST_REDIS, decode_responses=True)
    client.flushdb()
    yield client
    client.flushdb()


def enqueue(r, job_id, key="uploads/ok"):
    return r.xadd(worker.STREAM, {"job_id": job_id, "image_key": key})


def run_once(r, detector, consumer="w1"):
    calls = iter([False, True])  # one loop iteration, then stop
    worker.run(r, detector, STORE, consumer, lambda: next(calls))


@needs_redis
def test_job_is_processed_published_and_acked(r):
    worker.ensure_group(r)
    enqueue(r, "job1")
    run_once(r, FakeDetector())

    result = json.loads(r.get("vision:result:job1"))
    assert result["status"] == "done"
    _, notified = r.blpop("vision:done:job1", timeout=1)
    assert json.loads(notified) == result
    assert r.ttl("vision:result:job1") > 0  # results expire
    assert r.xpending(worker.STREAM, worker.GROUP)["pending"] == 0
    assert (worker.HEARTBEAT).exists()


@needs_redis
def test_job_left_by_a_dead_worker_is_reclaimed(r, monkeypatch):
    worker.ensure_group(r)
    enqueue(r, "job2")
    # Worker "w1" reads the job and dies before acking.
    r.xreadgroup(worker.GROUP, "w1", {worker.STREAM: ">"}, count=1)
    assert r.xpending(worker.STREAM, worker.GROUP)["pending"] == 1

    monkeypatch.setattr(worker, "CLAIM_IDLE_MS", 0)
    run_once(r, FakeDetector(), consumer="w2")

    assert json.loads(r.get("vision:result:job2"))["status"] == "done"
    assert r.xpending(worker.STREAM, worker.GROUP)["pending"] == 0


@needs_redis
def test_poison_job_is_dead_lettered_after_max_deliveries(r, monkeypatch):
    monkeypatch.setattr(worker, "CLAIM_IDLE_MS", 0)
    worker.ensure_group(r)
    enqueue(r, "poison")

    # First delivery and each retry crash; the entry stays pending each time.
    run_once(r, CrashingDetector())
    for _ in range(worker.MAX_DELIVERIES - 1):
        worker.reclaim(r, "w1", CrashingDetector(), STORE)
        assert r.get("vision:result:poison") is None
    # The next reclaim sees it has failed MAX_DELIVERIES times.
    worker.reclaim(r, "w1", CrashingDetector(), STORE)

    result = json.loads(r.get("vision:result:poison"))
    assert result == {
        "status": "error",
        "code": 500,
        "detail": "Detection failed repeatedly",
    }
    assert r.xpending(worker.STREAM, worker.GROUP)["pending"] == 0
    dead = r.xrange(worker.DEAD)
    assert [fields["job_id"] for _, fields in dead] == ["poison"]


@needs_redis
def test_ensure_group_is_idempotent(r):
    worker.ensure_group(r)
    worker.ensure_group(r)  # BUSYGROUP is swallowed
    assert [g["name"] for g in r.xinfo_groups(worker.STREAM)] == [worker.GROUP]


def test_socket_timeout_outlasts_the_blocking_read():
    # Regression: redis-py 8's default 5 s socket timeout equals BLOCK_MS, so
    # every idle worker crashed on its first empty read.
    r = worker.connect("redis://localhost:6379/0")
    assert (
        r.connection_pool.connection_kwargs["socket_timeout"] > worker.BLOCK_MS / 1000
    )


class FlakyRedis:
    """Fails like a restarting Redis on the first call, then works."""

    def __init__(self):
        self.calls = 0

    def xgroup_create(self, *args, **kwargs):
        from redis.exceptions import ConnectionError

        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("Connection refused")

    def xpending_range(self, *args, **kwargs):
        return []

    def xinfo_consumers(self, *args, **kwargs):
        return []

    def xreadgroup(self, *args, **kwargs):
        return []


def test_worker_survives_a_redis_outage(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "HEARTBEAT", tmp_path / "heartbeat")
    flaky = FlakyRedis()
    calls = iter([False, False, True])  # two loop iterations
    worker.run(flaky, FakeDetector(), STORE, "w1", lambda: next(calls), backoff_s=0)
    assert flaky.calls == 2  # the group is re-created after the error


@needs_redis
def test_stale_consumers_are_forgotten_but_busy_ones_kept(r, monkeypatch):
    worker.ensure_group(r)
    enqueue(r, "held")
    # "old-idle" is registered but holds nothing; "old-busy" holds a pending entry.
    r.xgroup_createconsumer(worker.STREAM, worker.GROUP, "old-idle")
    r.xreadgroup(worker.GROUP, "old-busy", {worker.STREAM: ">"}, count=1)
    monkeypatch.setattr(worker, "STALE_CONSUMER_MS", -1)
    worker.forget_stale_consumers(r, "me")
    names = {c["name"] for c in r.xinfo_consumers(worker.STREAM, worker.GROUP)}
    assert names == {"old-busy"}  # deleting it would drop its pending entry


@needs_redis
def test_reclaim_keeps_the_heartbeat_fresh(r, monkeypatch):
    monkeypatch.setattr(worker, "CLAIM_IDLE_MS", 0)
    worker.ensure_group(r)
    for i in range(3):
        enqueue(r, f"left{i}")
    # A worker died holding three jobs.
    r.xreadgroup(worker.GROUP, "dead", {worker.STREAM: ">"}, count=3)
    assert not worker.HEARTBEAT.exists()

    worker.reclaim(r, "alive", FakeDetector(), STORE)

    # Beaten during the pass itself (before the fix, only at the top of the loop).
    assert worker.HEARTBEAT.exists()
    assert r.xpending(worker.STREAM, worker.GROUP)["pending"] == 0
