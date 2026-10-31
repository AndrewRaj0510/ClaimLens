"""Vision queue worker: runs damage detection on jobs from a Redis stream.

Protocol (the gateway is the producer, see services/gateway/vision_jobs.py):

    stream  vision:jobs              entries {job_id, image_key}; image in object storage
    group   vision-workers           each entry goes to one worker in the group
    result  vision:result:<job_id>   JSON, kept RESULT_TTL_S for GET /jobs/<id>
    notify  vision:done:<job_id>     list with the same JSON, so a waiter can BLPOP
    dead    vision:dead              entries that failed MAX_DELIVERIES times

Result JSON: {"status": "done", "result": {...}} or
             {"status": "error", "code": 400|404|413|503|500, "detail": "..."}

Delivery is at-least-once. An entry is only ACKed after its result is written,
so if a worker dies mid-job the entry stays pending and another worker claims
it after CLAIM_IDLE_MS. Entries that keep failing (a "poison" image that crashes
the model) go to the dead-letter stream with an error result instead of looping
forever. Running the same job twice is harmless: the result is the same.

Scaling: run N copies (one consumer each). In Kubernetes, KEDA's redis-streams
scaler adds workers when the group's backlog grows (k8s/charts/vision-worker).

Metrics for Prometheus on METRICS_PORT (9100): jobs by outcome, time per job,
retries and dead letters. The backlog itself is KEDA's metric, so it's
measured once, not by every worker.
"""

import json
import logging
import os
import signal
import socket
import time
from pathlib import Path

from prometheus_client import Counter, Histogram, start_http_server

from analysis import ImageTooLarge, InvalidImage, analyse, max_bytes

STREAM = "vision:jobs"
GROUP = "vision-workers"
DEAD = "vision:dead"
RESULT_TTL_S = int(os.getenv("RESULT_TTL_S", str(24 * 3600)))
NOTIFY_TTL_S = 3600
CLAIM_IDLE_MS = int(os.getenv("CLAIM_IDLE_MS", "60000"))
MAX_DELIVERIES = 3
BLOCK_MS = 5000
# Each container registers its hostname as a consumer, and Redis keeps them
# forever; forget ones that have been idle this long with nothing pending.
STALE_CONSUMER_MS = 3600 * 1000
HEARTBEAT = Path(os.getenv("HEARTBEAT_FILE", "/tmp/worker-heartbeat"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "9100"))

log = logging.getLogger("vision.worker")

# outcome: "done", or the error result's code ("400", "503", ...)
JOBS = Counter("vision_jobs_total", "Jobs finished, by outcome", ["outcome"])
JOB_SECONDS = Histogram(
    "vision_job_seconds",
    "Time per job: fetching the image and running the model",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10),
)
RETRIES = Counter("vision_job_failures_total", "Jobs that raised and will be retried")
DEAD_LETTERS = Counter("vision_dead_letters_total", "Jobs dead-lettered")


def error(code, detail):
    return {"status": "error", "code": code, "detail": detail}


def run_job(fields, detector, store):
    """Turn one job into its result dict. Expected failures become error results;
    anything else raises, so the entry stays pending and is retried."""
    from storage import ObjectNotFound

    if detector is None:
        return error(503, "Model not loaded")
    try:
        data = store.get(fields["image_key"], max_bytes())
    except ObjectNotFound:
        return error(404, "Image not found")
    try:
        return {"status": "done", "result": analyse(detector, data).model_dump()}
    except ImageTooLarge as e:
        return error(413, str(e))
    except InvalidImage as e:
        return error(400, str(e))


def publish(r, job_id, result):
    """Store the result, wake any waiter, in one round trip (MULTI/EXEC)."""
    payload = json.dumps(result)
    pipe = r.pipeline()
    pipe.set(f"vision:result:{job_id}", payload, ex=RESULT_TTL_S)
    pipe.rpush(f"vision:done:{job_id}", payload)
    pipe.expire(f"vision:done:{job_id}", NOTIFY_TTL_S)
    pipe.execute()


def handle(r, entry_id, fields, detector, store):
    with JOB_SECONDS.time():
        result = run_job(fields, detector, store)
    publish(r, fields["job_id"], result)
    r.xack(STREAM, GROUP, entry_id)
    JOBS.labels("done" if result["status"] == "done" else str(result["code"])).inc()
    log.info("job %s -> %s", fields["job_id"], result["status"])


def ensure_group(r):
    from redis.exceptions import ResponseError

    try:
        # "0" + MKSTREAM: create the stream if needed and read it from the start.
        r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def reclaim(r, consumer, detector, store):
    """Take over entries another (dead or stuck) worker left pending."""
    pending = r.xpending_range(
        STREAM, GROUP, min="-", max="+", count=50, idle=CLAIM_IDLE_MS
    )
    for p in pending:
        entry_id = p["message_id"]
        claimed = r.xclaim(STREAM, GROUP, consumer, CLAIM_IDLE_MS, [entry_id])
        if not claimed or claimed[0][1] is None:
            r.xack(STREAM, GROUP, entry_id)  # entry was trimmed away; nothing to do
            continue
        fields = claimed[0][1]
        if p["times_delivered"] >= MAX_DELIVERIES:
            log.error(
                "job %s failed %d times; dead-lettering",
                fields.get("job_id"),
                p["times_delivered"],
            )
            publish(r, fields["job_id"], error(500, "Detection failed repeatedly"))
            r.xadd(
                DEAD, {**fields, "entry_id": entry_id}, maxlen=10000, approximate=True
            )
            r.xack(STREAM, GROUP, entry_id)
            DEAD_LETTERS.inc()
            continue
        safe_handle(r, entry_id, fields, detector, store)
    forget_stale_consumers(r, consumer)


def forget_stale_consumers(r, consumer):
    for c in r.xinfo_consumers(STREAM, GROUP):
        if (
            c["name"] != consumer
            and c["pending"] == 0
            and c["idle"] > STALE_CONSUMER_MS
        ):
            r.xgroup_delconsumer(STREAM, GROUP, c["name"])


def safe_handle(r, entry_id, fields, detector, store):
    try:
        handle(r, entry_id, fields, detector, store)
    except Exception:
        # Leave it pending: reclaim() retries it, and dead-letters it after MAX_DELIVERIES.
        log.exception("job %s failed; will be retried", fields.get("job_id"))
        RETRIES.inc()
    # Beat after every job, not only once per loop: a reclaim pass can run dozens
    # of jobs back to back, and a stale heartbeat gets the worker killed mid-way.
    HEARTBEAT.touch()


def connect(url):
    import redis

    # XREADGROUP blocks for BLOCK_MS, so the socket timeout must be longer.
    # redis-py 8 defaults it to 5 s, the same as the block time; that crashed
    # every idle worker the first time the stack ran.
    return redis.Redis.from_url(
        url, decode_responses=True, socket_timeout=BLOCK_MS / 1000 + 10
    )


def poll(r, consumer, detector, store, state, reclaim_every_s):
    if state.get("needs_group", True):
        ensure_group(r)
        state["needs_group"] = False
    if time.monotonic() - state.get("last_reclaim", 0.0) > reclaim_every_s:
        reclaim(r, consumer, detector, store)
        state["last_reclaim"] = time.monotonic()
    batches = r.xreadgroup(GROUP, consumer, {STREAM: ">"}, count=1, block=BLOCK_MS)
    for _stream, entries in batches or []:
        for entry_id, fields in entries:
            safe_handle(r, entry_id, fields, detector, store)


def run(r, detector, store, consumer, should_stop, reclaim_every_s=30, backoff_s=1.0):
    from redis.exceptions import RedisError

    state = {}
    while not should_stop():
        HEARTBEAT.touch()
        try:
            poll(r, consumer, detector, store, state, reclaim_every_s)
        except RedisError:
            # Redis restarting or a network blip: back off and carry on rather
            # than exit. The group is re-created in case Redis lost it.
            log.exception("redis error; retrying in %.0f s", backoff_s)
            state["needs_group"] = True
            time.sleep(backoff_s)


def main():
    from detector import Detector
    from storage import S3Store

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    model_path = Path(os.getenv("MODEL_PATH", "/models/best.onnx"))
    # Like the HTTP API, start without weights; jobs get a 503 result until the model exists.
    detector = Detector(str(model_path)) if model_path.exists() else None
    if detector is None:
        log.warning("model not found at %s; jobs will get a 503 result", model_path)

    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True  # finish the current job, then exit (Kubernetes sends SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    start_http_server(METRICS_PORT)
    r = connect(os.environ["REDIS_URL"])
    consumer = os.getenv("CONSUMER_NAME", socket.gethostname())
    log.info("worker %s listening on %s", consumer, STREAM)
    run(r, detector, S3Store(), consumer, lambda: stopping)


if __name__ == "__main__":
    main()
