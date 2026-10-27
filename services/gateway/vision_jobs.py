"""Producer side of the vision job queue.

The protocol (stream, result and notify keys, result JSON) is documented in
services/vision/worker.py, the consumer side.

The gateway stores the image first and only then enqueues the job, so a worker
never sees a job whose image isn't there yet.
"""

import asyncio
import json
import uuid

STREAM = "vision:jobs"
# Cap the stream's memory: old entries are trimmed (approximately, which is cheaper).
STREAM_MAXLEN = 10_000
JOB_TTL_S = 24 * 3600


class VisionJobs:
    def __init__(self, redis, store):
        self.redis = redis  # redis.asyncio.Redis with decode_responses=True
        self.store = store

    async def submit(self, data, content_type, key=None):
        """Store the image (at `key`, default uploads/<job_id>) and queue a job for it."""
        job_id = uuid.uuid4().hex
        key = key or f"uploads/{job_id}"
        await asyncio.to_thread(self.store.put, key, data, content_type)
        await self.redis.set(f"vision:job:{job_id}", "pending", ex=JOB_TTL_S)
        await self.redis.xadd(
            STREAM,
            {"job_id": job_id, "image_key": key},
            maxlen=STREAM_MAXLEN,
            approximate=True,
        )
        return job_id

    async def wait(self, job_id, timeout_s):
        """Block until the worker publishes the result, or return None on timeout."""
        item = await self.redis.blpop(f"vision:done:{job_id}", timeout=timeout_s)
        return json.loads(item[1]) if item else None

    async def status(self, job_id):
        """The result if done, {"status": "pending"} if queued, None if unknown."""
        result = await self.redis.get(f"vision:result:{job_id}")
        if result:
            return json.loads(result)
        if await self.redis.exists(f"vision:job:{job_id}"):
            return {"status": "pending"}
        return None
