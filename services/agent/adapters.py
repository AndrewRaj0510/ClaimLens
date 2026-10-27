"""The graph's connections to the outside world: photo storage, vision results
and the claims table. Each is a small class so tests can swap in a fake."""

import asyncio
import json
import os


class PhotoStore:
    """Reads claim photos from S3-compatible storage (SeaweedFS locally)."""

    # Same setting as the gateway and vision (one MAX_UPLOAD_MB in docker-compose.yml);
    # scripts/test_contracts.py checks the copies agree.
    MAX_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024)

    def __init__(self):
        import boto3

        self.bucket = os.getenv("S3_BUCKET", "claimlens")
        self.client = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT"),
            aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
            region_name=os.getenv("S3_REGION", "us-east-1"),
        )

    def _get(self, key):
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
        except self.client.exceptions.NoSuchKey as e:
            raise KeyError(key) from e
        with obj["Body"] as body:
            data = body.read(self.MAX_BYTES + 1)
        if len(data) > self.MAX_BYTES:
            raise ValueError("photo too large")
        return data

    async def get(self, key):
        return await asyncio.to_thread(self._get, key)  # boto3 is synchronous


class VisionResults:
    """Reads results of vision jobs the gateway queued (protocol: services/vision/worker.py).

    The gateway enqueues a job as soon as a photo is uploaded, so detection
    runs while the agent is still doing intake; usually the result is already
    there when damage_analysis asks for it."""

    def __init__(self, redis):
        self.redis = redis  # redis.asyncio.Redis, decode_responses=True

    async def result(self, job_id, timeout_s):
        stored = await self.redis.get(f"vision:result:{job_id}")
        if stored:
            return json.loads(stored)
        item = await self.redis.blpop(f"vision:done:{job_id}", timeout=timeout_s)
        if item:
            return json.loads(item[1])
        # The notification may have been taken by another waiter; the result key is authoritative.
        stored = await self.redis.get(f"vision:result:{job_id}")
        return json.loads(stored) if stored else None


class ClaimRecorder:
    """Writes the decided claim and its photo hashes to the claims tables, so the
    next claim's duplicate-photo check can find these photos. Idempotent: a
    retried final node inserts nothing twice."""

    def __init__(self, pool):
        self.pool = pool  # psycopg_pool.AsyncConnectionPool

    async def record(self, state):
        policy = state.get("policy")
        if not policy:
            return  # no policy, so no valid claims row (and nothing to match against)
        decision = state["decision"]
        status = {"approve": "approved", "edit": "approved", "reject": "denied"}[
            decision["action"]
        ]
        claim_id = f"CLM-{state['claim_id']}"
        async with self.pool.connection() as conn, conn.transaction():
            await conn.execute(
                """
                INSERT INTO claims (claim_id, policy_id, customer_id, vehicle_id, peril, description,
                                    damages, incident_date, reported_date, claimed_amount, deductible,
                                    status, paid_amount)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (claim_id) DO NOTHING
                """,
                (
                    claim_id,
                    policy["policy_id"],
                    policy["customer_id"],
                    policy["vehicle"]["vehicle_id"],
                    state["peril"],
                    state["description"],
                    json.dumps(
                        [
                            {"class": d["damage_class"], "severity": d["severity"]}
                            for d in state["damages"]
                        ]
                    ),
                    state["incident_date"],
                    state["reported_date"],
                    state.get("claimed_amount")
                    or (state.get("estimate") or {}).get("total", 0),
                    state["coverage"].get("deductible") or 0,
                    status,
                    decision["amount"] or 0,
                ),
            )
            for i, q in enumerate(state["quality"]):
                if q.get("phash"):
                    await conn.execute(
                        "INSERT INTO claim_photos VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                        (claim_id, i, phash_to_bigint(q["phash"])),
                    )


def phash_to_bigint(phash):
    value = int(phash, 16)
    return value - (1 << 64) if value >= 1 << 63 else value
