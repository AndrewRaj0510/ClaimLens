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


class GuardrailsUnavailable(RuntimeError):
    """The guardrails service couldn't give an answer. Callers fail closed:
    unscanned input never goes to an LLM."""


class GuardrailsRejected(GuardrailsUnavailable, ValueError):
    """Guardrails refused the input (4xx), e.g. a photo that isn't an image.
    Still "no answer" for fail-closed purposes; also a ValueError, so a photo
    check can treat it as an unreadable photo."""


class Guardrails:
    """Client for services/guardrails. Each call is retried once; after that
    the caller falls back (no LLM for that input) and flags the claim."""

    URL = os.getenv("GUARDRAILS_URL", "http://guardrails:8000")
    # OCR of a full photo takes a few seconds on CPU; allow for a queue.
    TIMEOUT_S = float(os.getenv("GUARDRAILS_TIMEOUT_S", "60"))

    def __init__(self, url=None, transport=None):
        import httpx

        self.httpx = httpx
        self.client = httpx.AsyncClient(
            base_url=url or self.URL, timeout=self.TIMEOUT_S, transport=transport
        )

    async def _post(self, path, **kwargs):
        last = None
        for _ in range(2):
            try:
                r = await self.client.post(path, **kwargs)
                if r.status_code < 500:
                    r.raise_for_status()
                    return r.json()
                last = f"HTTP {r.status_code}"
            except self.httpx.HTTPStatusError as e:  # 4xx: retrying won't help
                raise GuardrailsRejected(
                    f"{path}: HTTP {e.response.status_code}"
                ) from e
            except self.httpx.HTTPError as e:
                last = type(e).__name__
        raise GuardrailsUnavailable(f"{path}: {last}")

    async def scan_texts(self, texts, profile="untrusted"):
        """[{"flagged", "score", "rules"}] per text, in order. profile
        "document" is for our own policy wording (rules decide, see
        services/guardrails/text_scan.py)."""
        if not texts:
            return []
        body = {"texts": list(texts), "profile": profile}
        return (await self._post("/scan/text", json=body))["results"]

    async def inspect_photo(self, data, incident_date=None):
        """OCR + scan, redaction, forensics. `redacted_jpeg` is base64."""
        return await self._post(
            "/images/inspect",
            files={"file": ("photo", data, "application/octet-stream")},
            data={"incident_date": incident_date} if incident_date else {},
        )

    async def aclose(self):
        await self.client.aclose()


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
                ON CONFLICT DO NOTHING
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


class ClaimIndex:
    """One row per claim with its latest status and a short summary, for the
    adjuster's queue and a customer's claim list. The checkpoints hold the
    full state, but listing claims from them would mean loading every thread.

    The agent owns this table (like LangGraph owns its checkpoint tables) and
    creates it on startup. It's a cache of the checkpoints: if a row is stale,
    the checkpoint is the truth."""

    def __init__(self, pool):
        self.pool = pool  # psycopg_pool.AsyncConnectionPool, dict rows

    async def setup(self):
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_claims (
                    claim_id     text PRIMARY KEY,
                    customer_id  text,
                    policy_id    text NOT NULL,
                    status       text NOT NULL,
                    summary      jsonb NOT NULL DEFAULT '{}',
                    created_at   timestamptz NOT NULL DEFAULT now(),
                    updated_at   timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS agent_claims_customer_idx"
                " ON agent_claims (customer_id, updated_at DESC)"
            )

    async def add(self, claim_id, customer_id, policy_id):
        async with self.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO agent_claims (claim_id, customer_id, policy_id, status)"
                " VALUES (%s, %s, %s, 'processing') ON CONFLICT DO NOTHING",
                (claim_id, customer_id, policy_id),
            )

    async def update(self, claim_id, status, summary=None):
        async with self.pool.connection() as conn:
            await conn.execute(
                "UPDATE agent_claims SET status = %s,"
                " summary = COALESCE(%s, summary), updated_at = now()"
                " WHERE claim_id = %s",
                (
                    status,
                    json.dumps(summary) if summary is not None else None,
                    claim_id,
                ),
            )

    async def list(self, customer_id=None, status=None, limit=50):
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                SELECT claim_id, customer_id, policy_id, status, summary,
                       created_at, updated_at
                FROM agent_claims
                WHERE (%(customer)s::text IS NULL OR customer_id = %(customer)s)
                  AND (%(status)s::text IS NULL OR status = %(status)s)
                ORDER BY updated_at DESC
                LIMIT %(limit)s
                """,
                {"customer": customer_id, "status": status, "limit": limit},
            )
            rows = await cur.fetchall()
        return [
            {
                **r,
                "created_at": r["created_at"].isoformat(timespec="seconds"),
                "updated_at": r["updated_at"].isoformat(timespec="seconds"),
            }
            for r in rows
        ]
