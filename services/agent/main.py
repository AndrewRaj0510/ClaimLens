"""Agent service: runs the claim graph and exposes its state.

    POST /claims                  start a claim (from the gateway, after photos are stored and queued)
    GET  /claims                  claim list from the index (?customer_id=, ?status=)
    GET  /claims/{id}             status, the pending question (retake or approval), state, trace
    POST /claims/{id}/photos      answer a retake request
    POST /claims/{id}/decision    the adjuster's decision (the only way a claim gets decided)

Runs are started in the background and survive in the Postgres checkpointer:
the state after every node is saved, so a paused claim waits for days at no
cost and a restarted service picks up where it left off. Which claims are
running right now is in Redis (runstate.py), so any replica can answer for
any claim and the service scales horizontally.

Internal only: the gateway is the public entry point and owns authentication.
It decides who may see or act on a claim; this service trusts its caller.

With Langfuse keys set, every run is traced (tracing.py): one trace per claim,
a span per run, per node and per LLM call.
"""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from typing import Annotated

from fastapi import FastAPI, HTTPException, Path, Query
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator

import http_metrics
import runstate
import tracing
from graph import Decision, Deps, build_graph

DATABASE_URL = os.getenv("DATABASE_URL", "")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# On shutdown (a rollout or a scale-down), how long this replica's runs get to
# finish before they're cut off. Kubernetes' terminationGracePeriodSeconds
# must be longer (k8s/charts/agent/values.yaml).
DRAIN_S = float(os.getenv("DRAIN_S", "90"))

log = logging.getLogger("agent")

ClaimId = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")]
CustomerId = Annotated[str, Field(pattern=r"^C\d{5}$")]
STATUSES = (
    "processing",
    "awaiting_retake",
    "awaiting_approval",
    "completed",
    "failed",
    "interrupted",
)
ObjectKey = Annotated[str, Field(pattern=r"^claims/[0-9a-f]{32}/[0-9a-z_-]{1,64}$")]
JobId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
WAITING_FOR = {"approval": "a decision", "retake": "photos"}


class Photo(BaseModel):
    key: ObjectKey
    job_id: JobId


class NewClaim(BaseModel):
    claim_id: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")] | None = None
    # Who filed it (the gateway checks they own the policy). None for claims
    # filed by internal tools, which then belong to no customer.
    customer_id: CustomerId | None = None
    policy_id: Annotated[str, Field(pattern=r"^P\d{5}$")]
    incident_date: date
    description: Annotated[str, Field(min_length=10, max_length=4000)]
    claimed_amount: Annotated[float, Field(gt=0, le=1_000_000)] | None = None
    photos: Annotated[list[Photo], Field(min_length=1, max_length=10)]

    @field_validator("incident_date")
    @classmethod
    def not_in_the_future(cls, value):
        # Server date (UTC). US customers are behind UTC, so their "today" is
        # never ahead of it.
        if value > datetime.now(UTC).date():
            raise ValueError("Incident date can't be in the future")
        return value


class Retake(BaseModel):
    photos: Annotated[list[Photo], Field(min_length=1, max_length=10)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    import redis.asyncio as redis
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    from adapters import (
        ClaimIndex,
        ClaimRecorder,
        Guardrails,
        PhotoStore,
        VisionResults,
    )
    from llm import VLM_MODEL, VLM_THINK, OllamaLLM
    from tools import McpTools

    # The checkpointer needs autocommit and dict rows (langgraph-checkpoint-postgres docs).
    # check: see services/gateway/main.py (dead connections after a DB restart).
    async with AsyncConnectionPool(
        DATABASE_URL,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
        check=AsyncConnectionPool.check_connection,
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()  # creates its tables on first run
        r = redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=200)
        deps = Deps(
            llm=OllamaLLM(),
            tools=McpTools(),
            photos=PhotoStore(),
            vision=VisionResults(r),
            recorder=ClaimRecorder(pool),
            guardrails=Guardrails(),
            vlm=OllamaLLM(model=VLM_MODEL, think=VLM_THINK),
        )
        app.state.graph = build_graph(deps, checkpointer)
        app.state.index = ClaimIndex(pool)
        app.state.runstate = runstate.RunState(r)
        await app.state.index.setup()
        yield
        await r.aclose()
        await deps.guardrails.aclose()
        tracing.flush()


def summary_of(values):
    """The few fields the claim lists show, so they don't load full state."""
    rec, fraud, payable = (
        values.get("recommendation"),
        values.get("fraud"),
        values.get("payable"),
    )
    decision = values.get("decision")
    return {
        "incident_date": values.get("incident_date"),
        "damages": [
            f"{d['damage_class']} ({d['severity']})" for d in values.get("damages", [])
        ],
        "recommendation": rec["action"] if rec else None,
        "fraud_level": fraud["level"] if fraud else None,
        "fraud_score": fraud["score"] if fraud else None,
        "payable_expected": payable["expected"] if payable else None,
        # Guardrails findings, so the queue can show which claims carry them.
        "security_findings": len(values.get("security", [])),
        "decision": {k: decision[k] for k in ("action", "amount")}
        if decision
        else None,
    }


async def drain(runs, timeout_s):
    """Let this replica's runs finish, then cancel the rest. A cancelled run
    releases its lease, so its claim shows as interrupted (and retryable) at
    once instead of after the lease TTL."""
    pending = [t for t in runs.values() if not t.done()]
    if not pending:
        return
    log.info("draining %d run(s) for up to %.0fs", len(pending), timeout_s)
    _, pending = await asyncio.wait(pending, timeout=timeout_s)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


def create_app(lifespan=lifespan, drain_s=DRAIN_S):
    @asynccontextmanager
    async def serve(app):
        async with lifespan(app):
            yield
            await drain(app.state.runs, drain_s)  # before Redis and Postgres close

    app = FastAPI(title="ClaimLens Agent", version="0.1.0", lifespan=serve)
    http_metrics.instrument(app)
    # claim_id -> asyncio.Task, for runs started by this replica only. Whether
    # a claim is running anywhere is its lease in Redis (app.state.runstate).
    app.state.runs = {}

    def config(claim_id):
        return {"configurable": {"thread_id": claim_id}}

    async def keep_lease(claim_id, token, run_task):
        rs = app.state.runstate
        while True:
            await asyncio.sleep(runstate.LEASE_S / 3)
            try:
                if not await rs.renew(claim_id, token):
                    # Only after the lease expired and another replica started
                    # the claim: two runs on one checkpoint must not go on.
                    log.error("claim %s: lease lost, stopping this run", claim_id)
                    run_task.cancel()
                    return
            except Exception:  # Redis blip: keep going, the TTL gives slack
                log.warning("claim %s: lease renewal failed", claim_id, exc_info=True)

    async def run(claim_id, graph_input, kind, token):
        rs = app.state.runstate
        heartbeat = asyncio.create_task(
            keep_lease(claim_id, token, asyncio.current_task())
        )
        try:
            try:
                owner = None
                if tracing.ENABLED:
                    owner = await rs.owner(claim_id)
                    if owner is None:  # a resumed run: ask the checkpoint
                        snap = await app.state.graph.aget_state(config(claim_id))
                        owner = snap.values.get("customer_id")
                with tracing.claim_run(claim_id, kind, owner):
                    await app.state.graph.ainvoke(graph_input, config(claim_id))
                await rs.clear_failure(claim_id)
            except Exception as e:  # GET shows "failed"; the checkpoint allows a retry
                log.exception("claim %s failed", claim_id)
                try:
                    await rs.set_failure(claim_id, f"{type(e).__name__}: {e}"[:500])
                except Exception:
                    log.exception("claim %s: failure not recorded", claim_id)
            await sync_index(claim_id)
        finally:
            heartbeat.cancel()
            try:
                await rs.release(claim_id, token)
            except Exception:  # it expires on its own
                log.exception("claim %s: lease not released", claim_id)

    async def sync_index(claim_id):
        """Copy the claim's status and summary into the index once a run stops.
        A failure here is logged, not raised: the checkpoint is the truth, and
        the claim itself must not fail because the list view is behind."""
        try:
            snap, question = await snapshot(claim_id)
            if snap.values:  # the checkpoint knows the owner from here on
                await app.state.runstate.clear_owner(claim_id)
            status = await status_of(claim_id, snap, question, running=False)
            await app.state.index.update(claim_id, status, summary_of(snap.values))
        except Exception:
            log.exception("claim %s: index update failed", claim_id)

    async def start(claim_id, graph_input, kind):
        token = await app.state.runstate.acquire(claim_id)
        if token is None:  # running here or on another replica
            raise HTTPException(
                status_code=409, detail="Claim is already being processed"
            )
        task = asyncio.create_task(run(claim_id, graph_input, kind, token))
        app.state.runs[claim_id] = task

        def forget(t):
            if app.state.runs.get(claim_id) is t:
                del app.state.runs[claim_id]

        task.add_done_callback(forget)

    async def snapshot(claim_id):
        snap = await app.state.graph.aget_state(config(claim_id))
        # A run just started may not have saved its first checkpoint yet; it
        # still exists, it's just "processing".
        if not snap.values and not await is_running(claim_id):
            raise HTTPException(status_code=404, detail="Unknown claim")
        pending = [i.value for t in snap.tasks for i in t.interrupts]
        return snap, (pending[0] if pending else None)

    async def is_running(claim_id):
        return claim_id in await app.state.runstate.running([claim_id])

    async def status_of(claim_id, snap, question, running=None):
        # A run reporting on itself as it finishes passes running=False: it
        # still holds its lease.
        if running is None:
            running = await is_running(claim_id)
        if running:
            return "processing"
        if await app.state.runstate.failure(claim_id) is not None:
            return "failed"
        if question:
            return {"retake": "awaiting_retake", "approval": "awaiting_approval"}[
                question["type"]
            ]
        if not snap.next:
            return "completed"
        # Steps left, no lease and no failure recorded: the run was cut off,
        # by a pod that died or a shutdown that outlasted the drain. The
        # checkpoint is intact, so /retry continues it.
        return "interrupted"

    async def error_of(claim_id, status):
        if status == "interrupted":
            return "The run stopped before finishing (service restart); POST /retry continues it."
        return await app.state.runstate.failure(claim_id)

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "agent"}

    @app.post("/claims", status_code=202)
    async def create_claim(claim: NewClaim):
        claim_id = claim.claim_id or uuid.uuid4().hex
        for photo in claim.photos:
            # Photos must belong to this claim's folder: a caller can't point the
            # agent at someone else's uploads.
            if not photo.key.startswith(f"claims/{claim_id}/"):
                raise HTTPException(
                    status_code=422, detail="Photo key outside this claim"
                )
        snap = await app.state.graph.aget_state(config(claim_id))
        if snap.values:
            raise HTTPException(status_code=409, detail="Claim already exists")
        state = {
            "claim_id": claim_id,
            "policy_id": claim.policy_id,
            "incident_date": claim.incident_date.isoformat(),
            "reported_date": datetime.now(UTC).date().isoformat(),
            "description": claim.description,
            "claimed_amount": claim.claimed_amount,
            "photos": [p.model_dump() for p in claim.photos],
            "retakes": 0,
            "trace": [],
        }
        if claim.customer_id:
            state["customer_id"] = claim.customer_id
        # Indexed before the run starts, so the run's own update finds the row.
        await app.state.index.add(claim_id, claim.customer_id, claim.policy_id)
        if claim.customer_id:
            await app.state.runstate.set_owner(claim_id, claim.customer_id)
        await start(claim_id, state, "start")
        return {"claim_id": claim_id, "status": "processing"}

    @app.get("/claims")
    async def list_claims(
        customer_id: Annotated[str | None, Query(pattern=r"^C\d{5}$")] = None,
        status: Annotated[
            str | None, Query(pattern="^(" + "|".join(STATUSES) + ")$")
        ] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ):
        rows = await app.state.index.list(customer_id, status, limit)
        running = await app.state.runstate.running(r["claim_id"] for r in rows)
        for row in rows:
            if row["claim_id"] in running:
                row["status"] = "processing"
            elif row["status"] == "processing":
                # Not running anywhere, so the row is stale (e.g. a pod died
                # mid-run): ask the checkpoint.
                snap, question = await snapshot(row["claim_id"])
                row["status"] = await status_of(
                    row["claim_id"], snap, question, running=False
                )
        return {"claims": rows}

    @app.get("/claims/{claim_id}")
    async def get_claim(claim_id: ClaimId):
        snap, question = await snapshot(claim_id)
        status = await status_of(claim_id, snap, question)
        owner = snap.values.get("customer_id")
        if not snap.values:  # before the first checkpoint
            owner = await app.state.runstate.owner(claim_id)
        return {
            "claim_id": claim_id,
            # Top level, and known even before the first checkpoint: the
            # gateway authorizes on it.
            "customer_id": owner,
            "status": status,
            "question": question,
            "error": await error_of(claim_id, status),
            "state": {k: v for k, v in snap.values.items() if k != "trace"},
            "trace": snap.values.get("trace", []),
        }

    async def resume(claim_id, expected, value):
        snap, question = await snapshot(claim_id)
        if await status_of(claim_id, snap, question) != f"awaiting_{expected}":
            raise HTTPException(
                status_code=409,
                detail=f"Claim isn't waiting for {WAITING_FOR[expected]}",
            )
        await start(claim_id, Command(resume=value), expected)
        await app.state.index.update(claim_id, "processing")
        return {"claim_id": claim_id, "status": "processing"}

    @app.post("/claims/{claim_id}/photos", status_code=202)
    async def retake(claim_id: ClaimId, body: Retake):
        for photo in body.photos:
            if not photo.key.startswith(f"claims/{claim_id}/"):
                raise HTTPException(
                    status_code=422, detail="Photo key outside this claim"
                )
        return await resume(claim_id, "retake", body.model_dump())

    @app.post("/claims/{claim_id}/decision", status_code=202)
    async def decide(claim_id: ClaimId, decision: Decision):
        if decision.action == "edit" and decision.amount is None:
            raise HTTPException(
                status_code=422, detail="An edited decision needs an amount"
            )
        return await resume(claim_id, "approval", decision.model_dump())

    @app.post("/claims/{claim_id}/retry", status_code=202)
    async def retry(claim_id: ClaimId):
        snap, question = await snapshot(claim_id)
        if not snap.values or await status_of(claim_id, snap, question) not in (
            "failed",
            "interrupted",
        ):
            raise HTTPException(status_code=409, detail="Claim hasn't failed")
        await start(claim_id, None, "retry")  # None = continue from the last checkpoint
        await app.state.index.update(claim_id, "processing")
        return {"claim_id": claim_id, "status": "processing"}

    return app


app = create_app()
