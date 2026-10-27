"""Agent service: runs the claim graph and exposes its state.

    POST /claims                  start a claim (from the gateway, after photos are stored and queued)
    GET  /claims/{id}             status, the pending question (retake or approval), state, trace
    POST /claims/{id}/photos      answer a retake request
    POST /claims/{id}/decision    the adjuster's decision (the only way a claim gets decided)

Runs are started in the background and survive in the Postgres checkpointer:
the state after every node is saved, so a paused claim waits for days at no
cost and a restarted service picks up where it left off.

Internal only: the gateway is the public entry point and will own
authentication (customer vs adjuster) in Phase 3.
"""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from typing import Annotated

from fastapi import FastAPI, HTTPException, Path
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator

from graph import Decision, Deps, build_graph

DATABASE_URL = os.getenv("DATABASE_URL", "")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

log = logging.getLogger("agent")

ClaimId = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")]
ObjectKey = Annotated[str, Field(pattern=r"^claims/[0-9a-f]{32}/[0-9a-z_-]{1,64}$")]
JobId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
WAITING_FOR = {"approval": "a decision", "retake": "photos"}


class Photo(BaseModel):
    key: ObjectKey
    job_id: JobId


class NewClaim(BaseModel):
    claim_id: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")] | None = None
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

    from adapters import ClaimRecorder, PhotoStore, VisionResults
    from llm import OllamaLLM
    from tools import McpTools

    # The checkpointer needs autocommit and dict rows (langgraph-checkpoint-postgres docs).
    async with AsyncConnectionPool(
        DATABASE_URL,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
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
        )
        app.state.graph = build_graph(deps, checkpointer)
        yield
        await r.aclose()


def create_app(lifespan=lifespan):
    app = FastAPI(title="ClaimLens Agent", version="0.1.0", lifespan=lifespan)
    app.state.runs = {}  # claim_id -> asyncio.Task of the current run
    app.state.failures = {}  # claim_id -> error message of the last failed run

    def config(claim_id):
        return {"configurable": {"thread_id": claim_id}}

    async def run(claim_id, graph_input):
        try:
            await app.state.graph.ainvoke(graph_input, config(claim_id))
            app.state.failures.pop(claim_id, None)
        except (
            Exception
        ) as e:  # recorded so GET shows "failed"; the checkpoint allows a retry
            log.exception("claim %s failed", claim_id)
            app.state.failures[claim_id] = f"{type(e).__name__}: {e}"[:500]

    def start(claim_id, graph_input):
        current = app.state.runs.get(claim_id)
        if current and not current.done():
            raise HTTPException(
                status_code=409, detail="Claim is already being processed"
            )
        app.state.runs[claim_id] = asyncio.create_task(run(claim_id, graph_input))

    async def snapshot(claim_id):
        snap = await app.state.graph.aget_state(config(claim_id))
        # A run just started may not have saved its first checkpoint yet; it
        # still exists, it's just "processing".
        if not snap.values and claim_id not in app.state.runs:
            raise HTTPException(status_code=404, detail="Unknown claim")
        pending = [i.value for t in snap.tasks for i in t.interrupts]
        return snap, (pending[0] if pending else None)

    def status_of(claim_id, snap, question):
        task = app.state.runs.get(claim_id)
        if task and not task.done():
            return "processing"
        if claim_id in app.state.failures:
            return "failed"
        if question:
            return {"retake": "awaiting_retake", "approval": "awaiting_approval"}[
                question["type"]
            ]
        if not snap.next:
            return "completed"
        # Steps left, but nothing running here and no failure recorded: the run
        # was cut off, typically by a restart (runs live in this process's memory).
        # The checkpoint is intact, so /retry continues it.
        return "interrupted"

    def error_of(claim_id, status):
        if status == "interrupted":
            return "The run stopped before finishing (service restart); POST /retry continues it."
        return app.state.failures.get(claim_id)

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
        start(claim_id, state)
        return {"claim_id": claim_id, "status": "processing"}

    @app.get("/claims/{claim_id}")
    async def get_claim(claim_id: ClaimId):
        snap, question = await snapshot(claim_id)
        status = status_of(claim_id, snap, question)
        return {
            "claim_id": claim_id,
            "status": status,
            "question": question,
            "error": error_of(claim_id, status),
            "state": {k: v for k, v in snap.values.items() if k != "trace"},
            "trace": snap.values.get("trace", []),
        }

    async def resume(claim_id, expected, value):
        snap, question = await snapshot(claim_id)
        if status_of(claim_id, snap, question) != f"awaiting_{expected}":
            raise HTTPException(
                status_code=409,
                detail=f"Claim isn't waiting for {WAITING_FOR[expected]}",
            )
        start(claim_id, Command(resume=value))
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
        if not snap.values or status_of(claim_id, snap, question) not in (
            "failed",
            "interrupted",
        ):
            raise HTTPException(status_code=409, detail="Claim hasn't failed")
        start(claim_id, None)  # None = continue from the last checkpoint
        return {"claim_id": claim_id, "status": "processing"}

    return app


app = create_app()
