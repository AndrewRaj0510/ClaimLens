"""Claim endpoints: the public face of the agent service.

    GET  /me/policies               customer: their policies, for the claim form
    POST /claims                    customer: form fields + photos, on their own policy
    GET  /claims                    customer: their claims; adjuster: all (?status=)
    GET  /claims/{id}               customer: their view; adjuster: everything
    GET  /claims/{id}/photos/{n}    the claim's photo n, for whoever may see the claim
    POST /claims/{id}/photos        customer: retake photos
    POST /claims/{id}/decision      adjuster only
    POST /claims/{id}/retry         adjuster only: continue a failed or interrupted run

Photos are stored and their vision jobs queued here, before the agent starts,
so detection runs in parallel with the agent's first (LLM) step.

Authorization happens here; the agent trusts its caller. A customer asking
for someone else's claim gets 404, the same as for a claim that doesn't
exist, so claim IDs can't be probed. Customers get a reduced view: the fraud
score, signals, recommendation and the agent's reasoning are for adjusters.
"""

import asyncio
import os
import uuid
from datetime import UTC, date, datetime
from typing import Annotated, Any, Literal

import httpx
from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    Response,
    UploadFile,
)

from auth import (
    Adjuster,
    CurrentUser,
    Customer,
    RateLimiter,
    UserStore,
    rate_limiter,
    user_store,
)
from common import ALLOWED_TYPES, agent_client, photo_store, read_upload, vision_jobs
from storage import S3Store
from vision_jobs import VisionJobs

AGENT_URL = os.getenv("AGENT_URL", "http://agent:8000")
MAX_PHOTOS = 10
# Claims (and retakes) a customer may submit per hour: each one costs storage,
# vision jobs and LLM calls.
SUBMISSIONS_PER_HOUR = 10

router = APIRouter()
ClaimId = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")]
Agent = Annotated[httpx.AsyncClient, Depends(agent_client)]
Jobs = Annotated[VisionJobs, Depends(vision_jobs)]
Limiter = Annotated[RateLimiter, Depends(rate_limiter)]


async def store_photos(jobs, claim_id, photos):
    if not 1 <= len(photos) <= MAX_PHOTOS:
        raise HTTPException(status_code=422, detail=f"Send 1 to {MAX_PHOTOS} photos")
    files = [await read_upload(p) for p in photos]  # validate all before storing any
    stored = []
    try:
        for data, content_type in files:
            # Random object names: the customer's filenames are untrusted and never used.
            key = f"claims/{claim_id}/{uuid.uuid4().hex[:12]}"
            job_id = await jobs.submit(data, content_type, key=key)
            stored.append({"key": key, "job_id": job_id})
    except Exception as e:
        raise HTTPException(status_code=503, detail="Photo storage unavailable") from e
    return stored


async def call_agent(agent, method, path, **kwargs):
    try:
        r = await agent.request(method, path, **kwargs)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail="Agent service unavailable") from e
    if 400 <= r.status_code < 500:
        raise HTTPException(status_code=r.status_code, detail=r.json().get("detail"))
    if r.status_code >= 500:
        raise HTTPException(status_code=502, detail="Agent service error")
    return r.json()


async def limit_submissions(limiter, user):
    key = f"submit:{user.username}"
    if await limiter.hit(key, 3600) > SUBMISSIONS_PER_HOUR:
        raise HTTPException(
            status_code=429,
            detail="Too many submissions. Try again later.",
            headers={"Retry-After": str(await limiter.retry_after(key))},
        )


async def claim_for(agent, user, claim_id):
    """The full claim, if this user may see it; 404 otherwise."""
    claim = await call_agent(agent, "GET", f"/claims/{claim_id}")
    # customer_id is top level: it's set even before the agent's first
    # checkpoint, when the state is still empty.
    if user.role == "customer" and claim.get("customer_id") != user.customer_id:
        raise HTTPException(status_code=404, detail="Unknown claim")
    return claim


def customer_view(claim):
    """What the customer who filed the claim sees. Built field by field (an
    allowlist), so anything added to the agent's state later stays internal
    until someone decides customers should see it."""
    state, question, status = claim["state"], claim["question"], claim["status"]
    decision = state.get("decision")
    done = status == "completed"
    return {
        "claim_id": claim["claim_id"],
        "status": status,
        "policy_id": state.get("policy_id"),
        "incident_date": state.get("incident_date"),
        "reported_date": state.get("reported_date"),
        "description": state.get("description"),
        "claimed_amount": state.get("claimed_amount"),
        "photo_count": len(state.get("photos", [])),
        # Only retake requests: the approval question carries the fraud score.
        "question": question if question and question["type"] == "retake" else None,
        "outcome": {k: decision[k] for k in ("action", "amount")}
        if done and decision
        else None,
        "message": (state.get("report") or {}).get("customer_message")
        if done
        else None,
    }


def customer_row(row):
    summary = row.get("summary") or {}
    return {
        "claim_id": row["claim_id"],
        "policy_id": row["policy_id"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "incident_date": summary.get("incident_date"),
        "damages": summary.get("damages", []),
        "decision": summary.get("decision"),
    }


@router.get("/me/policies")
async def my_policies(user: Customer, users: Annotated[UserStore, Depends(user_store)]):
    return {"policies": await users.policies(user.customer_id)}


@router.post("/claims", status_code=202)
async def create_claim(
    user: Customer,
    agent: Agent,
    jobs: Jobs,
    limiter: Limiter,
    users: Annotated[UserStore, Depends(user_store)],
    policy_id: Annotated[str, Form(pattern=r"^P\d{5}$")],
    incident_date: Annotated[date, Form()],
    description: Annotated[str, Form(min_length=10, max_length=4000)],
    photos: Annotated[list[UploadFile], File()],
    claimed_amount: Annotated[float | None, Form(gt=0, le=1_000_000)] = None,
):
    # The agent validates this too, but checking first means nothing is stored
    # or queued for a claim that's going to be rejected.
    if incident_date > datetime.now(UTC).date():
        raise HTTPException(
            status_code=422, detail="Incident date can't be in the future"
        )
    # Someone else's policy gets the same answer as one that doesn't exist.
    if not await users.owns_policy(user.customer_id, policy_id):
        raise HTTPException(status_code=422, detail="Unknown policy")
    await limit_submissions(limiter, user)
    claim_id = uuid.uuid4().hex
    stored = await store_photos(jobs, claim_id, photos)
    body = {
        "claim_id": claim_id,
        "customer_id": user.customer_id,
        "policy_id": policy_id,
        "incident_date": incident_date.isoformat(),
        "description": description,
        "claimed_amount": claimed_amount,
        "photos": stored,
    }
    return await call_agent(agent, "POST", "/claims", json=body)


StatusFilter = Literal[
    "processing",
    "awaiting_retake",
    "awaiting_approval",
    "completed",
    "failed",
    "interrupted",
]


@router.get("/claims")
async def list_claims(
    user: CurrentUser,
    agent: Agent,
    status: Annotated[StatusFilter | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
):
    params = {"limit": limit}
    if status:
        params["status"] = status
    if user.role == "customer":
        params["customer_id"] = user.customer_id
    rows = (await call_agent(agent, "GET", "/claims", params=params))["claims"]
    if user.role == "customer":
        rows = [customer_row(r) for r in rows]
    return {"claims": rows}


@router.get("/claims/{claim_id}")
async def get_claim(claim_id: ClaimId, user: CurrentUser, agent: Agent):
    claim = await claim_for(agent, user, claim_id)
    return customer_view(claim) if user.role == "customer" else claim


@router.get("/claims/{claim_id}/photos/{index}")
async def get_photo(
    claim_id: ClaimId,
    index: Annotated[int, Path(ge=0, lt=MAX_PHOTOS)],
    user: CurrentUser,
    agent: Agent,
    store: Annotated[S3Store, Depends(photo_store)],
):
    """Photos are streamed through the gateway, so storage stays internal and
    the same authorization applies as for the claim itself."""
    claim = await claim_for(agent, user, claim_id)
    photos = claim["state"].get("photos", [])
    if index >= len(photos):
        raise HTTPException(status_code=404, detail="No such photo")
    try:
        data, content_type = await asyncio.to_thread(store.get, photos[index]["key"])
    except KeyError as e:
        raise HTTPException(status_code=404, detail="No such photo") from e
    except Exception as e:
        raise HTTPException(status_code=503, detail="Photo storage unavailable") from e
    # The stored type came from the uploader; only serve it if it's an image
    # type we accept, and stop browsers from sniffing anything else.
    if content_type not in ALLOWED_TYPES:
        content_type = "application/octet-stream"
    return Response(
        data,
        media_type=content_type,
        headers={
            "Cache-Control": "private, max-age=3600",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/claims/{claim_id}/photos", status_code=202)
async def retake(
    claim_id: ClaimId,
    user: Customer,
    agent: Agent,
    jobs: Jobs,
    limiter: Limiter,
    photos: Annotated[list[UploadFile], File()],
):
    # Check first, so photos aren't stored for a claim that isn't asking for them.
    current = await claim_for(agent, user, claim_id)
    if current["status"] != "awaiting_retake":
        raise HTTPException(status_code=409, detail="Claim isn't waiting for photos")
    await limit_submissions(limiter, user)
    stored = await store_photos(jobs, claim_id, photos)
    return await call_agent(
        agent, "POST", f"/claims/{claim_id}/photos", json={"photos": stored}
    )


@router.post("/claims/{claim_id}/decision", status_code=202)
async def decide(
    claim_id: ClaimId,
    user: Adjuster,
    decision: Annotated[dict[str, Any], Body()],
    agent: Agent,
):
    # The agent validates the decision's fields (action, amount, notes). The
    # adjuster's name comes from the session, never the request body, so the
    # audit trail records who was actually logged in.
    decision = {**decision, "adjuster": user.username}
    return await call_agent(
        agent, "POST", f"/claims/{claim_id}/decision", json=decision
    )


@router.post("/claims/{claim_id}/retry", status_code=202)
async def retry(claim_id: ClaimId, user: Adjuster, agent: Agent):
    return await call_agent(agent, "POST", f"/claims/{claim_id}/retry")
