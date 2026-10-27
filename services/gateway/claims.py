"""Claim endpoints: the public face of the agent service.

    POST /claims                   customer: form fields + photos
    GET  /claims/{id}              status, pending question, results
    POST /claims/{id}/photos       customer: retake photos
    POST /claims/{id}/decision     adjuster only (X-Adjuster-Key header)

Photos are stored and their vision jobs queued here, before the agent starts,
so detection runs in parallel with the agent's first (LLM) step.

Authentication is minimal until Phase 3: the decision endpoint needs a shared
adjuster key, compared in constant time. Without it, anyone who can reach the
gateway could approve a claim.
"""

import hmac
import os
import uuid
from datetime import UTC, date, datetime
from typing import Annotated, Any

import httpx
from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Path,
    UploadFile,
)

from common import agent_client, read_upload, vision_jobs
from vision_jobs import VisionJobs

AGENT_URL = os.getenv("AGENT_URL", "http://agent:8000")
ADJUSTER_API_KEY = os.getenv("ADJUSTER_API_KEY", "")
MIN_ADJUSTER_KEY_LENGTH = 16
MAX_PHOTOS = 10

router = APIRouter()
ClaimId = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")]


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


@router.post("/claims", status_code=202)
async def create_claim(
    agent: Annotated[httpx.AsyncClient, Depends(agent_client)],
    jobs: Annotated[VisionJobs, Depends(vision_jobs)],
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
    claim_id = uuid.uuid4().hex
    stored = await store_photos(jobs, claim_id, photos)
    body = {
        "claim_id": claim_id,
        "policy_id": policy_id,
        "incident_date": incident_date.isoformat(),
        "description": description,
        "claimed_amount": claimed_amount,
        "photos": stored,
    }
    return await call_agent(agent, "POST", "/claims", json=body)


@router.get("/claims/{claim_id}")
async def get_claim(
    claim_id: ClaimId, agent: Annotated[httpx.AsyncClient, Depends(agent_client)]
):
    return await call_agent(agent, "GET", f"/claims/{claim_id}")


@router.post("/claims/{claim_id}/photos", status_code=202)
async def retake(
    claim_id: ClaimId,
    agent: Annotated[httpx.AsyncClient, Depends(agent_client)],
    jobs: Annotated[VisionJobs, Depends(vision_jobs)],
    photos: Annotated[list[UploadFile], File()],
):
    # Check first, so photos aren't stored for a claim that isn't asking for them.
    current = await call_agent(agent, "GET", f"/claims/{claim_id}")
    if current["status"] != "awaiting_retake":
        raise HTTPException(status_code=409, detail="Claim isn't waiting for photos")
    stored = await store_photos(jobs, claim_id, photos)
    return await call_agent(
        agent, "POST", f"/claims/{claim_id}/photos", json={"photos": stored}
    )


def adjuster_key_usable(key):
    """An unset or short key locks the endpoint: a placeholder like "change-me"
    copied from a template must never become a working password."""
    return len(key) >= MIN_ADJUSTER_KEY_LENGTH


def require_adjuster(x_adjuster_key: Annotated[str, Header()] = ""):
    # compare_digest takes the same time whatever the input, so the key can't be
    # guessed byte by byte from response times. It's given bytes: with str it
    # raises on non-ASCII (headers decode as latin-1), which would be a 500.
    if not adjuster_key_usable(ADJUSTER_API_KEY) or not hmac.compare_digest(
        x_adjuster_key.encode(), ADJUSTER_API_KEY.encode()
    ):
        raise HTTPException(status_code=401, detail="Adjuster key required")


@router.post(
    "/claims/{claim_id}/decision",
    status_code=202,
    dependencies=[Depends(require_adjuster)],
)
async def decide(
    claim_id: ClaimId,
    decision: Annotated[dict[str, Any], Body()],
    agent: Annotated[httpx.AsyncClient, Depends(agent_client)],
):
    # The agent validates the decision's fields (action, amount, adjuster, notes).
    return await call_agent(
        agent, "POST", f"/claims/{claim_id}/decision", json=decision
    )
