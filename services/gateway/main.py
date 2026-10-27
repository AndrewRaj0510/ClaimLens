import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, HTTPException, Path, UploadFile
from fastapi.responses import JSONResponse

import claims
from common import read_upload, vision_jobs
from vision_jobs import VisionJobs

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
VISION_TIMEOUT_S = float(os.getenv("VISION_TIMEOUT_S", "30"))

JobId = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")]


@asynccontextmanager
async def lifespan(app: FastAPI):
    import redis.asyncio as redis

    from storage import S3Store

    if not claims.adjuster_key_usable(claims.ADJUSTER_API_KEY):
        logging.getLogger("gateway").warning(
            "ADJUSTER_API_KEY is unset or shorter than %d characters: decisions are locked",
            claims.MIN_ADJUSTER_KEY_LENGTH,
        )
    store = S3Store()
    await asyncio.to_thread(store.ensure_bucket)
    # The socket timeout must outlast a BLPOP wait, or the client gives up first.
    client = redis.Redis.from_url(
        REDIS_URL, decode_responses=True, socket_timeout=VISION_TIMEOUT_S + 5
    )
    app.state.jobs = VisionJobs(client, store)
    # One pooled client for the agent service, as the vision client was in Phase 1.
    async with httpx.AsyncClient(base_url=claims.AGENT_URL, timeout=30) as agent:
        app.state.agent = agent
        yield
    await client.aclose()


app = FastAPI(title="ClaimLens Gateway", version="0.2.0", lifespan=lifespan)
app.include_router(claims.router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "gateway"}


def job_response(job_id, out):
    """Map a worker result to an HTTP response."""
    if out["status"] == "done":
        return out["result"]
    if out["status"] == "pending":
        return JSONResponse(
            status_code=202, content={"job_id": job_id, "status": "pending"}
        )
    code = out["code"]
    if code == 503 or 400 <= code < 500:
        # Bad image, too large, model not loaded: meaningful to the caller.
        raise HTTPException(status_code=code, detail=out["detail"])
    # Anything else is our fault: don't leak internals.
    raise HTTPException(status_code=502, detail="Vision service error")


@app.post("/detect")
async def detect(
    file: UploadFile,
    jobs: Annotated[VisionJobs, Depends(vision_jobs)],
):
    data, content_type = await read_upload(file)
    # The user's filename isn't stored or forwarded: nothing needs it, and it's
    # untrusted input. The object key is a random job ID.
    try:
        job_id = await jobs.submit(data, content_type)
    except Exception as e:
        raise HTTPException(status_code=503, detail="Job queue unavailable") from e
    out = await jobs.wait(job_id, VISION_TIMEOUT_S)
    if out is None:
        # Workers are busy (or scaling up): hand back a job ID to poll.
        return JSONResponse(
            status_code=202, content={"job_id": job_id, "status": "pending"}
        )
    return job_response(job_id, out)


@app.get("/jobs/{job_id}")
async def get_job(job_id: JobId, jobs: Annotated[VisionJobs, Depends(vision_jobs)]):
    out = await jobs.status(job_id)
    if out is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job")
    return job_response(job_id, out)
