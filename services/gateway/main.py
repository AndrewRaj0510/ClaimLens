import os
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile

VISION_URL = os.getenv("VISION_URL", "http://vision:8000")
VISION_TIMEOUT_S = float(os.getenv("VISION_TIMEOUT_S", "30"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One client for the app's lifetime so connections to vision are pooled
    # instead of opening a new TCP connection per request.
    async with httpx.AsyncClient(
        base_url=VISION_URL, timeout=VISION_TIMEOUT_S
    ) as client:
        app.state.vision = client
        yield


app = FastAPI(title="ClaimLens Gateway", version="0.1.0", lifespan=lifespan)


def vision_client(request: Request) -> httpx.AsyncClient:
    """Dependency, so tests can swap in a mock transport via dependency_overrides."""
    return request.app.state.vision


@app.get("/health")
def health():
    return {"status": "ok", "service": "gateway"}


@app.post("/detect")
async def detect(
    file: UploadFile,
    vision: Annotated[httpx.AsyncClient, Depends(vision_client)],
):
    # Cheap checks at the edge. The content type is client-supplied, so vision
    # still decodes and validates the actual bytes; this just rejects obvious junk early.
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG or WebP image")
    max_bytes = int(MAX_UPLOAD_MB * 1024 * 1024)
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413, detail=f"Image larger than {MAX_UPLOAD_MB} MB"
        )

    try:
        # The user's filename isn't forwarded: vision doesn't need it, and it's untrusted input.
        r = await vision.post(
            "/detect", files={"file": ("upload", data, file.content_type)}
        )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail="Vision service unavailable") from e

    if r.status_code == 503 or (400 <= r.status_code < 500):
        # Client errors (bad image, too large) and "model not loaded" are meaningful to the caller.
        raise HTTPException(status_code=r.status_code, detail=r.json().get("detail"))
    if r.status_code >= 500:
        raise HTTPException(status_code=502, detail="Vision service error")
    return r.json()
