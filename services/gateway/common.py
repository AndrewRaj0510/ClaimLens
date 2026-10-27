"""Upload checks and dependencies shared by /detect (main.py) and the claim routes."""

import os

import httpx
from fastapi import HTTPException, Request, UploadFile

from vision_jobs import VisionJobs

MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


async def read_upload(file: UploadFile):
    """Cheap checks at the edge. The content type is client-supplied, so the
    vision worker still decodes and validates the actual bytes; this just
    rejects obvious junk early. Returns (bytes, content type)."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG or WebP image")
    max_bytes = int(MAX_UPLOAD_MB * 1024 * 1024)
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413, detail=f"Image larger than {MAX_UPLOAD_MB} MB"
        )
    return data, file.content_type


# Dependencies, so tests can swap in fakes via app.dependency_overrides.
def vision_jobs(request: Request) -> VisionJobs:
    return request.app.state.jobs


def agent_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.agent
