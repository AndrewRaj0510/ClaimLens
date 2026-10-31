"""Guardrails: checks on untrusted input before it reaches an LLM.

    POST /scan/text       prompt-injection scan (rules + classifier) of texts
    POST /images/inspect  OCR + injection scan of the text in a photo, face and
                          text redaction, metadata forensics, AI-image score;
                          returns a redacted, metadata-free copy for the vision LLM

Internal only: the agent calls it. It never decides anything; it reports, and
the agent keeps flagged input away from the LLM and in front of a human.

Model files: the classifier and the AI-image detector are mounted from
data/models (like vision's weights). Without them the service still starts,
and says so in /health and in every response (mode, ai_generated: null), so a
missing model is visible instead of silently passing everything.
"""

import base64
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

import forensics
import http_metrics
import photos
from text_scan import Classifier, Scanner

CLASSIFIER_DIR = os.getenv("CLASSIFIER_DIR", "/models/prompt-injection/onnx")
MAX_UPLOAD_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024)

log = logging.getLogger("guardrails")


def load_optional(name, factory, path):
    if not Path(path).exists():
        log.warning("%s not found at %s; running without it", name, path)
        return None
    loaded = factory(path)
    log.info("loaded %s from %s", name, path)
    return loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = app.state
    if not hasattr(s, "scanner"):  # tests set these before startup
        s.scanner = Scanner(load_optional("classifier", Classifier, CLASSIFIER_DIR))
        s.ocr = photos.Ocr()
        s.faces = photos.FaceDetector(photos.FACE_MODEL_PATH)
        s.ai_detector = load_optional(
            "AI-image detector", forensics.AiDetector, forensics.AI_DETECTOR_DIR
        )
    yield


app = FastAPI(title="ClaimLens Guardrails", version="0.1.0", lifespan=lifespan)
http_metrics.instrument(app)


@app.get("/health")
def health(request: Request):
    s = request.app.state
    return {
        "status": "ok",
        "service": "guardrails",
        "text_scan": s.scanner.mode,
        "ai_detector": s.ai_detector is not None,
    }


class TextScanRequest(BaseModel):
    texts: list[Annotated[str, Field(max_length=20_000)]] = Field(
        min_length=1, max_length=64
    )
    # untrusted: customer text, OCR text. document: our own policy wording,
    # where only rule hits flag (see text_scan.py for the measurement).
    profile: Literal["untrusted", "document"] = "untrusted"


# Sync endpoints on purpose: the models block the CPU, so FastAPI runs them in
# its threadpool instead of stalling the event loop.
@app.post("/scan/text")
def scan_text(request: Request, body: TextScanRequest):
    scanner = request.app.state.scanner
    return {
        "mode": scanner.mode,
        "profile": body.profile,
        "results": scanner.scan(body.texts, body.profile),
    }


@app.post("/images/inspect")
def inspect_image(
    request: Request,
    file: UploadFile,
    incident_date: Annotated[str | None, Form()] = None,
):
    s = request.app.state
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large")
    try:
        original = photos.open_image(data)
        rgb = photos.to_rgb(original)
    except photos.InvalidImage as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    metadata = forensics.check_metadata(
        original, forensics.parse_incident_date(incident_date)
    )
    work = photos.working_copy(rgb)
    regions = s.ocr.read(work)
    readable = [r for r in regions if r["score"] >= photos.OCR_MIN_SCORE]
    text = "\n".join(r["text"] for r in readable)
    text_scan = s.scanner.scan([text])[0] if text else None
    faces = s.faces.detect(work)
    ai = None
    if s.ai_detector is not None:
        score = s.ai_detector.score(rgb)
        ai = {
            "score": round(score, 4),
            "flagged": score >= s.ai_detector.threshold,
            "model": s.ai_detector.name,
        }
    redacted = photos.redacted_jpeg(
        work, [f["box"] for f in faces] + [r["box"] for r in regions]
    )
    # Boxes are in the working copy's pixels, the same as the redacted image.
    return {
        "width": rgb.width,
        "height": rgb.height,
        "work_size": list(work.size),
        "ocr": readable,
        "text_scan": text_scan,
        "text_scan_mode": s.scanner.mode,
        "redactions": {
            "faces": len(faces),
            "text_regions": len(regions),
            "plates": sum(photos.looks_like_plate(r["text"]) for r in readable),
        },
        "metadata": metadata,
        "ai_generated": ai,
        "redacted_jpeg": base64.b64encode(redacted).decode(),
    }
