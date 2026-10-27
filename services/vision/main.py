import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile

from analysis import (
    DetectResponse,
    ImageTooLarge,
    InvalidImage,
    analyse,
    max_bytes,
)
from detector import Detector

MODEL_PATH = os.getenv("MODEL_PATH", "/models/best.onnx")

log = logging.getLogger("vision")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start even without weights so the stack can come up before a model is
    # trained; /detect answers 503 until the file exists and the service restarts.
    if Path(MODEL_PATH).exists():
        app.state.detector = Detector(MODEL_PATH)
        log.info("loaded %s, classes=%s", MODEL_PATH, app.state.detector.names)
    else:
        app.state.detector = None
        log.warning("model not found at %s; /detect will return 503", MODEL_PATH)
    yield


app = FastAPI(title="ClaimLens Vision", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "service": "vision",
        "model_loaded": request.app.state.detector is not None,
    }


# Sync endpoint on purpose: onnxruntime blocks the CPU, so FastAPI runs it in
# its threadpool instead of stalling the event loop.
@app.post("/detect", response_model=DetectResponse)
def detect(request: Request, file: UploadFile):
    detector = request.app.state.detector
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    data = file.file.read(max_bytes() + 1)
    try:
        return analyse(detector, data)
    except ImageTooLarge as e:
        raise HTTPException(status_code=413, detail=str(e)) from e
    except InvalidImage as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
