import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

from detector import Detector

MODEL_PATH = os.getenv("MODEL_PATH", "/models/best.onnx")
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
# Phone photos can be 4000x3000. Masks are full-resolution bool arrays, one per
# detection, so we run on a downscaled copy and map boxes back to the original.
MAX_SIDE = int(os.getenv("MAX_SIDE", "1280"))

# Refuse "decompression bombs": tiny files that decode to gigapixel images.
Image.MAX_IMAGE_PIXELS = 50_000_000

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


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionOut(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    box: Box
    area_pct: float  # this damage's mask area as % of the photo


class DetectResponse(BaseModel):
    width: int
    height: int
    detections: list[DetectionOut]
    # Union of all masks as % of the photo (overlapping damage counted once).
    # It's relative to the photo, not the car, because we don't segment the car itself.
    damage_area_pct: float


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "service": "vision",
        "model_loaded": request.app.state.detector is not None,
    }


def load_image(data: bytes) -> Image.Image:
    """Decode untrusted bytes into an upright RGB image, or raise 400."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as e:
        raise HTTPException(status_code=400, detail="Not a valid image") from e
    # Phones store rotation in EXIF instead of rotating pixels; apply it so
    # boxes line up with what the user sees.
    return ImageOps.exif_transpose(image).convert("RGB")


# Sync endpoint on purpose: onnxruntime blocks the CPU, so FastAPI runs it in
# its threadpool instead of stalling the event loop.
@app.post("/detect", response_model=DetectResponse)
def detect(request: Request, file: UploadFile):
    detector = request.app.state.detector
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    max_bytes = int(MAX_UPLOAD_MB * 1024 * 1024)
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413, detail=f"Image larger than {MAX_UPLOAD_MB} MB"
        )

    image = load_image(data)
    width, height = image.size
    work = image.copy()
    work.thumbnail((MAX_SIDE, MAX_SIDE))
    scale = width / work.width

    detections = detector(work, conf=CONF_THRESHOLD)
    pixels = work.width * work.height
    union = np.zeros((work.height, work.width), dtype=bool)
    out = []
    for d in detections:
        union |= d.mask
        x1, y1, x2, y2 = (round(v * scale, 1) for v in d.box)
        out.append(
            DetectionOut(
                class_id=d.class_id,
                class_name=detector.names[d.class_id],
                confidence=round(d.confidence, 4),
                box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                area_pct=round(100 * d.mask.sum() / pixels, 3),
            )
        )
    return DetectResponse(
        width=width,
        height=height,
        detections=out,
        damage_area_pct=round(100 * union.sum() / pixels, 3),
    )
