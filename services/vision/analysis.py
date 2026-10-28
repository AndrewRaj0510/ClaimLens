"""Image -> damage detections. Shared by the HTTP API (main.py) and the queue
worker (worker.py), so both validate untrusted bytes the same way."""

import base64
import io
import math
import os

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
# Phone photos can be 4000x3000. Masks are full-resolution bool arrays, one per
# detection, so we run on a downscaled copy and map boxes back to the original.
MAX_SIDE = int(os.getenv("MAX_SIDE", "1280"))

# Refuse "decompression bombs": tiny files that decode to huge images. Pillow
# only *raises* above twice MAX_IMAGE_PIXELS (between 1x and 2x it just warns and
# decodes), so load_image also checks the header's size before decoding.
MAX_PIXELS = 50_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class InvalidImage(ValueError):
    """The bytes aren't a decodable image (maps to HTTP 400)."""


class ImageTooLarge(ValueError):
    """The file is over MAX_UPLOAD_MB (maps to HTTP 413)."""


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
    # The mask inside the box as a base64 1-bit PNG (white = damage), at
    # inference resolution: stretch it over the box to draw it. None if empty.
    mask_png: str | None = None


class DetectResponse(BaseModel):
    width: int
    height: int
    detections: list[DetectionOut]
    # Union of all masks as % of the photo (overlapping damage counted once).
    # It's relative to the photo, not the car, because we don't segment the car itself.
    damage_area_pct: float


def max_bytes():
    return int(MAX_UPLOAD_MB * 1024 * 1024)


def load_image(data: bytes) -> Image.Image:
    """Decode untrusted bytes into an upright RGB image."""
    if len(data) > max_bytes():
        raise ImageTooLarge(f"Image larger than {MAX_UPLOAD_MB} MB")
    try:
        image = Image.open(io.BytesIO(data))  # reads the header only
        if image.width * image.height > MAX_PIXELS:
            raise InvalidImage("Image has too many pixels")
        image.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as e:
        raise InvalidImage("Not a valid image") from e
    # Phones store rotation in EXIF instead of rotating pixels; apply it so
    # boxes line up with what the user sees.
    return ImageOps.exif_transpose(image).convert("RGB")


def mask_png(mask, box):
    """Crop a full-image bool mask to its box and encode it as a 1-bit PNG.

    PNG rather than polygons: Pillow already does it, holes and multi-part masks
    come out right, and a 1-bit PNG of a typical damage is a few hundred bytes.
    """
    h, w = mask.shape
    x1, y1 = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
    x2, y2 = min(w, math.ceil(box[2])), min(h, math.ceil(box[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = mask[y1:y2, x1:x2]
    if not crop.any():
        return None
    buf = io.BytesIO()
    Image.fromarray(crop).save(buf, "PNG", optimize=True)  # bool array -> mode "1"
    return base64.b64encode(buf.getvalue()).decode("ascii")


def analyse(detector, data: bytes) -> DetectResponse:
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
                mask_png=mask_png(d.mask, d.box),
            )
        )
    return DetectResponse(
        width=width,
        height=height,
        detections=out,
        damage_area_pct=round(100 * union.sum() / pixels, 3),
    )
