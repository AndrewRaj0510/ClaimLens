"""Photo checks before a photo may reach an LLM: OCR and redaction.

- OCR (RapidOCR: PaddleOCR's detection + recognition models on ONNX Runtime)
  reads any text in the photo. That text is scanned for injection like any
  other untrusted text: a sticker or a sheet of paper in the frame saying
  "ignore your instructions" is a visual prompt injection.
- Redaction pixelates faces (YuNet, OpenCV's small face detector) and every
  text region OCR found. Text regions cover number plates, and also mean the
  vision LLM never reads text written in the photo at all: damage assessment
  doesn't need it, so the visual-injection channel is closed rather than just
  monitored. The redacted copy has no EXIF (so no GPS) and is downscaled
  (WORK_SIDE).

Pixelating to a few blocks, not a Gaussian blur: light blurs of faces and text
can be partly reversed, a 6-block mosaic can't.
"""

import io
import os
import re
import threading

import numpy as np
from PIL import Image, ImageOps

MAX_PIXELS = 50_000_000  # same decompression-bomb limit as vision and the agent
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

OCR_MIN_SCORE = 0.5  # text below this is still redacted, but not scanned
FACE_MIN_SCORE = 0.6
# Everything runs on one working copy at most this big, and that same copy
# (redacted) is what the vision LLM gets. So OCR sees exactly the pixels the
# LLM will: text too small for OCR to find is too small for the LLM to read.
WORK_SIDE = 1280
PIXEL_BLOCKS = 6
MARGIN = 0.15  # grow each box by this fraction so edges of letters don't survive

PLATE = re.compile(r"[A-Z0-9]{2,4}[ \-\u00b7.]?[A-Z0-9]{2,5}")


class InvalidImage(ValueError):
    pass


def open_image(data):
    try:
        image = Image.open(io.BytesIO(data))  # reads the header only
        if image.width * image.height > MAX_PIXELS:
            raise InvalidImage("Image has too many pixels")
        image.load()
    except (OSError, Image.DecompressionBombError, SyntaxError) as e:
        raise InvalidImage("Not a readable image") from e
    return image


def to_rgb(image):
    """Upright (EXIF orientation applied) RGB copy."""
    return ImageOps.exif_transpose(image).convert("RGB")


def working_copy(rgb):
    small = rgb.copy()
    small.thumbnail((WORK_SIDE, WORK_SIDE), Image.Resampling.LANCZOS)
    return small


def looks_like_plate(text):
    compact = text.upper().strip()
    if not PLATE.fullmatch(compact):
        return False
    alnum = re.sub(r"[^A-Z0-9]", "", compact)
    return 5 <= len(alnum) <= 8 and any(c.isdigit() for c in alnum)


class Ocr:
    def __init__(self):
        from rapidocr import RapidOCR

        # text_score 0 keeps every detected region: low-confidence regions are
        # still redacted, they're just not trusted as readable text.
        self.engine = RapidOCR(
            params={"Global.log_level": "error", "Global.text_score": 0.0}
        )
        self.lock = threading.Lock()

    def read(self, rgb):
        """[{"text", "score", "box": [x1, y1, x2, y2]}] in reading order."""
        bgr = np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])
        with self.lock:
            out = self.engine(bgr)
        if out.boxes is None or out.txts is None:
            return []
        lines = []
        for poly, text, score in zip(out.boxes, out.txts, out.scores, strict=True):
            xs, ys = poly[:, 0], poly[:, 1]
            box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            lines.append(
                {
                    "text": text,
                    "score": round(float(score), 3),
                    "box": [round(v) for v in box],
                }
            )
        # Top to bottom, then left to right, so joined text reads naturally.
        lines.sort(key=lambda r: (round(r["box"][1] / 20), r["box"][0]))
        return lines


class FaceDetector:
    def __init__(self, model_path):
        import cv2

        self.cv2 = cv2
        self.detector = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), FACE_MIN_SCORE, 0.3, 50
        )
        self.lock = threading.Lock()  # the detector keeps its input size as state

    def detect(self, rgb):
        """[{"box": [x1, y1, x2, y2], "score"}] in the image's own pixels."""
        bgr = np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])
        with self.lock:
            self.detector.setInputSize(rgb.size)
            _, faces = self.detector.detect(bgr)
        if faces is None:
            return []
        return [
            {
                "box": [round(float(v)) for v in (x, y, x + fw, y + fh)],
                "score": round(float(f[-1]), 3),
            }
            for f in faces
            for x, y, fw, fh in [f[:4]]
        ]


def grow(box, size, margin=MARGIN):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    w, h = size
    return (
        max(0, int(x1 - dx)),
        max(0, int(y1 - dy)),
        min(w, int(x2 + dx + 1)),
        min(h, int(y2 + dy + 1)),
    )


def pixelate(image, boxes):
    """Copy of the image with each box replaced by a coarse mosaic."""
    out = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = grow(box, image.size)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        region = out.crop((x1, y1, x2, y2))
        bw = max(1, min(PIXEL_BLOCKS, region.width // 4))
        bh = max(1, round(bw * region.height / region.width))
        mosaic = region.resize((bw, bh), Image.Resampling.BOX).resize(
            region.size, Image.Resampling.NEAREST
        )
        out.paste(mosaic, (x1, y1))
    return out


def redacted_jpeg(work, boxes):
    """Pixelated JPEG of the working copy, with no metadata at all."""
    image = pixelate(work, boxes)
    buf = io.BytesIO()
    # A fresh RGB image saved without exif=... carries no EXIF, XMP or ICC data.
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


FACE_MODEL_PATH = os.getenv(
    "FACE_MODEL_PATH", "/app/models/face_detection_yunet_2023mar.onnx"
)
