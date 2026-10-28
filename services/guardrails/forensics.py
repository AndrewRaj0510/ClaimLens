"""Image forensics: metadata checks and an AI-generated-image detector.

Metadata (EXIF, XMP, PNG text chunks) is cheap to check and, when present,
specific: generators and editors often say who made the file. It's also
trivial to strip or fake, so a clean result proves nothing; findings are
signals for a human, never a verdict. Most phone photos uploaded through a
browser keep EXIF, but screenshots, messaging apps and some browsers strip it,
so missing metadata is reported as information only.

The detector (AiDetector) is a CLIP image encoder plus a linear probe trained
on GenImage (training/train_ai_detector.ipynb). It's optional: without the
model files the service reports "not checked" rather than "real".
"""

import json
import os
import re
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
from PIL import Image

# IPTC "digital source type" values that mean generative AI made or edited it.
# Adobe Firefly, OpenAI and Google write these into XMP / C2PA metadata.
AI_SOURCE_TYPES = (b"trainedAlgorithmicMedia", b"compositeWithTrainedAlgorithmicMedia")
# PNG text chunks written by Stable Diffusion front-ends (AUTOMATIC1111, ComfyUI).
AI_PNG_KEYS = {"parameters", "prompt", "workflow", "sd-metadata", "invokeai_metadata"}
AI_SOFTWARE = re.compile(
    r"midjourney|dall[-\u00b7e ]|stable ?diffusion|firefly|imagen|novelai|comfyui"
    r"|automatic1111|leonardo|ideogram|flux|gemini|chatgpt|openai",
    re.IGNORECASE,
)
EDIT_SOFTWARE = re.compile(
    r"photoshop|gimp|lightroom|snapseed|pixelmator|affinity|canva|picsart|paint\.net"
    r"|facetune|photopea|luminar",
    re.IGNORECASE,
)

EXIF_IFD, GPS_IFD = 0x8769, 0x8825
MAKE, MODEL, SOFTWARE, DATETIME = 0x010F, 0x0110, 0x0131, 0x0132
DATETIME_ORIGINAL = 0x9003
PREDATES_SLACK_DAYS = 1  # time zones and camera clocks set a day off


def finding(code, severity, detail):
    return {"code": code, "severity": severity, "detail": detail}


def metadata_blobs(image):
    """Raw metadata segments to search for markers (not the pixel data)."""
    blobs = []
    for _marker, payload in getattr(image, "applist", []):  # JPEG APPn segments
        blobs.append(payload)
    for key, value in image.info.items():
        if isinstance(value, bytes):
            blobs.append(value)
        elif isinstance(value, str):
            blobs.append(f"{key}={value}".encode())
    return blobs


def parse_exif_date(value):
    try:
        return datetime.strptime(str(value).strip()[:19], "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def check_metadata(image, incident_date=None, today=None):
    """Return {"camera", "software", "taken_at", "has_exif", "has_gps", "findings"}.

    `image` is the PIL image as opened (before any conversion, which drops
    metadata)."""
    today = today or datetime.now(UTC).date()
    exif = image.getexif()
    sub = exif.get_ifd(EXIF_IFD) if exif else {}
    camera = " ".join(str(exif.get(t, "")).strip() for t in (MAKE, MODEL)).strip()
    software = str(exif.get(SOFTWARE, "")).strip() or None
    taken = parse_exif_date(sub.get(DATETIME_ORIGINAL) or exif.get(DATETIME))
    has_gps = bool(exif.get_ifd(GPS_IFD)) if exif else False
    blobs = metadata_blobs(image)
    found = []

    if any(marker in blob for blob in blobs for marker in AI_SOURCE_TYPES):
        found.append(
            finding(
                "ai_metadata",
                "high",
                "Metadata declares the image as made or edited by generative AI.",
            )
        )
    png_keys = AI_PNG_KEYS & {k.lower() for k in image.info if isinstance(k, str)}
    if png_keys:
        found.append(
            finding(
                "ai_metadata",
                "high",
                f"Image carries image-generator settings ({', '.join(sorted(png_keys))}).",
            )
        )
    if software and AI_SOFTWARE.search(software):
        found.append(
            finding(
                "ai_metadata",
                "high",
                f"Made with image-generation software: {software}.",
            )
        )
    elif software and EDIT_SOFTWARE.search(software):
        found.append(
            finding("edited", "warning", f"Saved by editing software: {software}.")
        )
    if any(b"c2pa" in blob for blob in blobs):
        # Phones and cameras sign photos with C2PA too, so presence alone says
        # nothing; reading the manifest needs a C2PA library (future work).
        found.append(
            finding(
                "content_credentials", "info", "Image has C2PA content credentials."
            )
        )
    if taken and incident_date:
        days = (incident_date - taken.date()).days
        if days > PREDATES_SLACK_DAYS:
            found.append(
                finding(
                    "taken_before_incident",
                    "high",
                    f"Photo was taken {taken.date()}, {days} days before the reported incident.",
                )
            )
    if taken and (taken.date() - today).days > PREDATES_SLACK_DAYS:
        found.append(
            finding(
                "future_date",
                "warning",
                f"Photo's date {taken.date()} is in the future.",
            )
        )
    if not exif:
        found.append(
            finding(
                "no_camera_metadata",
                "info",
                "No camera metadata (normal for screenshots and some apps and browsers).",
            )
        )
    if has_gps:
        found.append(
            finding(
                "location_in_metadata",
                "info",
                "GPS location present (removed before any LLM sees the photo).",
            )
        )
    # Deduplicate by code: two AI markers are one finding to a reader.
    seen, unique = set(), []
    for f in found:
        if f["code"] not in seen:
            seen.add(f["code"])
            unique.append(f)
    return {
        "camera": camera or None,
        "software": software,
        "taken_at": taken.isoformat() if taken else None,
        "has_exif": bool(exif),
        "has_gps": has_gps,
        "findings": unique,
    }


AI_DETECTOR_DIR = os.getenv("AI_DETECTOR_DIR", "/models/ai-detector")
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def clip_preprocess(rgb, size=224):
    """CLIP's own preprocessing: shortest side to `size` (bicubic), centre crop,
    normalise with CLIP's mean and std. Returns (1, 3, size, size) float32.

    training/ai_detector.py imports this function, so the features the probe
    was trained on are computed exactly like the ones it scores here."""
    w, h = rgb.size
    scale = size / min(w, h)
    image = rgb.resize(
        (max(size, round(w * scale)), max(size, round(h * scale))),
        Image.Resampling.BICUBIC,
    )
    left = (image.width - size) // 2
    top = (image.height - size) // 2
    image = image.crop((left, top, left + size, top + size))
    x = (np.asarray(image, dtype=np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD
    return x.transpose(2, 0, 1)[None]


class AiDetector:
    """CLIP image features + a logistic-regression probe (the UnivFD recipe,
    Ojha et al. 2023): CLIP's features were learned on 400M real images, so a
    linear boundary in that space generalises to generators it wasn't trained
    on far better than a CNN trained end to end on one generator's artefacts.

    Files in `model_dir` (written by training/train_ai_detector.ipynb):
      clip_visual.onnx  image encoder, input "pixel_values" (N, 3, 224, 224)
      probe.json        {"model", "output", "l2_normalize", "weights", "bias",
                         "threshold", "image_size", "metrics": {...}}
    """

    def __init__(self, model_dir=AI_DETECTOR_DIR):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "probe.json").read_text())
        self.session = ort.InferenceSession(
            str(model_dir / "clip_visual.onnx"), providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0].name
        self.output = cfg.get("output")
        self.l2 = cfg.get("l2_normalize", True)
        self.weights = np.asarray(cfg["weights"], dtype=np.float32)
        self.bias = float(cfg["bias"])
        self.threshold = float(cfg["threshold"])
        self.size = int(cfg.get("image_size", 224))
        self.name = cfg.get("model", "clip-linear-probe")

    def preprocess(self, rgb):
        return clip_preprocess(rgb, self.size)

    def features(self, rgb):
        outputs = self.session.run(
            [self.output] if self.output else None, {self.input: self.preprocess(rgb)}
        )
        f = outputs[0].reshape(-1).astype(np.float32)
        return f / (np.linalg.norm(f) + 1e-8) if self.l2 else f

    def score(self, rgb):
        """Probability the image is AI-generated, by the probe."""
        z = float(self.features(rgb) @ self.weights + self.bias)
        return 1.0 / (1.0 + np.exp(-z))


def parse_incident_date(value):
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None
