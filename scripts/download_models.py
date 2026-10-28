"""Download the guardrails models into data/models (git-ignored), pinned and
checked, so a changed upstream file can't silently change what gets flagged.

    python scripts/download_models.py

- data/models/prompt-injection/: ProtectAI deberta-v3-base-prompt-injection-v2
  (Apache-2.0), the ONNX export, at a fixed Hugging Face commit. ~740 MB.
  Mounted into the guardrails container at /models.
- data/models/yunet/: OpenCV's YuNet face detector (MIT), for running the
  guardrails tests locally. The guardrails image downloads its own copy at
  build time, with the same checksum.

The AI-image detector (data/models/ai-detector/) is trained, not downloaded:
training/train_ai_detector.ipynb.
"""

import hashlib
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "data" / "models"

CLASSIFIER_REPO = "protectai/deberta-v3-base-prompt-injection-v2"
CLASSIFIER_REVISION = "90c9989b1a342275dd0d1a95aad283c04e075671"
CLASSIFIER_FILES = {
    "onnx/model.onnx": "f0ea7f239f765aedbde7c9e163a7cb38a79c5b8853d3f76db5152172047b228c",
    "onnx/tokenizer.json": "752fe5f0d5678ad563e1bd2ecc1ddf7a3ba7e2024d0ac1dba1a72975e26dff2f",
}
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a"
    "/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verified(path, expected):
    return path.exists() and sha256(path) == expected


def check(path, expected):
    actual = sha256(path)
    if actual != expected:
        path.unlink()
        raise SystemExit(f"{path}: SHA-256 {actual} != pinned {expected}; deleted it")


def download_classifier():
    from huggingface_hub import hf_hub_download

    target = MODELS / "prompt-injection"
    for name, expected in CLASSIFIER_FILES.items():
        path = target / name
        if verified(path, expected):
            print(f"ok      {path.relative_to(ROOT)}")
            continue
        print(f"fetch   {CLASSIFIER_REPO}/{name}")
        hf_hub_download(
            CLASSIFIER_REPO, name, revision=CLASSIFIER_REVISION, local_dir=target
        )
        check(path, expected)
    hf_hub_download(
        CLASSIFIER_REPO, "LICENSE", revision=CLASSIFIER_REVISION, local_dir=target
    )


def download_yunet():
    path = MODELS / "yunet" / "face_detection_yunet_2023mar.onnx"
    if verified(path, YUNET_SHA256):
        print(f"ok      {path.relative_to(ROOT)}")
        return
    print("fetch   YuNet")
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(YUNET_URL, timeout=60) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f)
    check(path, YUNET_SHA256)


if __name__ == "__main__":
    download_classifier()
    download_yunet()
