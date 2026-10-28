"""Train and evaluate the AI-generated-image detector on GenImage.

Used by train_ai_detector.ipynb (Colab); the logic lives here so it's tested.

Recipe (UnivFD, Ojha et al., CVPR 2023): a frozen CLIP image encoder turns each
image into a feature vector, and a logistic regression separates real from
generated in that space. CLIP was trained on 400M real images, so "real" is a
broad, well-shaped region of its feature space; a linear boundary there
generalises to generators it never saw much better than a CNN trained end to
end, which learns one generator's artefacts.

Protocol (GenImage's own): train on Stable Diffusion v1.4 only, test on the
val split of every generator. Generators other than SD v1.4 are unseen.

The JPEG shortcut: in GenImage, real images are ImageNet JPEGs and generated
ones are PNGs. A detector can score well by spotting JPEG compression, which
says nothing about generation and fails on the first JPEG fake (every phone
upload is a JPEG). So every image, of both classes, is re-encoded as JPEG at a
random quality and size before features are taken (`align`), and
`format_report` shows how lopsided the formats are before that.
"""

import io
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "guardrails"))
from forensics import clip_preprocess

TRAIN_GENERATOR = "sdv4"  # matched against folder names, e.g. imagenet_ai_0419_sdv4
JPEG_QUALITY = (65, 95)
DOWNSCALE = (0.5, 1.0)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_generators(root):
    """{generator: {split: {"ai": dir, "nature": dir}}} for GenImage's layout,
    <root>/<anything>/<generator folder>/<split>/{ai,nature}."""
    found = {}
    for ai_dir in sorted(Path(root).glob("**/ai")):
        split_dir = ai_dir.parent
        nature = split_dir / "nature"
        if split_dir.name not in ("train", "val") or not nature.is_dir():
            continue
        generator = split_dir.parent.name
        found.setdefault(generator, {})[split_dir.name] = {
            "ai": ai_dir,
            "nature": nature,
        }
    return found


def list_images(folder):
    return sorted(
        p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )


def sample(paths, n, seed):
    paths = list(paths)
    random.Random(seed).shuffle(paths)
    return paths[:n]


def format_report(paths_by_label):
    """Share of each file format per label, e.g. {"ai": {".png": 1.0}}."""
    report = {}
    for label, paths in paths_by_label.items():
        counts = {}
        for p in paths:
            counts[p.suffix.lower()] = counts.get(p.suffix.lower(), 0) + 1
        report[label] = {k: round(v / len(paths), 3) for k, v in sorted(counts.items())}
    return report


def align(image, rng):
    """Random downscale + JPEG re-encode, the same for real and generated
    images, so file format and compression stop being a tell."""
    image = image.convert("RGB")
    scale = rng.uniform(*DOWNSCALE)
    if scale < 1.0:
        w, h = image.size
        image = image.resize(
            (max(32, round(w * scale)), max(32, round(h * scale))),
            Image.Resampling.BICUBIC,
        )
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=rng.randint(*JPEG_QUALITY))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def load_batch(paths, rng, aligned=True, size=224):
    """(N, 3, size, size) float32, preprocessed exactly as the service does."""
    out = []
    for p in paths:
        with Image.open(p) as image:
            rgb = align(image, rng) if aligned else image.convert("RGB")
        out.append(clip_preprocess(rgb, size)[0])
    return np.stack(out)


def extract(paths, encode, batch_size=64, seed=0, aligned=True):
    """Features for `paths` with `encode`: (N, 3, 224, 224) array -> (N, D)
    array (the CLIP image encoder). L2-normalised, as the service does."""
    rng = random.Random(seed)
    feats = []
    for i in range(0, len(paths), batch_size):
        f = np.asarray(encode(load_batch(paths[i : i + batch_size], rng, aligned)))
        feats.append(f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8))
    return np.concatenate(feats).astype(np.float32)


def fit_probe(X, y, seed=0):
    """Logistic regression, C chosen by 5-fold cross-validation."""
    from sklearn.linear_model import LogisticRegressionCV

    return LogisticRegressionCV(
        Cs=[0.1, 0.3, 1, 3, 10, 30], cv=5, max_iter=5000, random_state=seed
    ).fit(X, y)


def probe_scores(X, weights, bias):
    """What the service computes: sigmoid(features . w + b)."""
    return 1.0 / (1.0 + np.exp(-(X @ weights + bias)))


def pick_threshold(real_scores, max_fpr=0.01):
    """The lowest threshold that flags at most `max_fpr` of real photos.
    A false flag sends an honest customer's claim to an investigator, so the
    threshold is set by the false-positive rate, not by accuracy."""
    real_scores = np.sort(np.asarray(real_scores))
    k = int(np.ceil(len(real_scores) * (1 - max_fpr))) - 1
    k = min(max(k, 0), len(real_scores) - 1)
    return float(np.nextafter(real_scores[k], np.inf))


def metrics(ai_scores, real_scores, threshold):
    from sklearn.metrics import roc_auc_score

    ai_scores, real_scores = np.asarray(ai_scores), np.asarray(real_scores)
    y = np.r_[np.ones(len(ai_scores)), np.zeros(len(real_scores))]
    s = np.r_[ai_scores, real_scores]
    return {
        "auc": round(float(roc_auc_score(y, s)), 4),
        "accuracy": round(float(((s >= threshold) == y).mean()), 4),
        "tpr": round(float((ai_scores >= threshold).mean()), 4),  # fakes caught
        "fpr": round(float((real_scores >= threshold).mean()), 4),  # reals flagged
        "n": len(s),
    }


def probe_json(weights, bias, threshold, results, model_name, output_name, notes=""):
    """The file services/guardrails/forensics.AiDetector loads."""
    return {
        "model": model_name,
        "output": output_name,
        "l2_normalize": True,
        "image_size": 224,
        "weights": [round(float(w), 7) for w in weights],
        "bias": float(bias),
        "threshold": float(threshold),
        "metrics": results,
        "notes": notes,
    }


def save_probe(path, probe):
    Path(path).write_text(json.dumps(probe, indent=1))
