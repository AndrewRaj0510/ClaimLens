import random
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import ai_detector as ad

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "services" / "guardrails")
)
import forensics


def make_genimage(root, generators=("imagenet_ai_0419_sdv4", "imagenet_midjourney")):
    for g in generators:
        for split in ("train", "val"):
            for label, ext in (("ai", ".png"), ("nature", ".JPEG")):
                d = root / "GenImage" / g / g / split / label
                d.mkdir(parents=True)
                for i in range(3):
                    Image.new("RGB", (64, 48), (i * 40, 80, 120)).save(
                        d / f"{i}{ext}", format="PNG" if ext == ".png" else "JPEG"
                    )
    (root / "GenImage" / "README.txt").write_text("not a split")


def test_find_generators_reads_genimage_layout(tmp_path):
    make_genimage(tmp_path)
    found = ad.find_generators(tmp_path)
    assert set(found) == {"imagenet_ai_0419_sdv4", "imagenet_midjourney"}
    assert set(found["imagenet_midjourney"]) == {"train", "val"}
    assert found["imagenet_midjourney"]["val"]["ai"].name == "ai"
    assert len(ad.list_images(found["imagenet_midjourney"]["val"]["nature"])) == 3


def test_format_report_exposes_the_jpeg_shortcut(tmp_path):
    make_genimage(tmp_path)
    g = ad.find_generators(tmp_path)["imagenet_ai_0419_sdv4"]["train"]
    report = ad.format_report(
        {label: ad.list_images(g[label]) for label in ("ai", "nature")}
    )
    assert report == {"ai": {".png": 1.0}, "nature": {".jpeg": 1.0}}


def test_sample_is_seeded():
    paths = list(range(100))
    assert ad.sample(paths, 5, seed=1) == ad.sample(paths, 5, seed=1)
    assert ad.sample(paths, 5, seed=1) != ad.sample(paths, 5, seed=2)


def test_align_reencodes_as_jpeg_within_size_bounds():
    rng = random.Random(0)
    png = Image.effect_noise((400, 300), 50).convert("RGB")
    for _ in range(10):
        out = ad.align(png, rng)
        assert out.mode == "RGB"
        assert 200 <= out.width <= 400 and 150 <= out.height <= 300
        assert not np.array_equal(  # lossy: JPEG changed the pixels
            np.asarray(out.resize(png.size)), np.asarray(png)
        )


def test_load_batch_matches_the_services_preprocessing(tmp_path):
    p = tmp_path / "x.png"
    Image.effect_noise((300, 200), 60).convert("RGB").save(p)
    batch = ad.load_batch([p], random.Random(0), aligned=False)
    expected = forensics.clip_preprocess(Image.open(p).convert("RGB"))
    assert batch.shape == (1, 3, 224, 224)
    assert np.array_equal(batch, expected)


def test_extract_normalises_features(tmp_path):
    paths = []
    for i in range(5):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (64, 64), (i * 50, 10, 10)).save(p)
        paths.append(p)

    def encode(x):  # stands in for CLIP: 3 channel means + a constant
        return np.c_[x.mean(axis=(2, 3)), np.ones(len(x))]

    f = ad.extract(paths, encode, batch_size=2)
    assert f.shape == (5, 4) and f.dtype == np.float32
    assert np.allclose(np.linalg.norm(f, axis=1), 1.0, atol=1e-5)


def test_threshold_keeps_false_positives_under_the_target():
    rng = np.random.default_rng(0)
    real = rng.uniform(0, 1, 1000)
    t = ad.pick_threshold(real, max_fpr=0.01)
    assert (real >= t).mean() <= 0.01
    assert (real >= t - 0.01).mean() > 0.005  # and not needlessly high


def test_probe_trained_here_scores_the_same_in_the_service():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 16)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
    clf = ad.fit_probe(X, y)
    w, b = clf.coef_[0], clf.intercept_[0]
    ours = ad.probe_scores(X, w, b)
    assert np.allclose(ours, clf.predict_proba(X)[:, 1], atol=1e-6)

    probe = ad.probe_json(w, b, 0.5, {}, "test", None)
    det = forensics.AiDetector.__new__(forensics.AiDetector)
    det.weights = np.asarray(probe["weights"], dtype=np.float32)
    det.bias, det.l2 = probe["bias"], probe["l2_normalize"]

    class Session:  # the encoder returns a fixed feature vector
        def __init__(self, f):
            self.f = f

        def run(self, outputs, feeds):
            return [self.f[None]]

    det.input, det.output, det.size = "pixel_values", None, 224
    for i in range(5):
        det.session = Session(X[i])
        assert det.score(Image.new("RGB", (224, 224))) == pytest.approx(
            ours[i], abs=1e-5
        )


def test_metrics():
    pytest.importorskip("sklearn")
    m = ad.metrics([0.9, 0.8, 0.2], [0.1, 0.3, 0.6], threshold=0.5)
    assert m == {
        "auc": 0.7778,
        "accuracy": 0.6667,
        "tpr": 0.6667,
        "fpr": 0.3333,
        "n": 6,
    }
