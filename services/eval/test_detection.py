"""Detection + cost eval on a synthetic YOLO-seg split (CarDD can't be in the
repo or CI). A fake detector answers from the labels, so the expected numbers
are known; the real model, if present, is only smoke-tested here."""

import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

import detection
import paths

NAMES = {
    0: "dent",
    1: "scratch",
    2: "crack",
    3: "glass shatter",
    4: "lamp broken",
    5: "tire flat",
}

# Per image: a background colour (the fake detector's key) and labelled polygons.
SCENES = {
    (200, 40, 40): [(0, [(0.1, 0.1), (0.4, 0.1), (0.4, 0.4), (0.1, 0.4)])],  # dent, 9%
    (40, 200, 40): [
        (
            1,
            [(0.5, 0.5), (0.9, 0.5), (0.9, 0.55), (0.5, 0.55)],
        ),  # scratch, 2% -> severe
        (
            4,
            [(0.05, 0.6), (0.15, 0.6), (0.15, 0.7), (0.05, 0.7)],
        ),  # lamp, 1% -> moderate
    ],
}


def write_split(root, size=(800, 600)):
    (root / "images" / "test").mkdir(parents=True)
    (root / "labels" / "test").mkdir(parents=True)
    for i, (colour, labels) in enumerate(SCENES.items()):
        img = Image.new("RGB", size, colour)
        d = ImageDraw.Draw(img)
        for _, pts in labels:
            d.polygon([(x * size[0], y * size[1]) for x, y in pts], fill=(90, 90, 90))
        img.save(root / "images" / "test" / f"{i:03d}.jpg", quality=95)
        lines = [
            f"{c} " + " ".join(f"{v:.4f}" for p in pts for v in p) for c, pts in labels
        ]
        (root / "labels" / "test" / f"{i:03d}.txt").write_text("\n".join(lines))


class FakeDetector:
    """Returns the labelled polygons as detections, optionally changed."""

    names = NAMES

    def __init__(self, drop=(), extra=None, scale=1.0, score=0.9):
        self.drop, self.extra, self.scale, self.score = set(drop), extra, scale, score

    def __call__(self, image, conf=0.25, iou=0.7):
        from detector import Detection

        colour = image.getpixel((2, 2))
        key = min(SCENES, key=lambda c: sum(abs(a - b) for a, b in zip(c, colour)))
        labels = [lb for lb in SCENES[key] if lb[0] not in self.drop]
        if self.extra and key == self.extra[0]:
            labels = labels + [self.extra[1]]
        out = []
        for cls, pts in labels:
            cx = sum(x for x, _ in pts) / len(pts)
            cy = sum(y for _, y in pts) / len(pts)
            pts = [
                (cx + (x - cx) * self.scale, cy + (y - cy) * self.scale) for x, y in pts
            ]
            mask = detection.rasterise(pts, image.size)
            box = tuple(float(v) for v in detection.polygon_box(pts, image.size))
            if self.score >= conf:
                out.append(Detection(cls, self.score, box, mask))
        return out


def run(tmp_path, detector):
    write_split(tmp_path)
    samples = ((p.read_bytes(), lb) for p, lb in detection.load_split(tmp_path))
    return detection.evaluate_split(detector, samples, detection.pricing_tables())


def test_labels_are_read_and_rasterised(tmp_path):
    write_split(tmp_path)
    items = list(detection.load_split(tmp_path))
    assert len(items) == 2 and [c for c, _ in items[1][1]] == [1, 4]
    mask = detection.rasterise(items[0][1][0][1], (100, 100))
    assert mask.sum() == pytest.approx(31 * 31, rel=0.1)


def test_perfect_detections(tmp_path):
    out = run(tmp_path, FakeDetector())
    assert out["box"]["map50"] == 1.0 and out["mask"]["map50"] == 1.0
    assert out["mask"]["mean_iou_matched"] > 0.95
    cost = out["cost"]
    assert cost["mae"] == 0 and cost["damage_list_exact"] == 1.0
    assert cost["true_cost_in_quoted_range"] == 1.0
    assert cost["per_class"]["scratch"]["correct"] == 1
    # The 9% dent and the 2% scratch are both severe; the 1% lamp is moderate.
    assert cost["severity_confusion"]["severe"]["severe"] == 2
    assert cost["severity_confusion"]["moderate"]["moderate"] == 1


def test_a_missed_damage_costs_its_whole_repair(tmp_path):
    out = run(tmp_path, FakeDetector(drop={4}))
    lamp = out["cost"]["per_class"]["lamp broken"]
    assert lamp["missed"] == 1 and lamp["missed_rate"] == 1.0
    tables = detection.pricing_tables()
    _, lamp_cost, _, _ = detection.price(
        tables, [{"damage_class": "lamp broken", "severity": "moderate"}]
    )
    assert lamp["mean_abs_error"] == pytest.approx(lamp_cost)
    assert out["cost"]["bias"] == pytest.approx(-lamp_cost / 2)  # one of two photos
    assert out["mask"]["per_class"]["lamp broken"]["recall"] == 0.0
    assert out["cost"]["severity_confusion"]["moderate"]["missed"] == 1


def test_spurious_damage_and_a_shrunken_mask(tmp_path):
    extra = ((200, 40, 40), (2, [(0.6, 0.6), (0.7, 0.6), (0.7, 0.7), (0.6, 0.7)]))
    out = run(tmp_path, FakeDetector(extra=extra, scale=0.5))
    crack = out["cost"]["per_class"]["crack"]
    assert crack["spurious"] == 1 and crack["photos_with_it"] == 0
    # Half the side = a quarter of the area: the 9% dent becomes moderate.
    dent = out["cost"]["per_class"]["dent"]
    assert dent["severity_off"] == 1 and dent["severity_agreement"] == 0.0
    assert out["box"]["per_class"]["dent"]["ap50"] == 0.0  # box IoU 0.25 < 0.5


def test_the_eval_runs_the_production_threshold(tmp_path):
    out = run(tmp_path, FakeDetector(score=0.3))
    # 0.3 passes the vision service's 0.25 but not the agent's 0.35 floor.
    assert out["operating_confidence"] == 0.35
    assert out["mask"]["map50"] == 1.0  # AP doesn't depend on a threshold
    assert out["mask"]["per_class"]["dent"]["recall"] == 0.0
    assert out["cost"]["per_class"]["dent"]["missed"] == 1


MODEL = os.getenv("VISION_TEST_MODEL") or str(paths.DATA / "models" / "best.onnx")


@pytest.mark.skipif(not os.path.exists(MODEL), reason="no trained model")
def test_real_model_runs_end_to_end(tmp_path):
    out = detection.run(_split(tmp_path), MODEL)
    assert out["images"] == 2 and len(out["model_sha256"]) == 64
    assert set(out["mask"]["per_class"]) == set(NAMES.values())


def _split(tmp_path):
    write_split(tmp_path)
    return tmp_path


def test_pricing_matches_the_generator():
    """The eval prices with the pricing server's code; the synthetic generator
    has its own copy of the formula. They must agree, or the eval's "true" cost
    isn't what the database would say."""
    import generate_synthetic as synthetic

    tables = detection.pricing_tables()
    for cls, sev in synthetic.REPAIR_OPERATIONS:
        _, total, _, _ = detection.price(
            tables, [{"damage_class": cls, "severity": sev}]
        )
        assert total == pytest.approx(
            synthetic.repair_cost(cls, sev, detection.SEGMENT, detection.STATE),
            abs=0.01,
        )
    assert np.isclose(detection.price(tables, [])[1], 0)
