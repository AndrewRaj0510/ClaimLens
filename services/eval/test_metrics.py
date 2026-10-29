"""AP and IoU: hand-checked cases, then agreement with pycocotools itself on
random detections (boxes and masks)."""

import contextlib
import io

import numpy as np
import pytest

import metrics

NAMES = {0: "dent", 1: "scratch", 2: "crack"}


def image(gt_classes, det_classes, det_scores, ious):
    return metrics.ImageResult(
        np.array(gt_classes, int),
        np.array(det_classes, int),
        np.array(det_scores, float),
        np.array(ious, float).reshape(len(det_classes), len(gt_classes)),
    )


def test_box_iou():
    iou = metrics.box_iou(
        [[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]]
    )
    assert iou[0].tolist() == pytest.approx([1.0, 1 / 3, 0.0])


def test_mask_iou_matches_the_definition():
    a = np.zeros((2, 20, 20), bool)
    a[0, 0:10, 0:10] = True
    a[1, 15:20, 15:20] = True
    b = np.zeros((1, 20, 20), bool)
    b[0, 5:15, 0:10] = True
    iou = metrics.mask_iou(a, b)
    assert iou[0, 0] == pytest.approx(50 / 150)
    assert iou[1, 0] == 0.0
    assert metrics.mask_iou(a[:0], b).shape == (0, 1)


def test_perfect_detections_score_one():
    ims = [image([0, 1], [0, 1], [0.9, 0.8], [[1.0, 0.0], [0.0, 1.0]])]
    out = metrics.evaluate(ims, NAMES, operating_threshold=0.5)
    assert out["map50"] == 1.0 and out["map50_95"] == 1.0
    assert out["per_class"]["dent"]["precision"] == 1.0
    assert out["per_class"]["crack"] == {"n_gt": 0}  # no ground truth: not averaged in


def test_a_false_positive_ranked_first_halves_precision_at_full_recall():
    # Class 0: a false alarm (IoU 0.1) scored above the true detection.
    ims = [image([0], [0, 0], [0.9, 0.6], [[0.1], [0.8]])]
    out = metrics.evaluate(ims, {0: "dent"})
    # Precision 0.5 at recall 1 after the envelope -> 101-point AP is 0.5.
    assert out["per_class"]["dent"]["ap50"] == pytest.approx(0.5)


def test_a_duplicate_detection_is_a_false_positive():
    ims = [image([0], [0, 0], [0.9, 0.8], [[0.9], [0.85]])]
    out = metrics.evaluate(ims, {0: "dent"}, operating_threshold=0.5)
    row = out["per_class"]["dent"]
    assert row["ap50"] == pytest.approx(1.0)  # recall 1 is reached at precision 1
    assert row["precision"] == 0.5 and row["recall"] == 1.0


def test_a_missed_object_caps_recall():
    ims = [image([0, 0], [0], [0.9], [[0.9, 0.0]])]
    out = metrics.evaluate(ims, {0: "dent"}, operating_threshold=0.5)
    assert out["per_class"]["dent"]["recall"] == 0.5
    assert out["per_class"]["dent"]["ap50"] == pytest.approx(51 / 101, abs=1e-4)


def test_strict_thresholds_lower_map50_95():
    ims = [image([0], [0], [0.9], [[0.72]])]
    out = metrics.evaluate(ims, {0: "dent"})
    assert out["per_class"]["dent"]["ap50"] == 1.0
    # A match at IoU 0.72 counts at 0.50-0.70 (5 of 10 thresholds).
    assert out["per_class"]["dent"]["ap50_95"] == pytest.approx(0.5)
    assert out["mean_iou_matched"] == pytest.approx(0.72)


# ---- agreement with pycocotools ----


def random_scene(rng, n_images=12, size=(64, 48)):
    """Random ground truth and noisy detections, as boxes and masks."""
    w, h = size
    scenes = []
    for _ in range(n_images):
        gts, dets = [], []
        for _ in range(rng.integers(0, 5)):
            x0, y0 = rng.integers(0, w - 12), rng.integers(0, h - 12)
            x1, y1 = x0 + rng.integers(6, 12), y0 + rng.integers(6, 12)
            gts.append((int(rng.integers(0, 3)), (x0, y0, x1, y1)))
        for c, (x0, y0, x1, y1) in gts:
            if rng.random() < 0.8:  # most objects are found, with jitter
                j = rng.integers(-3, 4, size=4)
                cls = c if rng.random() < 0.9 else int(rng.integers(0, 3))
                dets.append(
                    (
                        cls,
                        float(rng.random()),
                        (x0 + j[0], y0 + j[1], x1 + j[2], y1 + j[3]),
                    )
                )
        for _ in range(rng.integers(0, 4)):  # false alarms
            x0, y0 = rng.integers(0, w - 10), rng.integers(0, h - 10)
            dets.append(
                (int(rng.integers(0, 3)), float(rng.random()), (x0, y0, x0 + 8, y0 + 8))
            )
        scenes.append((gts, dets))
    return scenes


def to_mask(box, size):
    w, h = size
    x0, y0, x1, y1 = (int(np.clip(v, 0, m)) for v, m in zip(box, (w, h, w, h)))
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    # Cut a corner so masks aren't just boxes.
    m[y0 : y0 + 2, x0 : x0 + 2] = False
    return m


def ours(scenes, size, kind):
    results = []
    for gts, dets in scenes:
        if kind == "bbox":
            ious = metrics.box_iou([d[2] for d in dets], [g[1] for g in gts])
        else:
            ious = metrics.mask_iou(
                [to_mask(d[2], size) for d in dets], [to_mask(g[1], size) for g in gts]
            )
        results.append(
            metrics.ImageResult(
                np.array([g[0] for g in gts], int),
                np.array([d[0] for d in dets], int),
                np.array([d[1] for d in dets], float),
                ious.reshape(len(dets), len(gts)),
            )
        )
    out = metrics.evaluate(results, NAMES)
    return out["map50"], out["map50_95"]


def theirs(scenes, size, kind):
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    w, h = size
    images, anns, dts = [], [], []
    for i, (gts, dets) in enumerate(scenes):
        images.append({"id": i, "width": w, "height": h})
        for c, b in gts:
            rle = mask_utils.encode(
                np.asfortranarray(to_mask(b, size).astype(np.uint8))
            )
            rle["counts"] = rle["counts"].decode()
            anns.append(
                {
                    "id": len(anns) + 1,
                    "image_id": i,
                    "category_id": c,
                    "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                    "area": float((b[2] - b[0]) * (b[3] - b[1])),
                    "segmentation": rle,
                    "iscrowd": 0,
                }
            )
        for c, s, b in dets:
            rle = mask_utils.encode(
                np.asfortranarray(to_mask(b, size).astype(np.uint8))
            )
            rle["counts"] = rle["counts"].decode()
            dts.append(
                {
                    "image_id": i,
                    "category_id": c,
                    "bbox": [
                        float(b[0]),
                        float(b[1]),
                        float(b[2] - b[0]),
                        float(b[3] - b[1]),
                    ],
                    "segmentation": rle,
                    "score": s,
                }
            )
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {
            "images": images,
            "annotations": anns,
            "categories": [{"id": c, "name": n} for c, n in NAMES.items()],
        }
        gt.createIndex()
        dt = gt.loadRes(dts) if dts else COCO()
        ev = COCOeval(gt, dt, kind)
        ev.params.maxDets = [100]
        ev.params.areaRng = [[0, 1e10]]
        ev.params.areaRngLbl = ["all"]
        ev.evaluate()
        ev.accumulate()
    p = ev.eval["precision"][:, :, :, 0, 0]  # T, R, K
    per_class_50 = [
        p[0, :, k].mean() for k in range(p.shape[2]) if (p[0, :, k] > -1).all()
    ]
    per_class_all = [
        p[:, :, k].mean() for k in range(p.shape[2]) if (p[:, :, k] > -1).all()
    ]
    return float(np.mean(per_class_50)), float(np.mean(per_class_all))


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("kind", ["bbox", "segm"])
def test_map_matches_pycocotools(seed, kind):
    pytest.importorskip("pycocotools")
    size = (64, 48)
    scenes = random_scene(np.random.default_rng(seed), size=size)
    got = ours(scenes, size, kind)
    want = theirs(scenes, size, kind)
    # Same boxes (pycocotools' x, y, w, h has no +1 either) and same masks, so
    # the numbers must agree up to our 4-decimal rounding.
    assert got == pytest.approx(want, abs=1e-4)
