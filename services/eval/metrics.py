"""Detection metrics, written out in numpy: IoU, COCO-style AP, operating point.

AP follows COCO's evaluation (pycocotools' COCOeval) step for step, so the
numbers mean what people expect and test_metrics.py can check them against
pycocotools itself:
- per image and class, detections are taken in score order (at most 100) and
  each is matched to the unmatched ground truth it overlaps most, if IoU >= t;
- per class, detections from all images are ranked by score, precision is made
  monotone (the "envelope"), and read at 101 recall points 0, 0.01, ..., 1;
- mAP is the mean over classes that have ground truth; mAP@0.5:0.95 also
  averages over IoU thresholds 0.50, 0.55, ..., 0.95.

Ultralytics' val (the numbers in docs/metrics.md) interpolates the curve
differently and matches at the prototype resolution, so expect small
differences, not equality.
"""

from dataclasses import dataclass

import numpy as np

IOU_THRESHOLDS = np.round(np.linspace(0.5, 0.95, 10), 2)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)
MAX_DETS = 100


def box_iou(a, b):
    """a: (N, 4), b: (M, 4) as x1, y1, x2, y2. Returns (N, M)."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def mask_iou(a, b):
    """a: (N, H, W), b: (M, H, W) boolean masks. Returns (N, M).

    Pairs are compared only inside their joint bounding region, since a
    prediction usually covers a small part of the photo."""
    n, m = len(a), len(b)
    out = np.zeros((n, m))
    if not n or not m:
        return out
    area_a = np.array([x.sum() for x in a], float)
    area_b = np.array([x.sum() for x in b], float)
    extent_a = [_extent(x) for x in a]
    extent_b = [_extent(x) for x in b]
    for i in range(n):
        if extent_a[i] is None:
            continue
        for j in range(m):
            if extent_b[j] is None:
                continue
            y0 = max(extent_a[i][0], extent_b[j][0])
            y1 = min(extent_a[i][1], extent_b[j][1])
            x0 = max(extent_a[i][2], extent_b[j][2])
            x1 = min(extent_a[i][3], extent_b[j][3])
            if y0 >= y1 or x0 >= x1:
                continue
            inter = np.logical_and(a[i][y0:y1, x0:x1], b[j][y0:y1, x0:x1]).sum()
            out[i, j] = inter / (area_a[i] + area_b[j] - inter)
    return out


def _extent(mask):
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any():
        return None
    y = np.flatnonzero(rows)
    x = np.flatnonzero(cols)
    return y[0], y[-1] + 1, x[0], x[-1] + 1


@dataclass
class ImageResult:
    """One image: ground-truth classes, detections, and their IoU matrix."""

    gt_classes: np.ndarray  # (G,)
    det_classes: np.ndarray  # (D,)
    det_scores: np.ndarray  # (D,)
    ious: np.ndarray  # (D, G)


def match(ious, scores, threshold):
    """COCOeval's greedy matching for one image and class. Returns, in score
    order: the detection indices, whether each is a true positive, and the IoU
    of its match (0 if none)."""
    order = np.argsort(-scores, kind="mergesort")[:MAX_DETS]
    matched = np.zeros(ious.shape[1], bool)
    tp = np.zeros(len(order), bool)
    best_iou = np.zeros(len(order))
    for k, d in enumerate(order):
        best, m = min(threshold, 1 - 1e-10), -1
        for g in range(ious.shape[1]):
            if matched[g] or ious[d, g] < best:
                continue
            best, m = ious[d, g], g
        if m >= 0:
            matched[m] = True
            tp[k] = True
            best_iou[k] = ious[d, m]
    return order, tp, best_iou


def average_precision(tp_by_image, scores_by_image, n_gt):
    """COCO 101-point AP for one class at one IoU threshold."""
    if n_gt == 0:
        return None
    scores = np.concatenate(scores_by_image) if scores_by_image else np.zeros(0)
    tps = np.concatenate(tp_by_image) if tp_by_image else np.zeros(0, bool)
    if not len(scores):
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    tps = tps[order]
    tp_sum = np.cumsum(tps)
    fp_sum = np.cumsum(~tps)
    recall = tp_sum / n_gt
    precision = tp_sum / (tp_sum + fp_sum + np.spacing(1))
    for i in range(len(precision) - 1, 0, -1):  # the precision envelope
        precision[i - 1] = max(precision[i - 1], precision[i])
    idx = np.searchsorted(recall, RECALL_POINTS, side="left")
    q = np.array([precision[i] if i < len(precision) else 0.0 for i in idx])
    return float(q.mean())


def evaluate(images, names, operating_threshold=None):
    """images: list of ImageResult; names: class id -> name.

    Returns mAP@0.5, mAP@0.5:0.95, per class AP, mean IoU of the matches at
    IoU 0.5, and precision/recall at the production confidence threshold."""
    per_class = {}
    for c, name in names.items():
        n_gt = int(sum((im.gt_classes == c).sum() for im in images))
        aps = []
        ious_at_50, op = [], {"tp": 0, "fp": 0}
        for t in IOU_THRESHOLDS:
            tps, scores = [], []
            for im in images:
                d = np.flatnonzero(im.det_classes == c)
                g = np.flatnonzero(im.gt_classes == c)
                if not len(d):
                    continue
                sub = im.ious[np.ix_(d, g)] if len(g) else np.zeros((len(d), 0))
                order, tp, best = match(sub, im.det_scores[d], t)
                tps.append(tp)
                scores.append(im.det_scores[d][order])
                if t == IOU_THRESHOLDS[0]:
                    ious_at_50 += list(best[tp])
                    if operating_threshold is not None:
                        keep = im.det_scores[d][order] >= operating_threshold
                        op["tp"] += int(tp[keep].sum())
                        op["fp"] += int((~tp[keep]).sum())
            aps.append(average_precision(tps, scores, n_gt))
        row = {"n_gt": n_gt}
        if n_gt:
            row |= {
                "ap50": round(aps[0], 4),
                "ap50_95": round(float(np.mean(aps)), 4),
                "mean_iou_matched": round(float(np.mean(ious_at_50)), 4)
                if ious_at_50
                else None,
            }
            if operating_threshold is not None:
                found = op["tp"] + op["fp"]
                row["precision"] = round(op["tp"] / found, 4) if found else None
                row["recall"] = round(op["tp"] / n_gt, 4)
        per_class[name] = row
    scored = [r for r in per_class.values() if r["n_gt"]]
    summary = {
        "map50": round(float(np.mean([r["ap50"] for r in scored])), 4)
        if scored
        else None,
        "map50_95": round(float(np.mean([r["ap50_95"] for r in scored])), 4)
        if scored
        else None,
        "per_class": per_class,
    }
    ious = [
        r["mean_iou_matched"] for r in scored if r.get("mean_iou_matched") is not None
    ]
    summary["mean_iou_matched"] = round(float(np.mean(ious)), 4) if ious else None
    return summary
