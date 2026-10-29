"""Damage detection and cost-estimate eval on a labelled YOLO-seg split.

    python run.py detection --data /path/to/cardd_yolo --model data/models/best.onnx

The split is CarDD's test set as training/convert_cardd.py writes it
(images/test/*.jpg + labels/test/*.txt, one polygon per line). CarDD can't be
committed or put in CI, so this suite runs where the data is: Colab
(training/eval_cardd.ipynb) or a local copy. Only the summary is kept.

Two measurements, both through the services' own code:

1. Detection. services/vision's ONNX post-processing at conf 0.001 (as
   detector evals do, so the whole precision-recall curve is seen), on a copy
   whose longest side is 640 px (the model's input size, so nothing is lost),
   with the labels rasterised at the same size. Box and mask AP (metrics.py),
   mean IoU of the matches, and precision/recall at the confidence the agent
   really uses.

2. Cost error: what detection errors cost in dollars. Each photo is one claim.
   - predicted: the production path. analysis.analyse() (the vision service's
     confidence threshold and resizing), then the agent's damage.aggregate()
     (its own confidence floor, one damage per class, severity from area %),
     then the pricing server's estimate.
   - "true": the same severity rule and the same pricing applied to the
     labelled masks.
   So this isolates the error the vision pipeline adds. It is not the error
   against real repair invoices: CarDD has none, and the prices are synthetic.
   The vehicle is fixed (midsize, ME) so photos are comparable.
"""

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import metrics
import paths

EVAL_SIDE = 640
EVAL_CONF = 0.001
SEGMENT, STATE = "midsize", "ME"
SEVERITIES = ("minor", "moderate", "severe")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


# ---- data ----


def read_labels(path):
    """YOLO-seg label file -> [(class_id, [(x, y), ...] normalised)]."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().split("\n"):
        values = line.split()
        if len(values) < 7:  # a class and at least three points
            continue
        coords = [float(v) for v in values[1:]]
        out.append((int(values[0]), list(zip(coords[::2], coords[1::2]))))
    return out


def load_split(data_dir, split="test"):
    """Yields (image path, labels) for a split in the convert_cardd layout."""
    data_dir = Path(data_dir)
    images = sorted(
        p
        for p in (data_dir / "images" / split).iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES
    )
    for p in images:
        yield p, read_labels(data_dir / "labels" / split / f"{p.stem}.txt")


def open_upright(data):
    # Same decoding as the vision service: EXIF rotation applied.
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")


def rasterise(points, size):
    w, h = size
    canvas = Image.new("1", size, 0)
    ImageDraw.Draw(canvas).polygon([(x * w, y * h) for x, y in points], fill=1)
    return np.asarray(canvas, dtype=bool)


def polygon_box(points, size):
    w, h = size
    xs, ys = [x * w for x, _ in points], [y * h for _, y in points]
    return [min(xs), min(ys), max(xs), max(ys)]


# ---- cost ----


def pricing_tables():
    """The pricing server's Tables, from the same seeded source the database is
    loaded from (scripts/generate_synthetic.py), so no database is needed."""
    import generate_synthetic as synthetic

    pricing = paths.pricing()
    t = synthetic.pricing_table()
    return pricing.Tables(
        labour_rates={k: float(v) for k, v in t["labour_rates_per_hour"].items()},
        multipliers={k: float(v) for k, v in t["segment_parts_multiplier"].items()},
        operations={
            (o["damage_class"], o["severity"]): (
                o["operation"],
                float(o["parts_usd"]),
                float(o["labour_hours"]),
                float(o["paint_hours"]),
            )
            for o in t["operations"]
        },
        paint_materials_per_hour=float(t["paint_materials_per_hour"]),
    )


def price(tables, damages):
    """{class: line cost}, total, low, high for a list of aggregated damages."""
    if not damages:
        return {}, 0.0, 0.0, 0.0
    pricing = paths.pricing()
    est = pricing.estimate(
        tables,
        [
            pricing.Damage(damage_class=d["damage_class"], severity=d["severity"])
            for d in damages
        ],
        SEGMENT,
        STATE,
    )
    lines = {i.damage_class: i.cost for i in est.line_items}
    return lines, est.total, est.low, est.high


def compare(true_damages, predicted_damages, tables):
    """One photo's cost comparison, with each class's outcome."""
    t_lines, t_total, _, _ = price(tables, true_damages)
    p_lines, p_total, p_low, p_high = price(tables, predicted_damages)
    t_sev = {d["damage_class"]: d["severity"] for d in true_damages}
    p_sev = {d["damage_class"]: d["severity"] for d in predicted_damages}
    classes = {}
    for c in sorted(set(t_sev) | set(p_sev)):
        if c not in p_sev:
            outcome = "missed"
        elif c not in t_sev:
            outcome = "spurious"
        elif t_sev[c] == p_sev[c]:
            outcome = "correct"
        else:
            outcome = "severity_off"
        classes[c] = {
            "outcome": outcome,
            "true": t_sev.get(c),
            "predicted": p_sev.get(c),
            "error": round(p_lines.get(c, 0.0) - t_lines.get(c, 0.0), 2),
        }
    return {
        "true_total": t_total,
        "predicted_total": p_total,
        "predicted_low": p_low,
        "predicted_high": p_high,
        "classes": classes,
    }


def summarise_cost(rows, class_names):
    errors = np.array([r["predicted_total"] - r["true_total"] for r in rows])
    true = np.array([r["true_total"] for r in rows])
    priced = true > 0
    ape = np.abs(errors[priced]) / true[priced]
    in_range = [
        r["predicted_low"] <= r["true_total"] <= r["predicted_high"]
        for r in rows
        if r["true_total"] > 0
    ]
    exact = [
        all(c["outcome"] == "correct" for c in r["classes"].values()) for r in rows
    ]
    per_class = {}
    for name in class_names:
        cells = [r["classes"][name] for r in rows if name in r["classes"]]
        labelled = [c for c in cells if c["outcome"] != "spurious"]
        count = {
            k: sum(c["outcome"] == k for c in cells)
            for k in ("correct", "severity_off", "missed", "spurious")
        }
        per_class[name] = {
            "photos_with_it": len(labelled),
            **count,
            "missed_rate": round(count["missed"] / len(labelled), 3)
            if labelled
            else None,
            "severity_agreement": round(
                count["correct"] / (count["correct"] + count["severity_off"]), 3
            )
            if count["correct"] + count["severity_off"]
            else None,
            "mean_abs_error": round(float(np.mean([abs(c["error"]) for c in cells])), 2)
            if cells
            else None,
            "total_abs_error": round(sum(abs(c["error"]) for c in cells), 2),
        }
    total_abs = sum(v["total_abs_error"] for v in per_class.values()) or 1.0
    for v in per_class.values():
        v["share_of_error"] = round(v["total_abs_error"] / total_abs, 3)
    severity = {
        t: {p: 0 for p in (*SEVERITIES, "missed")} for t in SEVERITIES
    }  # true severity -> predicted
    for r in rows:
        for c in r["classes"].values():
            if c["true"]:
                severity[c["true"]][c["predicted"] or "missed"] += 1
    return {
        "photos": len(rows),
        "vehicle": f"{SEGMENT}, {STATE}",
        "mean_true_cost": round(float(true.mean()), 2) if len(rows) else None,
        "mae": round(float(np.abs(errors).mean()), 2) if len(rows) else None,
        "bias": round(float(errors.mean()), 2) if len(rows) else None,
        "median_ape": round(float(np.median(ape)), 3) if len(ape) else None,
        "within_20pct": round(float(np.mean(ape <= 0.2)), 3) if len(ape) else None,
        "true_cost_in_quoted_range": round(float(np.mean(in_range)), 3)
        if in_range
        else None,
        "damage_list_exact": round(float(np.mean(exact)), 3) if exact else None,
        "per_class": per_class,
        "severity_confusion": severity,
    }


# ---- the run ----


def evaluate_split(detector, samples, tables, limit=None):
    """detector: services/vision Detector (or a stand-in with the same call and
    .names). samples: (image bytes, labels) pairs. Returns the summary."""
    import analysis
    import damage

    names = dict(detector.names)
    box_results, mask_results, cost_rows = [], [], []
    for i, (data, labels) in enumerate(samples):
        if limit is not None and i >= limit:
            break
        image = open_upright(data)
        work = image.copy()
        work.thumbnail((EVAL_SIDE, EVAL_SIDE))
        gt_classes = np.array([c for c, _ in labels], int)
        gt_boxes = np.array([polygon_box(p, work.size) for _, p in labels]).reshape(
            -1, 4
        )
        gt_masks = [rasterise(p, work.size) for _, p in labels]

        dets = detector(work, conf=EVAL_CONF)
        det_classes = np.array([d.class_id for d in dets], int)
        det_scores = np.array([d.confidence for d in dets], float)
        det_boxes = np.array([d.box for d in dets]).reshape(-1, 4)
        box_results.append(
            metrics.ImageResult(
                gt_classes,
                det_classes,
                det_scores,
                metrics.box_iou(det_boxes, gt_boxes),
            )
        )
        mask_results.append(
            metrics.ImageResult(
                gt_classes,
                det_classes,
                det_scores,
                metrics.mask_iou([d.mask for d in dets], gt_masks),
            )
        )

        # Cost: the production path on the photo as uploaded.
        response = analysis.analyse(detector, data)
        predicted = damage.aggregate(
            [{"detections": [d.model_dump() for d in response.detections]}]
        )
        pixels = work.width * work.height
        true = damage.aggregate(
            [
                {
                    "detections": [
                        {
                            "class_name": names[c],
                            "area_pct": 100 * m.sum() / pixels,
                            "confidence": 1.0,
                        }
                        for c, m in zip(gt_classes, gt_masks)
                    ]
                }
            ]
        )
        cost_rows.append(compare(true, predicted, tables))

    # The confidence the agent effectively uses: the vision service's
    # threshold, then damage.aggregate's floor on top.
    operating = max(analysis.CONF_THRESHOLD, damage.MIN_CONFIDENCE)
    return {
        "images": len(box_results),
        "eval_side": EVAL_SIDE,
        "operating_confidence": operating,
        "box": metrics.evaluate(box_results, names, operating),
        "mask": metrics.evaluate(mask_results, names, operating),
        "cost": summarise_cost(cost_rows, list(names.values())),
    }


def run(data_dir, model_path, split="test", limit=None):
    from detector import Detector

    detector = Detector(str(model_path))
    samples = ((p.read_bytes(), labels) for p, labels in load_split(data_dir, split))
    out = evaluate_split(detector, samples, pricing_tables(), limit)
    out["split"] = split
    out["model_sha256"] = sha256(model_path)
    return out


def sha256(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
