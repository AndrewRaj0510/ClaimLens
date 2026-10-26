"""Convert CarDD (COCO instance-segmentation format) to the YOLO-seg format.

Expected CarDD layout (the official CarDD_COCO release):

    CarDD_COCO/
      annotations/instances_{train,val,test}2017.json
      {train,val,test}2017/*.jpg

Output layout (what Ultralytics expects):

    out/
      data.yaml
      images/{train,val,test}/*.jpg
      labels/{train,val,test}/*.txt   one line per instance: "cls x1 y1 x2 y2 ..." (normalised 0-1)

Stdlib only, so it can be unit-tested without installing torch/ultralytics.

Usage:
    python convert_cardd.py --src /path/to/CarDD_COCO --out /content/cardd_yolo
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

SPLITS = ("train", "val", "test")


def merge_polygons(parts):
    """Join a multi-part COCO polygon into one YOLO polygon.

    YOLO-seg allows one polygon per instance, but COCO splits an object into
    several parts when it is occluded (e.g. a scratch interrupted by a door gap).
    Writing each part as its own line would turn one damage into several and
    distort mAP. Instead we connect the parts with zero-width "bridges": walk
    part 0, jump from its first point to the nearest point of the next part,
    walk that part, and jump back. When rasterised, the bridges cover no area.

    `parts` is a list of point lists [[x, y], ...]. Returns one point list.
    """
    if len(parts) == 1:
        return parts[0]
    anchor = parts[0][0]
    merged = parts[0] + [anchor]
    for part in parts[1:]:
        k = min(
            range(len(part)),
            key=lambda i: (part[i][0] - anchor[0]) ** 2 + (part[i][1] - anchor[1]) ** 2,
        )
        rotated = part[k:] + part[:k]
        merged += rotated + [rotated[0], anchor]
    return merged


def coco_polygon_to_yolo(segmentation, width, height):
    """Turn a COCO polygon segmentation into a flat list of normalised coords."""
    parts = [
        [[seg[i], seg[i + 1]] for i in range(0, len(seg) - 1, 2)]
        for seg in segmentation
        if len(seg) >= 6  # a polygon needs at least 3 points
    ]
    if not parts:
        return []
    coords = []
    for x, y in merge_polygons(parts):
        coords.append(min(max(x / width, 0.0), 1.0))
        coords.append(min(max(y / height, 0.0), 1.0))
    return coords


def build_class_map(categories):
    """Map COCO category ids (CarDD uses 1-6) to contiguous YOLO ids (0-5)."""
    ordered = sorted(categories, key=lambda c: c["id"])
    return {c["id"]: i for i, c in enumerate(ordered)}, [c["name"] for c in ordered]


def convert_split(ann_file, image_dir, out_dir, split, class_map):
    """Convert one split. Returns a Counter of stats for logging."""
    coco = json.loads(Path(ann_file).read_text())
    stats = Counter()

    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    img_out = Path(out_dir) / "images" / split
    lbl_out = Path(out_dir) / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for image in coco["images"]:
        src = Path(image_dir) / image["file_name"]
        if not src.exists():
            stats["missing_images"] += 1
            continue
        shutil.copy2(src, img_out / src.name)
        stats["images"] += 1

        lines = []
        for ann in anns_by_image.get(image["id"], []):
            # Crowd regions are RLE masks over groups of objects; COCO eval
            # ignores them, so we leave them out of training too.
            if ann.get("iscrowd", 0) or not isinstance(ann["segmentation"], list):
                stats["skipped_crowd"] += 1
                continue
            coords = coco_polygon_to_yolo(
                ann["segmentation"], image["width"], image["height"]
            )
            if not coords:
                stats["skipped_empty"] += 1
                continue
            cls = class_map[ann["category_id"]]
            lines.append(f"{cls} " + " ".join(f"{c:.6f}" for c in coords))
            stats["instances"] += 1

        # An empty label file marks a background image (no damage), which is valid.
        (lbl_out / f"{src.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
    return stats


def write_data_yaml(out_dir, names):
    """Write the Ultralytics dataset config. json.dumps quotes names safely (YAML is a JSON superset)."""
    out = Path(out_dir)
    lines = [
        f"path: {json.dumps(str(out.resolve()))}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *[f"  {i}: {json.dumps(name)}" for i, name in enumerate(names)],
    ]
    (out / "data.yaml").write_text("\n".join(lines) + "\n")


def convert(src, out):
    src, out = Path(src), Path(out)
    class_map, names = None, None
    all_stats = {}
    for split in SPLITS:
        ann_file = src / "annotations" / f"instances_{split}2017.json"
        image_dir = src / f"{split}2017"
        if not ann_file.exists():
            raise FileNotFoundError(f"Missing annotation file: {ann_file}")
        if class_map is None:
            # All CarDD splits share the same categories, so read them once.
            class_map, names = build_class_map(
                json.loads(ann_file.read_text())["categories"]
            )
        all_stats[split] = convert_split(ann_file, image_dir, out, split, class_map)
    write_data_yaml(out, names)
    return names, all_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True, help="Path to the CarDD_COCO folder")
    parser.add_argument(
        "--out", required=True, help="Output folder for the YOLO dataset"
    )
    args = parser.parse_args()

    names, all_stats = convert(args.src, args.out)
    print(f"classes: {names}")
    for split, stats in all_stats.items():
        print(f"{split:>5}: {dict(stats)}")


if __name__ == "__main__":
    main()
