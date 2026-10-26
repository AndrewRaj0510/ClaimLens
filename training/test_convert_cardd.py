import json

import pytest

from convert_cardd import SPLITS, coco_polygon_to_yolo, convert, merge_polygons

CATEGORIES = [
    {"id": 2, "name": "scratch"},
    {"id": 1, "name": "dent"},
]


def make_cardd(root, annotations_per_split):
    """Build a tiny fake CarDD_COCO tree. Images are dummy bytes: the converter never decodes them."""
    (root / "annotations").mkdir(parents=True)
    for split in SPLITS:
        images, anns = annotations_per_split[split]
        (root / f"{split}2017").mkdir()
        for image in images:
            (root / f"{split}2017" / image["file_name"]).write_bytes(b"fake")
        coco = {"images": images, "annotations": anns, "categories": CATEGORIES}
        (root / "annotations" / f"instances_{split}2017.json").write_text(
            json.dumps(coco)
        )


def image(id_, name, w=100, h=200):
    return {"id": id_, "file_name": name, "width": w, "height": h}


def ann(id_, image_id, cat, seg, iscrowd=0):
    return {
        "id": id_,
        "image_id": image_id,
        "category_id": cat,
        "segmentation": seg,
        "iscrowd": iscrowd,
    }


@pytest.fixture
def dataset(tmp_path):
    src = tmp_path / "CarDD_COCO"
    square = [[10, 20, 50, 20, 50, 100]]
    make_cardd(
        src,
        {
            "train": (
                [image(1, "a.jpg"), image(2, "b.jpg")],
                [
                    ann(1, 1, 1, square),
                    ann(2, 1, 2, {"counts": "xyz", "size": [200, 100]}, iscrowd=1),
                ],
            ),
            "val": ([image(3, "c.jpg")], [ann(3, 3, 2, square)]),
            "test": ([image(4, "d.jpg")], []),
        },
    )
    out = tmp_path / "out"
    names, stats = convert(src, out)
    return out, names, stats


def test_class_ids_are_contiguous_and_sorted_by_coco_id(dataset):
    _, names, _ = dataset
    assert names == ["dent", "scratch"]


def test_label_is_normalised_polygon(dataset):
    out, _, _ = dataset
    label = (out / "labels" / "train" / "a.txt").read_text().splitlines()
    assert label == ["0 0.100000 0.100000 0.500000 0.100000 0.500000 0.500000"]


def test_crowd_annotation_is_skipped(dataset):
    _, _, stats = dataset
    assert stats["train"]["skipped_crowd"] == 1
    assert stats["train"]["instances"] == 1


def test_image_without_damage_gets_empty_label(dataset):
    out, _, _ = dataset
    assert (out / "labels" / "train" / "b.txt").read_text() == ""
    assert (out / "labels" / "test" / "d.txt").read_text() == ""


def test_images_are_copied_into_split_folders(dataset):
    out, _, _ = dataset
    for split, name in [("train", "a.jpg"), ("val", "c.jpg"), ("test", "d.jpg")]:
        assert (out / "images" / split / name).exists()


def test_data_yaml(dataset):
    out, _, _ = dataset
    text = (out / "data.yaml").read_text()
    assert "train: images/train" in text
    assert "test: images/test" in text
    assert '  0: "dent"' in text
    assert '  1: "scratch"' in text


def test_missing_split_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert(tmp_path, tmp_path / "out")


def test_coords_are_clipped_to_image():
    coords = coco_polygon_to_yolo([[-5, 0, 150, 0, 150, 250]], width=100, height=200)
    assert coords == [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]


def test_degenerate_polygon_is_dropped():
    assert coco_polygon_to_yolo([[1, 2, 3, 4]], width=10, height=10) == []


def test_merge_single_part_is_unchanged():
    part = [[0, 0], [1, 0], [1, 1]]
    assert merge_polygons([part]) == part


def test_merge_bridges_to_nearest_point_and_back():
    a = [[0, 0], [1, 0], [1, 1]]
    b = [[10, 10], [5, 5], [10, 5]]  # [5, 5] is nearest to a's anchor [0, 0]
    merged = merge_polygons([a, b])
    assert merged == a + [[0, 0]] + [[5, 5], [10, 5], [10, 10]] + [[5, 5], [0, 0]]
    # every point of every part survives
    assert all(p in merged for p in a + b)
