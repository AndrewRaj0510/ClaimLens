import io
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import analysis
import main
from detector import Detection

FIXTURE = Path(__file__).parent / "fixtures" / "panel.jpg"


class FakeDetector:
    """Stands in for the ONNX model: returns two fixed, partly overlapping detections."""

    def __init__(self):
        self.names = {0: "dent", 1: "scratch"}

    def __call__(self, image, conf=0.25):
        w, h = image.size
        dent = np.zeros((h, w), dtype=bool)
        dent[:, : w // 10] = True  # 10% of the image
        scratch = np.zeros((h, w), dtype=bool)
        scratch[:, : w // 20] = True  # 5%, entirely inside the dent
        return [
            Detection(0, 0.91, (10.0, 10.0, 110.0, 60.0), dent),
            Detection(1, 0.55, (0.0, 0.0, 50.0, 50.0), scratch),
        ]


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def fake_model(client):
    main.app.state.detector = FakeDetector()
    yield client
    main.app.state.detector = None


def jpeg(size=(320, 240), exif=None):
    buf = io.BytesIO()
    kwargs = {"exif": exif} if exif else {}
    Image.new("RGB", size, (90, 90, 90)).save(buf, "JPEG", **kwargs)
    return buf.getvalue()


def post(client, data, name="car.jpg", ctype="image/jpeg"):
    return client.post("/detect", files={"file": (name, data, ctype)})


def test_health_reports_missing_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "vision", "model_loaded": False}


def test_detect_without_model_is_503(client):
    assert post(client, FIXTURE.read_bytes()).status_code == 503


def test_detect_returns_detections_and_area(fake_model):
    r = post(fake_model, FIXTURE.read_bytes())
    assert r.status_code == 200
    body = r.json()
    assert (body["width"], body["height"]) == (320, 240)
    assert [d["class_name"] for d in body["detections"]] == ["dent", "scratch"]
    assert body["detections"][0]["confidence"] == 0.91
    assert body["detections"][0]["area_pct"] == 10.0
    assert body["detections"][1]["area_pct"] == 5.0
    # union: the scratch lies inside the dent, so it's not double counted
    assert body["damage_area_pct"] == 10.0


def test_large_photo_runs_downscaled_but_boxes_are_in_original_pixels(fake_model):
    r = post(fake_model, jpeg((2560, 1920)))  # downscaled 2x to MAX_SIDE=1280
    body = r.json()
    assert (body["width"], body["height"]) == (2560, 1920)
    assert body["detections"][0]["box"] == {
        "x1": 20.0,
        "y1": 20.0,
        "x2": 220.0,
        "y2": 120.0,
    }


def test_exif_rotation_is_applied(fake_model):
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 degrees clockwise when displaying
    body = post(fake_model, jpeg((300, 200), exif=exif)).json()
    assert (body["width"], body["height"]) == (200, 300)


def test_garbage_bytes_are_rejected(fake_model):
    r = post(fake_model, b"definitely not an image")
    assert r.status_code == 400


def test_oversized_upload_is_rejected(fake_model, monkeypatch):
    monkeypatch.setattr(analysis, "MAX_UPLOAD_MB", 0.001)  # ~1 KB
    assert post(fake_model, FIXTURE.read_bytes()).status_code == 413


@pytest.mark.skipif(
    not os.getenv("VISION_TEST_MODEL"),
    reason="set VISION_TEST_MODEL=path/to/best.onnx to run against the real model",
)
def test_real_model_on_fixture(client):
    from detector import Detector

    detector = Detector(os.environ["VISION_TEST_MODEL"])
    main.app.state.detector = detector
    try:
        r = post(client, FIXTURE.read_bytes())
    finally:
        main.app.state.detector = None
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["damage_area_pct"] <= 100
    for d in body["detections"]:
        assert d["class_name"] in detector.names.values()
        assert 0 <= d["confidence"] <= 1


def test_decompression_bomb_between_1x_and_2x_the_limit_is_rejected(fake_model):
    # Pillow only raises above 2x MAX_IMAGE_PIXELS; this 64 MP PNG is ~62 KB on
    # disk and would decode to ~190 MB of RGB without the explicit check.
    buf = io.BytesIO()
    Image.new("L", (8000, 8000)).save(buf, "PNG")
    r = post(fake_model, buf.getvalue(), name="bomb.png", ctype="image/png")
    assert r.status_code == 400
    assert r.json()["detail"] == "Image has too many pixels"


def test_masks_come_back_as_png_cropped_to_the_box(fake_model):
    import base64

    body = post(fake_model, FIXTURE.read_bytes()).json()
    dent = body["detections"][0]  # box (10, 10)-(110, 60); mask = left 32 columns
    mask = Image.open(io.BytesIO(base64.b64decode(dent["mask_png"])))
    assert mask.mode == "1"
    assert mask.size == (100, 50)
    pixels = np.asarray(mask)
    assert pixels[:, :22].all()  # columns 10-31 of the photo
    assert not pixels[:, 22:].any()


def test_mask_png_is_none_when_the_box_holds_no_mask():
    mask = np.zeros((40, 40), dtype=bool)
    mask[:5, :5] = True
    assert analysis.mask_png(mask, (20.0, 20.0, 30.0, 30.0)) is None
    assert analysis.mask_png(mask, (30.0, 30.0, 30.0, 35.0)) is None  # zero width
