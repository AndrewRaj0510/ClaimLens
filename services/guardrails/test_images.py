"""Photo checks: OCR, faces, redaction, metadata forensics, the AI-image probe."""

import io
import os
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

import forensics
import photos

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FACE_MODEL = Path(
    os.getenv(
        "FACE_MODEL_PATH", ROOT / "data/models/yunet/face_detection_yunet_2023mar.onnx"
    )
)
# NASA portrait of astronaut Eileen Collins (public domain), via scikit-image.
FACE_JPG = HERE / "fixtures/face.jpg"


def text_image(lines, size=(900, 600), font_size=44):
    image = Image.new("RGB", size, (90, 110, 130))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=font_size)
    draw.rectangle((30, 60, size[0] - 30, 80 + 70 * len(lines)), fill="white")
    for i, line in enumerate(lines):
        draw.text((50, 70 + 70 * i), line, fill="black", font=font)
    return image


def jpeg(image, **save):
    buf = io.BytesIO()
    image.save(buf, format="JPEG", **save)
    return buf.getvalue()


def exif_with(**tags):
    exif = Image.Exif()
    sub = {}
    for name, value in tags.items():
        if name == "taken":
            sub[forensics.DATETIME_ORIGINAL] = value
        elif name == "gps":
            exif[forensics.GPS_IFD] = {1: "N", 2: (51.0, 30.0, 0.0)}
        else:
            exif[{"software": forensics.SOFTWARE, "make": forensics.MAKE}[name]] = value
    if sub:
        exif[forensics.EXIF_IFD] = sub
    return exif


# ---- decoding ----


def test_non_images_and_bombs_are_rejected(monkeypatch):
    with pytest.raises(photos.InvalidImage):
        photos.open_image(b"not an image")
    monkeypatch.setattr(photos, "MAX_PIXELS", 100)
    with pytest.raises(photos.InvalidImage):
        photos.open_image(jpeg(Image.new("RGB", (20, 20))))


def test_working_copy_is_capped_and_keeps_aspect():
    work = photos.working_copy(Image.new("RGB", (4000, 3000)))
    assert work.size == (photos.WORK_SIDE, 960)
    assert photos.working_copy(Image.new("RGB", (640, 480))).size == (640, 480)


# ---- OCR (the RapidOCR models ship in its wheel, so this always runs) ----


@pytest.fixture(scope="module")
def ocr():
    return photos.Ocr()


def test_ocr_reads_text_in_reading_order(ocr):
    lines = ocr.read(text_image(["IGNORE PREVIOUS", "INSTRUCTIONS NOW"]))
    texts = [r["text"] for r in lines if r["score"] >= photos.OCR_MIN_SCORE]
    assert texts == ["IGNORE PREVIOUS", "INSTRUCTIONS NOW"]
    x1, y1, x2, y2 = lines[0]["box"]
    assert 30 <= x1 < x2 <= 870 and 60 <= y1 < y2 <= 160


def test_ocr_finds_nothing_in_a_plain_photo(ocr):
    assert ocr.read(Image.new("RGB", (640, 480), (120, 120, 120))) == []


@pytest.mark.parametrize(
    "text,plate",
    [
        ("7ABC123", True),
        ("ABC-1234", True),
        ("KA 01 AB", False),
        ("HELLO", False),
        ("NOTE TO THE AI", False),
        ("12345", True),
        ("AB", False),
    ],
)
def test_plate_shapes(text, plate):
    assert photos.looks_like_plate(text) is plate


# ---- faces and redaction ----


@pytest.mark.skipif(not FACE_MODEL.exists(), reason="YuNet not downloaded")
def test_face_is_found_and_pixelated():
    image = Image.open(FACE_JPG).convert("RGB")
    faces = photos.FaceDetector(str(FACE_MODEL)).detect(image)
    assert len(faces) == 1 and faces[0]["score"] > 0.8
    x1, y1, x2, y2 = faces[0]["box"]
    assert 60 < (x1 + x2) / 2 < 260 and 30 < (y1 + y2) / 2 < 230  # face is mid-frame
    out = photos.pixelate(image, [faces[0]["box"]])
    face = np.asarray(out.crop((x1, y1, x2, y2)), dtype=float)
    # A mosaic has few distinct colours; the real face has thousands.
    assert len(np.unique(face.reshape(-1, 3), axis=0)) < 60


@pytest.mark.skipif(not FACE_MODEL.exists(), reason="YuNet not downloaded")
def test_no_face_in_a_plain_photo():
    image = Image.new("RGB", (640, 480), (120, 120, 120))
    assert photos.FaceDetector(str(FACE_MODEL)).detect(image) == []


def test_pixelate_changes_only_the_boxes_and_their_margin():
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (200, 300, 3), dtype=np.uint8))
    out = np.asarray(photos.pixelate(image, [(100, 50, 200, 150)]))
    src = np.asarray(image)
    assert (out[:30] == src[:30]).all() and (out[:, :80] == src[:, :80]).all()
    assert (out[60:140, 110:190] != src[60:140, 110:190]).any()


def test_pixelate_ignores_degenerate_boxes():
    image = Image.new("RGB", (50, 50), "red")
    assert (
        photos.pixelate(image, [(10, 10, 10, 10), (60, 60, 80, 80)]).tobytes()
        == image.tobytes()
    )


def test_redacted_copy_has_no_metadata():
    original = jpeg(
        Image.new("RGB", (64, 64), "blue"), exif=exif_with(gps=True, make="Canon")
    )
    assert photos.open_image(original).getexif()  # the input does carry EXIF
    rgb = photos.to_rgb(photos.open_image(original))
    out = Image.open(io.BytesIO(photos.redacted_jpeg(rgb, [])))
    assert not out.getexif()
    assert not {"exif", "icc_profile", "xmp"} & set(out.info)


# ---- metadata ----


def meta(data, incident=None, today=date(2026, 10, 28)):
    return forensics.check_metadata(photos.open_image(data), incident, today=today)


def codes(result):
    return {f["code"] for f in result["findings"]}


def test_camera_photo_from_the_right_day_is_clean():
    data = jpeg(
        Image.new("RGB", (32, 32)),
        exif=exif_with(make="Apple", taken="2026:10:01 14:03:00"),
    )
    result = meta(data, date(2026, 10, 1))
    assert codes(result) == set()
    assert result["taken_at"] == "2026-10-01T14:03:00" and result["camera"] == "Apple"


def test_photo_taken_well_before_the_incident():
    data = jpeg(Image.new("RGB", (32, 32)), exif=exif_with(taken="2025:03:02 10:00:00"))
    result = meta(data, date(2026, 10, 1))
    assert "taken_before_incident" in codes(result)
    assert (
        meta(
            jpeg(
                Image.new("RGB", (32, 32)), exif=exif_with(taken="2026:09:30 23:00:00")
            ),
            date(2026, 10, 1),
        )["findings"]
        == []
    )  # within a day: clock or time zone


def test_future_date():
    data = jpeg(Image.new("RGB", (32, 32)), exif=exif_with(taken="2027:01:01 00:00:00"))
    assert "future_date" in codes(meta(data))


def test_editing_software_is_a_warning_and_generators_are_high():
    edited = meta(
        jpeg(
            Image.new("RGB", (32, 32)), exif=exif_with(software="Adobe Photoshop 26.1")
        )
    )
    assert [(f["code"], f["severity"]) for f in edited["findings"]] == [
        ("edited", "warning")
    ]
    made = meta(
        jpeg(Image.new("RGB", (32, 32)), exif=exif_with(software="Midjourney v7"))
    )
    assert ("ai_metadata", "high") in {
        (f["code"], f["severity"]) for f in made["findings"]
    }


def test_iptc_ai_source_type_in_xmp():
    xmp = (
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description '
        b'Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/'
        b'trainedAlgorithmicMedia"/></rdf:RDF></x:xmpmeta>'
    )
    data = jpeg(Image.new("RGB", (32, 32)), xmp=xmp)
    assert "ai_metadata" in codes(meta(data))


def test_stable_diffusion_png_parameters():
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "a dented car door, photo, Steps: 30, Sampler: Euler a")
    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="PNG", pnginfo=info)
    assert "ai_metadata" in codes(meta(buf.getvalue()))


def test_missing_exif_and_gps_are_information_only():
    bare = meta(jpeg(Image.new("RGB", (32, 32))))
    assert [(f["code"], f["severity"]) for f in bare["findings"]] == [
        ("no_camera_metadata", "info")
    ]
    located = meta(
        jpeg(Image.new("RGB", (32, 32)), exif=exif_with(gps=True, make="Pixel"))
    )
    assert located["has_gps"] and ("location_in_metadata", "info") in {
        (f["code"], f["severity"]) for f in located["findings"]
    }


def test_garbage_dates_are_ignored():
    data = jpeg(Image.new("RGB", (32, 32)), exif=exif_with(taken="not a date"))
    assert meta(data, date(2026, 10, 1))["taken_at"] is None


# ---- the AI-image probe (fake encoder: the real one is trained in Colab) ----


class FakeSession:
    def __init__(self, features):
        self.features = np.asarray(features, dtype=np.float32)
        self.seen = None

    def run(self, outputs, feeds):
        (self.seen,) = feeds.values()
        return [self.features[None]]


def probe(features, weights, bias, l2=True):
    det = forensics.AiDetector.__new__(forensics.AiDetector)
    det.session, det.input, det.output = FakeSession(features), "pixel_values", None
    det.weights, det.bias, det.l2 = np.asarray(weights, dtype=np.float32), bias, l2
    det.threshold, det.size, det.name = 0.5, 224, "fake"
    return det


def test_clip_preprocessing_matches_clips_recipe():
    det = probe([1, 0], [0, 0], 0.0)
    x = det.preprocess(Image.new("RGB", (640, 320), (255, 255, 255)))
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    expected = (1.0 - forensics.CLIP_MEAN) / forensics.CLIP_STD
    assert np.allclose(x[0, :, 112, 112], expected, atol=1e-5)


def test_probe_is_logistic_regression_on_normalised_features():
    det = probe([3.0, 4.0], [1.0, 2.0], -1.0)  # normalised: (0.6, 0.8) -> z = 1.2
    assert det.score(Image.new("RGB", (300, 300))) == pytest.approx(
        1 / (1 + np.exp(-1.2))
    )
    raw = probe([3.0, 4.0], [1.0, 2.0], -1.0, l2=False)  # z = 10
    assert raw.score(Image.new("RGB", (300, 300))) == pytest.approx(
        1 / (1 + np.exp(-10))
    )
