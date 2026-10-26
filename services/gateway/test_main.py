import httpx
import pytest
from fastapi.testclient import TestClient

import main
from main import app, vision_client

client = TestClient(app)

VISION_RESPONSE = {"width": 10, "height": 10, "detections": [], "damage_area_pct": 0.0}


@pytest.fixture
def vision():
    """Replace the vision service with a handler function: vision(handler)."""

    def use(handler):
        mock = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://vision"
        )
        app.dependency_overrides[vision_client] = lambda: mock

    yield use
    app.dependency_overrides.clear()


def upload(data=b"jpegbytes", ctype="image/jpeg"):
    return client.post("/detect", files={"file": ("car.jpg", data, ctype)})


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "gateway"}


def test_detect_forwards_image_to_vision(vision):
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = request.read()
        return httpx.Response(200, json=VISION_RESPONSE)

    vision(handler)
    r = upload(b"jpegbytes")
    assert r.status_code == 200
    assert r.json() == VISION_RESPONSE
    assert seen["path"] == "/detect"
    assert b"jpegbytes" in seen["body"]
    assert b"car.jpg" not in seen["body"]  # user filename isn't passed on


def test_non_image_type_is_rejected_without_calling_vision(vision):
    vision(lambda request: pytest.fail("vision should not be called"))
    assert upload(ctype="text/plain").status_code == 415


def test_oversized_upload_is_rejected(vision, monkeypatch):
    vision(lambda request: pytest.fail("vision should not be called"))
    monkeypatch.setattr(main, "MAX_UPLOAD_MB", 0.001)  # ~1 KB
    assert upload(b"x" * 2000).status_code == 413


def test_vision_down_is_502(vision):
    def handler(request):
        raise httpx.ConnectError("connection refused")

    vision(handler)
    assert upload().status_code == 502


@pytest.mark.parametrize(
    ("vision_status", "expected"),
    [(400, 400), (413, 413), (503, 503), (500, 502)],
)
def test_vision_errors_are_mapped(vision, vision_status, expected):
    vision(lambda request: httpx.Response(vision_status, json={"detail": "nope"}))
    r = upload()
    assert r.status_code == expected
    if expected != 502:
        assert r.json()["detail"] == "nope"
