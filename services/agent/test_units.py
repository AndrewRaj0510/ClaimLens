"""Unit tests for the deterministic parts: image checks, pHash, severity, fraud score."""

import io
import os
from datetime import date

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

import damage
import fraud
import images
from tools import ToolNotAllowed, check_allowed


def scene(seed=0, size=(800, 600)):
    """A synthetic 'photo': random shapes with hard edges."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", size, (120, 130, 140))
    draw = ImageDraw.Draw(img)
    for _ in range(30):
        x, y = rng.integers(0, size[0]), rng.integers(0, size[1])
        w, h = rng.integers(20, 200, size=2)
        colour = tuple(int(c) for c in rng.integers(0, 255, size=3))
        draw.rectangle([x, y, x + w, y + h], fill=colour)
    return img


def jpeg_bytes(img, quality=90):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


# ---- quality ----


def test_sharp_well_lit_photo_passes():
    q = images.assess(scene())
    assert q["ok"], q


def test_blurry_photo_fails():
    q = images.assess(scene().filter(ImageFilter.GaussianBlur(6)))
    assert not q["ok"]
    assert "blurry" in q["problems"][0]


def test_dark_and_overexposed_photos_fail():
    dark = Image.eval(scene(), lambda v: v // 8)
    bright = Image.eval(scene(), lambda v: 235 + v // 25)
    assert "too dark" in " ".join(images.assess(dark)["problems"])
    assert "over-exposed" in " ".join(images.assess(bright)["problems"])


def test_small_photo_fails():
    q = images.assess(scene(size=(320, 240)))
    assert "too small" in q["problems"][0]


def test_exif_rotation_is_applied():
    img = scene(size=(800, 600))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    assert images.open_image(buf.getvalue()).size == (600, 800)


# ---- pHash ----


def test_phash_is_stable_under_resave_and_resize():
    original = scene(seed=1)
    h = images.phash(original)
    resaved = images.open_image(jpeg_bytes(original, quality=60))
    resized = original.resize((400, 300))
    cropped = original.crop((10, 8, 790, 592)).resize((800, 600))
    assert images.hamming(h, images.phash(resaved)) <= 4
    assert images.hamming(h, images.phash(resized)) <= 4
    assert images.hamming(h, images.phash(cropped)) <= 10


def test_phash_differs_for_different_photos():
    distances = [
        images.hamming(images.phash(scene(seed=1)), images.phash(scene(seed=s)))
        for s in range(2, 12)
    ]
    assert min(distances) > 12  # well clear of the duplicate threshold (8)


def test_phash_format():
    h = images.phash(scene())
    assert len(h) == 16 and int(h, 16) >= 0


# ---- severity and aggregation ----


@pytest.mark.parametrize(
    ("cls", "area", "expected"),
    [
        ("dent", 0.5, "minor"),
        ("dent", 2.0, "moderate"),
        ("dent", 4.0, "severe"),
        ("glass shatter", 5.0, "moderate"),
        ("scratch", 2.5, "severe"),
    ],
)
def test_severity_thresholds(cls, area, expected):
    assert damage.severity(cls, area) == expected


def detection(cls, area, conf=0.9):
    return {"class_name": cls, "area_pct": area, "confidence": conf}


def test_aggregate_merges_the_same_class_across_photos():
    results = [
        {"detections": [detection("dent", 1.5), detection("scratch", 0.2)]},
        None,  # a photo whose vision job failed
        {"detections": [detection("dent", 5.0), detection("crack", 0.8, conf=0.2)]},
    ]
    damages = damage.aggregate(results)
    assert [(d["damage_class"], d["severity"], d["photo_index"]) for d in damages] == [
        ("dent", "severe", 2),  # largest view wins
        ("scratch", "minor", 0),
    ]  # the low-confidence crack is dropped


# ---- fraud ----


def base_signals(**overrides):
    args = {
        "incident_date": date(2026, 9, 1),
        "reported_date": date(2026, 9, 3),
        "policy_start": date(2024, 1, 1),
        "claims_last_12_months": 0,
        "duplicate_matches": [],
        "claimed_amount": None,
        "estimate_high": 1000.0,
        "mentioned_date": None,
    }
    return fraud.signals(**{**args, **overrides})


def names(found):
    return [s["signal"] for s in found]


def test_clean_claim_has_no_signals():
    result = fraud.score(base_signals())
    assert result == {"score": 0.0, "level": "low", "signals": []}


@pytest.mark.parametrize(
    ("overrides", "signal"),
    [
        (
            {"duplicate_matches": [{"same_customer": False}]},
            "photo_reused_other_customer",
        ),
        (
            {"duplicate_matches": [{"same_customer": True}]},
            "photo_reused_same_customer",
        ),
        ({"policy_start": date(2026, 8, 20)}, "early_claim"),
        ({"reported_date": date(2026, 10, 15)}, "late_report"),
        ({"claims_last_12_months": 2}, "frequent_claims"),
        ({"claimed_amount": 2500.0}, "amount_inflated"),
        ({"mentioned_date": date(2026, 8, 1)}, "date_mismatch"),
    ],
)
def test_each_signal(overrides, signal):
    assert names(base_signals(**overrides)) == [signal]


def test_reused_photo_alone_is_high_risk():
    result = fraud.score(base_signals(duplicate_matches=[{"same_customer": False}]))
    assert result["level"] == "high"


def test_weak_signals_combine_with_noisy_or():
    found = base_signals(
        policy_start=date(2026, 8, 20), reported_date=date(2026, 10, 15)
    )
    result = fraud.score(found)
    assert result["score"] == round(1 - (1 - 0.25) * (1 - 0.2), 3) == 0.4
    assert result["level"] == "medium"


# ---- tool allowlist ----


def test_nodes_can_only_use_their_own_tools():
    check_allowed("policy_lookup", "policy", "search_clauses")
    with pytest.raises(ToolNotAllowed):
        check_allowed("policy_lookup", "pricing", "estimate_repair_cost")
    with pytest.raises(ToolNotAllowed):
        check_allowed("intake", "policy", "get_policy")  # intake has no tools
    with pytest.raises(ToolNotAllowed):
        check_allowed("human_approval", "policy", "approve_claim")


def test_open_image_refuses_a_decompression_bomb_before_decoding():
    buf = io.BytesIO()
    Image.new("L", (8000, 8000)).save(buf, "PNG")  # 64 MP, ~62 KB
    assert len(buf.getvalue()) < 100_000
    with pytest.raises(ValueError, match="too many pixels"):
        images.open_image(buf.getvalue())


# ---- ClaimIndex against a real Postgres ----

TEST_DB = os.getenv("TEST_DATABASE_URL")


@pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to run")
def test_claim_index_sql():
    import asyncio

    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    from adapters import ClaimIndex

    async def scenario():
        async with AsyncConnectionPool(
            TEST_DB, kwargs={"autocommit": True, "row_factory": dict_row}, open=False
        ) as pool:
            index = ClaimIndex(pool)
            await index.setup()
            await index.setup()  # idempotent
            async with pool.connection() as conn:
                await conn.execute("TRUNCATE agent_claims")
            await index.add("a" * 32, "C00001", "P00001")
            await index.add("b" * 32, "C00002", "P00002")
            await index.add("a" * 32, "C00009", "P00009")  # duplicate: ignored
            await index.update("a" * 32, "awaiting_approval", {"fraud_level": "low"})
            await index.update("b" * 32, "completed")  # summary kept
            all_rows = await index.list()
            mine = await index.list(customer_id="C00001")
            waiting = await index.list(status="awaiting_approval")
            return all_rows, mine, waiting

    # psycopg's async mode needs a selector loop (Windows defaults to proactor).
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        all_rows, mine, waiting = runner.run(scenario())
    assert [r["claim_id"] for r in all_rows] == ["b" * 32, "a" * 32]  # newest first
    assert all_rows[0]["summary"] == {}
    assert [(r["customer_id"], r["summary"]) for r in mine] == [
        ("C00001", {"fraud_level": "low"})
    ]
    assert [r["claim_id"] for r in waiting] == ["a" * 32]
    assert isinstance(all_rows[0]["created_at"], str)
