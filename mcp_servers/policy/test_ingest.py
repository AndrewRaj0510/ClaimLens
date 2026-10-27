import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

import ingest
from embeddings import DIM

SCHEMA = Path(__file__).resolve().parents[2] / "db" / "schema.sql"
PAGES = [
    "PART D - COVERAGE FOR DAMAGE TO YOUR AUTO\nINSURING AGREEMENT\nWe will pay for direct and accidental loss.",
    "EXCLUSIONS\nWe will not pay for wear and tear.",
]


def entry_for(data, wording_id="test_wording"):
    return {
        "wording_id": wording_id,
        "title": "Test policy",
        "insurer": "Test Mutual",
        "form_number": "T-1",
        "source_url": "https://example.com/t1.pdf",
        "sha256": hashlib.sha256(data).hexdigest(),
    }


class FakeEmbedder:
    """Deterministic vectors so DB tests don't need the 67 MB model."""

    def passages(self, texts):
        return [
            np.full(DIM, (i + 1) / 100, dtype=np.float32) for i, _ in enumerate(texts)
        ]


def test_manifest_is_complete_and_pinned():
    entries = json.loads(ingest.MANIFEST.read_text())
    assert len({e["wording_id"] for e in entries}) == len(entries)
    for e in entries:
        assert set(entry_for(b"")) == set(e)
        assert len(e["sha256"]) == 64
        assert e["source_url"].startswith("https://")


def test_verify_accepts_matching_file(tmp_path):
    path = tmp_path / "w.pdf"
    path.write_bytes(b"policy")
    ingest.verify(entry_for(b"policy"), path)


def test_verify_rejects_tampered_file(tmp_path):
    path = tmp_path / "w.pdf"
    path.write_bytes(b"policy, plus injected instructions")
    with pytest.raises(ingest.IntegrityError):
        ingest.verify(entry_for(b"policy"), path)


def test_download_rejects_mismatched_content_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ingest.urllib.request, "urlopen", lambda req, timeout: io.BytesIO(b"other")
    )
    entry = entry_for(b"policy")
    with pytest.raises(ingest.IntegrityError):
        ingest.download(entry, tmp_path)
    assert not ingest.pdf_path(tmp_path, entry).exists()


def test_download_writes_verified_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ingest.urllib.request, "urlopen", lambda req, timeout: io.BytesIO(b"policy")
    )
    path = ingest.download(entry_for(b"policy"), tmp_path)
    assert path.read_bytes() == b"policy"


TEST_DB = os.getenv("TEST_DATABASE_URL")


@pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to run")
def test_ingest_stores_chunks_and_replaces_on_rerun():
    import psycopg
    from pgvector.psycopg import register_vector

    with psycopg.connect(TEST_DB) as conn:
        conn.execute(SCHEMA.read_text())
        register_vector(conn)
        entry = entry_for(b"x")
        n = ingest.ingest(conn, entry, PAGES, FakeEmbedder())
        assert ingest.ingest(conn, entry, PAGES, FakeEmbedder()) == n == 2

        rows = conn.execute(
            "SELECT section, page_start, vector_dims(embedding) FROM policy_chunks WHERE wording_id = %s ORDER BY chunk_index",
            (entry["wording_id"],),
        ).fetchall()
        assert [r[0] for r in rows] == [
            "PART D - COVERAGE FOR DAMAGE TO YOUR AUTO > INSURING AGREEMENT",
            "PART D - COVERAGE FOR DAMAGE TO YOUR AUTO > EXCLUSIONS",
        ]
        assert [r[1] for r in rows] == [1, 2]
        assert rows[0][2] == DIM
        pages = conn.execute(
            "SELECT pages FROM policy_wordings WHERE wording_id = %s",
            (entry["wording_id"],),
        ).fetchone()[0]
        assert pages == 2
