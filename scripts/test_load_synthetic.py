import os
from pathlib import Path

import pytest

from generate_synthetic import generate
from load_synthetic import load, phash_to_bigint, to_rows

SCHEMA = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


@pytest.mark.parametrize(
    "h",
    [
        "0000000000000000",
        "7fffffffffffffff",
        "8000000000000000",
        "ffffffffffffffff",
        "5807f4a4135ce2f3",
    ],
)
def test_phash_round_trips_through_signed_bigint(h):
    value = phash_to_bigint(h)
    assert -(1 << 63) <= value < 1 << 63
    assert f"{value & ((1 << 64) - 1):016x}" == h


def test_to_rows_flattens_nested_fields():
    data = generate(seed=3, n_customers=100)
    tables = to_rows(data)

    columns, rows = tables["policies"]
    assert {"collision", "comprehensive", "full_glass"} <= set(columns)
    assert "coverages" not in columns
    assert len(rows) == len(data["policies"])

    columns, rows = tables["claims"]
    assert "photo_phashes" not in columns
    assert isinstance(rows[0][columns.index("damages")], str)  # JSON text for jsonb

    _, photos = tables["claim_photos"]
    assert len(photos) == sum(len(c["photo_phashes"]) for c in data["claims"])


# Needs a throwaway database: the load TRUNCATEs every synthetic table.
#   docker compose exec postgres createdb -U claimlens claimlens_test
#   TEST_DATABASE_URL=postgresql://claimlens:<pw>@127.0.0.1:5433/claimlens_test python -m pytest
TEST_DB = os.getenv("TEST_DATABASE_URL")


@pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to run")
def test_load_into_postgres_and_find_near_duplicate_photo():
    import psycopg

    data = generate(seed=5, n_customers=300)
    with psycopg.connect(TEST_DB) as conn:
        conn.execute(SCHEMA.read_text())
        counts = load(conn, data)
        load(conn, data)  # second load replaces, doesn't duplicate
        n_claims = conn.execute("SELECT count(*) FROM claims").fetchone()[0]
        assert n_claims == counts["claims"] == len(data["claims"])

        # Hamming distance in SQL matches Python on a planted near-duplicate.
        claim = data["claims"][0]
        original = int(claim["photo_phashes"][0], 16)
        probe = original ^ 0b1011  # 3 bits flipped
        row = conn.execute(
            """
            SELECT claim_id, bit_count((phash # %s)::bit(64)) AS distance
            FROM claim_photos ORDER BY distance, claim_id LIMIT 1
            """,
            (phash_to_bigint(f"{probe:016x}"),),
        ).fetchone()
        assert row == (claim["claim_id"], 3)
