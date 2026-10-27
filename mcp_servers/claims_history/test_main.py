import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import pytest
from mcp import Client

from main import (
    ClaimsStore,
    PastClaim,
    PhotoMatch,
    create_server,
    phash_to_bigint,
    summarise,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def past_claim(claim_id, incident, paid=100.0):
    return PastClaim(
        claim_id=claim_id,
        policy_id="P00001",
        peril="collision",
        description="Backed into a pole.",
        damages=[{"class": "dent", "severity": "minor"}],
        incident_date=incident,
        reported_date=incident,
        claimed_amount=paid,
        status="paid",
        paid_amount=paid,
    )


def test_summary_counts_last_12_months_and_total_paid():
    claims = [
        past_claim("CL1", date(2026, 9, 1), 200),
        past_claim("CL2", date(2025, 11, 1), 50.5),
        past_claim("CL3", date(2024, 1, 1), 1000),
    ]
    history = summarise("C00001", claims, as_of=date(2026, 10, 1))
    assert history.claims_last_12_months == 2
    assert history.total_paid == 1250.5


@pytest.mark.parametrize(
    "h", ["0000000000000000", "8000000000000000", "ffffffffffffffff"]
)
def test_phash_conversion_matches_the_loader(h):
    value = phash_to_bigint(h)
    assert f"{value & ((1 << 64) - 1):016x}" == h


class FakeStore:
    def __init__(self):
        self.calls = []

    def history(self, customer_id, start, end):
        self.calls.append(("history", customer_id, start, end))
        return [past_claim("CL1", date(2026, 9, 1))]

    def duplicates(self, hashes, customer_id, max_distance):
        self.calls.append(("duplicates", hashes, customer_id, max_distance))
        return [
            PhotoMatch(
                photo_index=0,
                claim_id="CL9",
                same_customer=False,
                distance=2,
                incident_date=date(2025, 1, 1),
            )
        ]


def server_with(fake):
    @asynccontextmanager
    async def lifespan(server):
        yield fake

    return create_server(lifespan=lifespan)


async def test_tools_listed_read_only():
    async with Client(server_with(FakeStore())) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {"get_claim_history", "find_duplicate_photos"}
    assert all(t.annotations.read_only_hint for t in tools.values())


async def test_history_window_is_years_before_as_of():
    fake = FakeStore()
    async with Client(server_with(fake)) as client:
        result = await client.call_tool(
            "get_claim_history",
            {"customer_id": "C00001", "years": 2, "as_of": "2026-10-01"},
        )
    assert result.structured_content["claims_last_12_months"] == 1
    assert fake.calls == [("history", "C00001", date(2024, 10, 1), date(2026, 10, 1))]


async def test_duplicate_match_hides_the_other_customer():
    async with Client(server_with(FakeStore())) as client:
        result = await client.call_tool(
            "find_duplicate_photos",
            {"phashes": ["5807f4a4135ce2f3"], "customer_id": "C00001"},
        )
    match = result.structured_content["matches"][0]
    assert match["same_customer"] is False
    assert "customer_id" not in match


@pytest.mark.parametrize(
    "bad",
    [
        {"phashes": []},
        {"phashes": ["not-a-hash"]},
        {"phashes": ["5807F4A4135CE2F3"]},  # hex must be lowercase, as stored
        {"customer_id": "C1"},
        {"max_distance": 40},  # past 16 bits a "match" is close to chance
    ],
)
async def test_invalid_arguments_rejected(bad):
    args = {"phashes": ["5807f4a4135ce2f3"], "customer_id": "C00001", **bad}
    async with Client(server_with(FakeStore())) as client:
        result = await client.call_tool("find_duplicate_photos", args)
    assert result.is_error


# ---- SQL against a real Postgres ----

TEST_DB = os.getenv("TEST_DATABASE_URL")
SCHEMA = Path(__file__).resolve().parents[2] / "db" / "schema.sql"


@pytest.fixture
def db_store():
    import psycopg
    from psycopg_pool import ConnectionPool

    with psycopg.connect(TEST_DB, autocommit=True) as conn:
        conn.execute(SCHEMA.read_text())
        conn.execute("TRUNCATE customers CASCADE")
        for cid in ("C00001", "C00002"):
            conn.execute(
                "INSERT INTO customers VALUES (%s,'A','B','1980-01-01','ME','a@example.com','(207) 555-0100','2020-01-01')",
                (cid,),
            )
            conn.execute(
                "INSERT INTO vehicles VALUES (%s,%s,%s,'Honda','Civic',2020,'economy')",
                ("V" + cid[1:], cid, "VIN" + cid),
            )
            conn.execute(
                "INSERT INTO policies VALUES (%s,%s,%s,'w1','ME','2020-01-01','2026-12-31','active',NULL,"
                "true,true,true,false,false,500,250,1500)",
                ("P" + cid[1:], cid, "V" + cid[1:]),
            )
        claims = [
            ("CL000001", "C00001", "2026-09-01", 300, "aaaaaaaaaaaaaaaa"),
            ("CL000002", "C00001", "2019-01-01", 999, "0000000000000000"),
            ("CL000003", "C00002", "2025-05-05", 120, "fffffffffffffff0"),
        ]
        for claim_id, cid, day, paid, phash in claims:
            conn.execute(
                "INSERT INTO claims VALUES (%s,%s,%s,%s,'collision','x','[]',%s,%s,%s,500,'paid',%s)",
                (claim_id, "P" + cid[1:], cid, "V" + cid[1:], day, day, paid, paid),
            )
            conn.execute(
                "INSERT INTO claim_photos VALUES (%s, 0, %s)",
                (claim_id, phash_to_bigint(phash)),
            )
    pool = ConnectionPool(TEST_DB, min_size=1, max_size=2, open=True)
    yield ClaimsStore(pool)
    pool.close()


@pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to run")
def test_history_only_returns_this_customer_in_window(db_store):
    claims = db_store.history("C00001", date(2021, 10, 1), date(2026, 10, 1))
    assert [c.claim_id for c in claims] == ["CL000001"]  # CL000002 is too old
    assert claims[0].damages == []


@pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to run")
def test_duplicate_search_by_hamming_distance(db_store):
    # ffff...ff differs from CL000003's ffff...f0 in 4 bits; from CL000002's 0000 in 64.
    matches = db_store.duplicates(["ffffffffffffffff"], "C00001", 8)
    assert [(m.claim_id, m.distance, m.same_customer) for m in matches] == [
        ("CL000003", 4, False)
    ]
    assert db_store.duplicates(["ffffffffffffffff"], "C00001", 3) == []
    # Several hashes at once: photo_index says which one matched.
    matches = db_store.duplicates(["1234123412341234", "aaaaaaaaaaaaaaab"], "C00001", 8)
    assert [
        (m.photo_index, m.claim_id, m.distance, m.same_customer) for m in matches
    ] == [(1, "CL000001", 1, True)]
