import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from mcp import Client

from embeddings import DIM
from main import (
    Clause,
    Policy,
    PolicyStore,
    Vehicle,
    Wording,
    create_server,
    evaluate_coverage,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_policy(**overrides):
    fields = {
        "policy_id": "P00001",
        "customer_id": "C00001",
        "state": "ME",
        "status": "active",
        "start_date": date(2025, 1, 1),
        "end_date": date(2026, 12, 31),
        "cancelled_date": None,
        "collision": True,
        "comprehensive": True,
        "full_glass": False,
        "rental_reimbursement": False,
        "collision_deductible": 500,
        "comprehensive_deductible": 250,
        "vehicle": Vehicle(
            vehicle_id="V00001",
            make="Honda",
            model="Civic",
            year=2020,
            segment="economy",
        ),
        "wording": Wording(
            wording_id="iso_pap_1998",
            title="Personal Auto Policy",
            insurer="Test",
            form_number="PP 00 01",
        ),
    }
    return Policy(**{**fields, **overrides})


# ---- coverage rules (pure) ----


def test_covered_collision_uses_collision_deductible():
    cov = evaluate_coverage(make_policy(), date(2026, 3, 1), "collision", ["dent"])
    assert cov.covered and cov.in_force and cov.deductible == 500
    assert cov.reasons == []


def test_incident_before_start_is_not_in_force():
    cov = evaluate_coverage(make_policy(), date(2024, 12, 31), "collision", ["dent"])
    assert not cov.in_force and not cov.covered
    assert "before the policy started" in cov.reasons[0]


def test_incident_after_cancellation_is_not_in_force():
    policy = make_policy(
        status="cancelled", cancelled_date=date(2025, 6, 1), end_date=date(2025, 6, 1)
    )
    cov = evaluate_coverage(policy, date(2025, 6, 1), "collision", ["dent"])
    assert not cov.covered
    assert "cancelled" in cov.reasons[0]


def test_lapsed_policy_is_not_in_force():
    policy = make_policy(status="lapsed", end_date=date(2025, 12, 31))
    cov = evaluate_coverage(policy, date(2026, 2, 1), "comprehensive", ["dent"])
    assert not cov.covered and "ended" in cov.reasons[0]


def test_peril_not_purchased():
    policy = make_policy(collision=False, collision_deductible=None)
    cov = evaluate_coverage(policy, date(2026, 3, 1), "collision", ["dent"])
    assert cov.in_force and not cov.peril_covered and not cov.covered
    assert cov.deductible is None


def test_full_glass_waives_deductible_for_glass_only():
    policy = make_policy(full_glass=True)
    glass = evaluate_coverage(
        policy, date(2026, 3, 1), "comprehensive", ["glass shatter"]
    )
    mixed = evaluate_coverage(
        policy, date(2026, 3, 1), "comprehensive", ["glass shatter", "dent"]
    )
    assert glass.deductible == 0
    assert mixed.deductible == 250


# ---- MCP protocol, with a fake store ----


class FakeStore:
    def __init__(self):
        self.searches = []

    def get(self, policy_id):
        return make_policy() if policy_id == "P00001" else None

    def search(self, query, wording_id, limit):
        self.searches.append((query, wording_id, limit))
        return [
            Clause(
                wording_id="iso_pap_1998",
                section="PART D > EXCLUSIONS",
                page_start=8,
                page_end=8,
                text="We will not pay for wear and tear.",
                score=0.03,
            )
        ]


def server_with(fake):
    @asynccontextmanager
    async def lifespan(server):
        yield fake

    return create_server(lifespan=lifespan)


async def test_tools_are_listed_read_only():
    async with Client(server_with(FakeStore())) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {"get_policy", "check_coverage", "search_clauses"}
    assert all(t.annotations.read_only_hint for t in tools.values())


async def test_get_policy_returns_ids_not_personal_data():
    async with Client(server_with(FakeStore())) as client:
        result = await client.call_tool("get_policy", {"policy_id": "P00001"})
    assert not result.is_error
    policy = result.structured_content
    assert policy["customer_id"] == "C00001"
    assert policy["vehicle"]["segment"] == "economy"
    assert not {"first_name", "last_name", "email", "phone"} & policy.keys()


async def test_unknown_policy_is_a_clear_error():
    async with Client(server_with(FakeStore())) as client:
        result = await client.call_tool("get_policy", {"policy_id": "P99999"})
    assert result.is_error
    assert "No policy P99999" in result.content[0].text


@pytest.mark.parametrize("bad_id", ["P1", "'; DROP TABLE policies;--", "C00001"])
async def test_malformed_policy_id_is_rejected_before_any_lookup(bad_id):
    async with Client(server_with(FakeStore())) as client:
        result = await client.call_tool("get_policy", {"policy_id": bad_id})
    assert result.is_error


async def test_check_coverage_tool():
    args = {
        "policy_id": "P00001",
        "incident_date": "2026-03-01",
        "peril": "comprehensive",
        "damage_classes": ["dent"],
    }
    async with Client(server_with(FakeStore())) as client:
        result = await client.call_tool("check_coverage", args)
        bad = await client.call_tool("check_coverage", {**args, "peril": "flood"})
    assert result.structured_content["covered"] is True
    assert result.structured_content["deductible"] == 250
    assert bad.is_error  # peril must be one of the enum values


async def test_search_scopes_to_the_policys_wording():
    fake = FakeStore()
    async with Client(server_with(fake)) as client:
        result = await client.call_tool(
            "search_clauses",
            {"query": "wear and tear", "policy_id": "P00001", "limit": 3},
        )
    assert result.structured_content["wording_id"] == "iso_pap_1998"
    assert result.structured_content["results"][0]["section"] == "PART D > EXCLUSIONS"
    assert fake.searches == [("wear and tear", "iso_pap_1998", 3)]


async def test_search_limit_is_capped():
    async with Client(server_with(FakeStore())) as client:
        result = await client.call_tool(
            "search_clauses", {"query": "hail", "limit": 500}
        )
    assert result.is_error


# ---- SQL against a real Postgres ----

SCHEMA = Path(__file__).resolve().parents[2] / "db" / "schema.sql"
TEST_DB = os.getenv("TEST_DATABASE_URL")


class KeywordEmbedder:
    """Puts 'glass' texts and queries near each other, so the vector half of the
    hybrid search is predictable without the real model."""

    @staticmethod
    def _vec(text):
        v = np.full(DIM, 0.01, dtype=np.float32)
        v[0] = 1.0 if "glass" in text.lower() else 0.0
        v[1] = 0.0 if "glass" in text.lower() else 1.0
        return v

    def query(self, text):
        return self._vec(text)


@pytest.fixture
def db_store():
    import psycopg
    from pgvector.psycopg import register_vector
    from psycopg_pool import ConnectionPool

    with psycopg.connect(TEST_DB, autocommit=True) as conn:
        conn.execute(SCHEMA.read_text())
        register_vector(conn)
        conn.execute("TRUNCATE customers, policy_wordings CASCADE")
        conn.execute(
            "INSERT INTO customers VALUES ('C00001','Ann','Lee','1980-01-01','ME','a@example.com','(207) 555-0100','2020-01-01')"
        )
        conn.execute(
            "INSERT INTO vehicles VALUES ('V00001','C00001','VIN00000000000001','Honda','Civic',2020,'economy')"
        )
        conn.execute(
            "INSERT INTO policies VALUES ('P00001','C00001','V00001','w1','ME','2025-01-01','2026-12-31','active',NULL,"
            "true,true,true,false,false,500,250,1500)"
        )
        conn.execute(
            "INSERT INTO policy_wordings VALUES ('w1','Test policy','Test Mutual','T-1','https://example.com','"
            + "0" * 64
            + "',3)"
        )
        chunks = [
            (
                "PART D > INSURING AGREEMENT",
                "We will pay for loss caused by breakage of glass.",
            ),
            (
                "PART D > EXCLUSIONS",
                "We will not pay for damage due to wear and tear or freezing.",
            ),
            (
                "PART E > DUTIES",
                "You must promptly notify us of how, when and where the accident happened.",
            ),
        ]
        conn.execute(
            "INSERT INTO policy_wordings VALUES ('w2','Other policy','Other Mutual','O-1','https://example.com','"
            + "0" * 64
            + "',1)"
        )
        chunks = [
            *(("w1", s, t) for s, t in chunks),
            ("w2", "PART D > GLASS", "Glass breakage is covered."),
        ]
        for i, (wording, section, text) in enumerate(chunks):
            conn.execute(
                "INSERT INTO policy_chunks (wording_id, chunk_index, section, page_start, page_end, content, embedding) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (wording, i, section, i + 1, i + 1, text, KeywordEmbedder._vec(text)),
            )
    pool = ConnectionPool(
        TEST_DB, min_size=1, max_size=2, configure=register_vector, open=True
    )
    yield PolicyStore(pool, KeywordEmbedder())
    pool.close()


@pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to run")
def test_store_loads_policy_with_vehicle_and_wording(db_store):
    policy = db_store.get("P00001")
    assert policy.vehicle.make == "Honda"
    assert policy.wording.wording_id == "w1"
    assert db_store.get("P00002") is None


@pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to run")
def test_search_ranks_by_similarity(db_store):
    results = db_store.search("my windshield glass broke", "w1", 3)
    assert results[0].section == "PART D > INSURING AGREEMENT"
    assert results[0].score == pytest.approx(
        1.0
    )  # cosine similarity of identical vectors
    assert results[0].score > results[1].score


@pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to run")
def test_search_stays_within_one_wording(db_store):
    # Another insurer's glass clause is a better match, but it isn't this policy's wording.
    assert {c.wording_id for c in db_store.search("glass", "w1", 10)} == {"w1"}
    assert {c.wording_id for c in db_store.search("glass", None, 10)} == {"w1", "w2"}
    assert len(db_store.search("glass", None, 2)) == 2
