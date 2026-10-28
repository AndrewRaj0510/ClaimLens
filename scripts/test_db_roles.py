"""Per-service database roles (db/roles.sql): each role can do what its service
needs and nothing else.

Runs against the throwaway test database (see test_load_synthetic.py). The
roles are cluster-wide, so the test never sets passwords (that would change
the running stack's): it switches roles with SET ROLE as the owner."""

import os
from pathlib import Path

import pytest

from apply_db_roles import apply
from generate_synthetic import generate
from load_synthetic import load

ROOT = Path(__file__).resolve().parent.parent
TEST_DB = os.getenv("TEST_DATABASE_URL")

READS = {
    "claimlens_gateway": ["auth.users", "policies", "vehicles"],
    "claimlens_policy": ["policies", "vehicles", "policy_wordings", "policy_chunks"],
    "claimlens_pricing": [
        "labour_rates",
        "segment_multipliers",
        "repair_operations",
        "pricing_settings",
    ],
    "claimlens_claims_history": ["claims", "claim_photos"],
}
SENSITIVE = ["auth.users", "eval.fraud_labels", "customers", "claims", "policies"]


@pytest.fixture(scope="module")
def conn():
    import psycopg

    with psycopg.connect(TEST_DB, autocommit=True) as c:
        c.execute((ROOT / "db" / "schema.sql").read_text())
        load(c, generate(seed=7, n_customers=50))
        # A table the agent created in public before Phase 4, to be migrated.
        c.execute("DROP SCHEMA IF EXISTS agent CASCADE")
        c.execute("DROP TABLE IF EXISTS public.agent_claims")
        c.execute("CREATE TABLE public.agent_claims (claim_id text PRIMARY KEY)")
        c.execute(
            "INSERT INTO public.agent_claims VALUES ('old') ON CONFLICT DO NOTHING"
        )
        apply(c, {})
        apply(c, {})  # idempotent
        yield c


def can(conn, role, statement, params=None):
    import psycopg

    conn.execute("BEGIN")
    try:
        conn.execute(f"SET LOCAL ROLE {role}")
        conn.execute(statement, params)
        return True
    except psycopg.errors.InsufficientPrivilege:
        return False
    finally:
        conn.execute("ROLLBACK")


@pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to run")
@pytest.mark.parametrize("role", [*READS, "claimlens_agent"])
def test_each_role_reads_its_tables_and_nothing_else(conn, role):
    allowed = set(READS.get(role, []))
    for table in sorted(set(SENSITIVE) | allowed):
        expected = table in allowed
        assert can(conn, role, f"SELECT 1 FROM {table} LIMIT 1") is expected, (
            role,
            table,
        )


@pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to run")
@pytest.mark.parametrize("role", [*READS, "claimlens_agent"])
def test_no_role_can_write_shared_tables_or_create_in_public(conn, role):
    assert not can(conn, role, "UPDATE policies SET annual_premium = 0")
    assert not can(conn, role, "DELETE FROM claims")
    assert not can(conn, role, "UPDATE auth.users SET role = 'adjuster'")
    assert not can(conn, role, "CREATE TABLE public.sneaky (x int)")


@pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to run")
def test_agent_records_claims_and_owns_its_schema(conn):
    policy = conn.execute(
        "SELECT policy_id, customer_id, vehicle_id FROM policies LIMIT 1"
    ).fetchone()
    insert = (
        "INSERT INTO claims VALUES (%s, %s, %s, %s, 'collision', 'x', '[]', '2026-01-01',"
        " '2026-01-02', 1, 0, 'approved', 1) ON CONFLICT DO NOTHING"
    )
    # ClaimRecorder's insert, including the ON CONFLICT it uses for retries.
    assert can(conn, "claimlens_agent", insert, ("CLM-roles-test", *policy))
    assert can(
        conn,
        "claimlens_agent",
        # SET ROLE doesn't apply the role's login search_path, so name the schema.
        "CREATE TABLE agent.checkpoints_probe (x int); DROP TABLE agent.checkpoints_probe",
    )
    # Its old table moved into its schema with the data, and it owns it.
    row = conn.execute(
        "SELECT schemaname, tableowner FROM pg_tables WHERE tablename = 'agent_claims'"
    ).fetchall()
    assert row == [("agent", "claimlens_agent")]
    assert conn.execute("SELECT claim_id FROM agent.agent_claims").fetchall() == [
        ("old",)
    ]
    # Its default search path finds its own tables first.
    settings = conn.execute(
        "SELECT unnest(setconfig) FROM pg_db_role_setting s JOIN pg_roles r"
        " ON r.oid = s.setrole WHERE r.rolname = 'claimlens_agent'"
    ).fetchall()
    assert ("search_path=agent, public",) in settings
