import os
from pathlib import Path

import pytest

from create_users import create_demo_customers, upsert
from generate_synthetic import generate
from load_synthetic import load
from passwords import verify_password  # the gateway's module, via create_users

SCHEMA = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
TEST_DB = os.getenv("TEST_DATABASE_URL")


@pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to run")
def test_demo_customers_and_adjusters():
    import psycopg

    with psycopg.connect(TEST_DB) as conn:
        conn.execute(SCHEMA.read_text())
        load(conn, generate(seed=3, n_customers=100))
        conn.execute("TRUNCATE auth.users")

        demo = create_demo_customers(conn, 3)
        # The next ones along, as the load test's logins: no overlap.
        more = create_demo_customers(conn, 2, "same-password-1", offset=3)
        assert not {c for *_, c in more} & {c for *_, c in demo}
        conn.execute(
            "DELETE FROM auth.users WHERE customer_id = ANY(%s)",
            ([c for *_, c in more],),
        )
        upsert(conn, " Sam ", "first-password-123", "adjuster", "Sam")
        upsert(conn, "sam", "second-password-456", "adjuster", "Sam")  # updates
        rows = conn.execute(
            "SELECT username, role, customer_id, password_hash FROM auth.users"
            " ORDER BY username"
        ).fetchall()
        # Every demo customer can actually file any kind of claim.
        covered = conn.execute(
            "SELECT count(DISTINCT customer_id) FROM policies WHERE customer_id = ANY(%s)"
            " AND status = 'active' AND collision AND comprehensive",
            ([c for _, _, c in demo],),
        ).fetchone()[0]
        conn.rollback()

    assert len(demo) == 3 and covered == 3
    by_name = {r[0]: r for r in rows}
    assert len(rows) == 4
    assert by_name["sam"][1:3] == ("adjuster", None)
    assert verify_password("second-password-456", by_name["sam"][3])
    for username, password, customer_id in demo:
        assert username.endswith("@example.com")
        assert by_name[username][1:3] == ("customer", customer_id)
        assert verify_password(password, by_name[username][3])
    assert len({p for _, p, _ in demo}) == 3  # each gets its own password
