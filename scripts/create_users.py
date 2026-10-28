"""Create logins for the gateway: adjusters, and demo logins for synthetic customers.

Usage:
    python create_users.py --adjuster sam            # asks for a password
    python create_users.py --demo-customers 3        # prints a generated password for each
    (connects to DATABASE_URL, read from the repo's .env if not already set)

Non-interactive: ADJUSTER_PASSWORD sets the adjuster password, DEMO_PASSWORD
gives every demo customer the same password instead of a generated one.

Re-running updates passwords (upsert). Applies db/schema.sql first, which is
idempotent, so the auth table also appears on a database created before it.
Demo customers are the first customers with an active policy that has both
collision and comprehensive cover, so every kind of claim can be tried.
"""

import argparse
import getpass
import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "db" / "schema.sql"
# The gateway's own hashing module, not a copy, so the two can't disagree.
sys.path.insert(0, str(ROOT / "services" / "gateway"))
from passwords import hash_password

MIN_ADJUSTER_PASSWORD = 12

DEMO_CUSTOMERS_SQL = """
SELECT c.customer_id, c.email, c.first_name || ' ' || c.last_name
FROM customers c
WHERE EXISTS (
    SELECT 1 FROM policies p
    WHERE p.customer_id = c.customer_id
      AND p.status = 'active' AND p.collision AND p.comprehensive
)
ORDER BY c.customer_id
LIMIT %s
"""

UPSERT_SQL = """
INSERT INTO auth.users (username, password_hash, role, display_name, customer_id)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (username) DO UPDATE
SET password_hash = EXCLUDED.password_hash, role = EXCLUDED.role,
    display_name = EXCLUDED.display_name, customer_id = EXCLUDED.customer_id
"""


def upsert(conn, username, password, role, name, customer_id=None):
    conn.execute(
        UPSERT_SQL,
        (username.strip().lower(), hash_password(password), role, name, customer_id),
    )


def adjuster_password():
    password = os.environ.get("ADJUSTER_PASSWORD")
    if password is None:
        password = getpass.getpass("Adjuster password: ")
        if getpass.getpass("Again: ") != password:
            sys.exit("Passwords don't match")
    if len(password) < MIN_ADJUSTER_PASSWORD:
        sys.exit(f"Use at least {MIN_ADJUSTER_PASSWORD} characters")
    return password


def create_demo_customers(conn, n, password=None):
    """Returns [(username, password, customer_id)]."""
    created = []
    for customer_id, email, name in conn.execute(DEMO_CUSTOMERS_SQL, (n,)).fetchall():
        pw = password or secrets.token_urlsafe(9)
        upsert(conn, email, pw, "customer", name, customer_id)
        created.append((email, pw, customer_id))
    return created


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adjuster", action="append", default=[], metavar="USERNAME")
    parser.add_argument("--demo-customers", type=int, default=0, metavar="N")
    args = parser.parse_args()
    if not args.adjuster and not args.demo_customers:
        parser.error("nothing to do: give --adjuster and/or --demo-customers")

    import psycopg

    if "DATABASE_URL" not in os.environ:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv())
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        conn.execute(SCHEMA.read_text())
        for username in args.adjuster:
            upsert(conn, username, adjuster_password(), "adjuster", username)
            print(f"adjuster  {username}")
        demo = create_demo_customers(
            conn, args.demo_customers, os.environ.get("DEMO_PASSWORD")
        )
        for username, password, customer_id in demo:
            # Printed once; only the hash is stored.
            print(f"customer  {username:<34} {password:<14} {customer_id}")


if __name__ == "__main__":
    main()
