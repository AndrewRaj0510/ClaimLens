"""Create or update the per-service database roles (db/roles.sql) on an
existing database, and set their passwords from .env.

A fresh Postgres volume does this by itself (db/roles.sql and
db/role-passwords.sh run from docker-entrypoint-initdb.d). Run this once on a
volume created before Phase 4, or after changing a DB_PASSWORD_* value:

    python scripts/apply_db_roles.py

Connects as the owner (DATABASE_URL in .env, or --database-url). Idempotent.
"""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROLES_SQL = ROOT / "db" / "roles.sql"
PASSWORDS = {
    "claimlens_gateway": "DB_PASSWORD_GATEWAY",
    "claimlens_agent": "DB_PASSWORD_AGENT",
    "claimlens_policy": "DB_PASSWORD_POLICY",
    "claimlens_pricing": "DB_PASSWORD_PRICING",
    "claimlens_claims_history": "DB_PASSWORD_CLAIMS_HISTORY",
}


def apply(conn, passwords):
    """Run roles.sql, then set each role's password (None: leave it as is)."""
    from psycopg import sql

    conn.execute(ROLES_SQL.read_text())
    for role, password in passwords.items():
        if password:
            conn.execute(
                sql.SQL("ALTER ROLE {} PASSWORD {}").format(
                    sql.Identifier(role), sql.Literal(password)
                )
            )


def main():
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    args = parser.parse_args()
    missing = [env for env in PASSWORDS.values() if not os.getenv(env)]
    if missing:
        raise SystemExit(f"Set these in .env first: {', '.join(missing)}")

    import psycopg

    with psycopg.connect(args.database_url, autocommit=True) as conn:
        apply(conn, {role: os.getenv(env) for role, env in PASSWORDS.items()})
    print(f"Applied {ROLES_SQL.name} and set {len(PASSWORDS)} role passwords.")


if __name__ == "__main__":
    main()
