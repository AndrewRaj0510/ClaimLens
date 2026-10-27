"""Load the output of generate_synthetic.py into Postgres.

Replaces all synthetic tables in one transaction, so re-running is safe and a
failure leaves the previous data intact. Policy wordings are not touched
(mcp_servers/policy/ingest.py owns those).

Usage:
    python load_synthetic.py --data ../data/synthetic
    (connects to DATABASE_URL, read from the repo's .env if not already set)
"""

import argparse
import json
import os
from pathlib import Path

# Children before parents, for TRUNCATE order clarity (CASCADE handles it anyway).
TABLES = (
    "eval.fraud_labels",
    "claim_photos",
    "claims",
    "policies",
    "vehicles",
    "customers",
    "labour_rates",
    "segment_multipliers",
    "repair_operations",
    "pricing_settings",
)


def phash_to_bigint(phash):
    """16 hex chars (unsigned 64-bit) -> signed bigint, which is what Postgres stores."""
    value = int(phash, 16)
    return value - (1 << 64) if value >= 1 << 63 else value


def to_rows(data):
    """Map the generator's JSON to {table: (columns, rows)}. Pure, so it's testable without a DB."""
    pricing = data["pricing"]
    rows = {
        "customers": data["customers"],
        "vehicles": data["vehicles"],
        "policies": [
            {**{k: v for k, v in p.items() if k != "coverages"}, **p["coverages"]}
            for p in data["policies"]
        ],
        "claims": [
            {
                **{k: v for k, v in c.items() if k != "photo_phashes"},
                "damages": json.dumps(c["damages"]),
            }
            for c in data["claims"]
        ],
        "claim_photos": [
            {"claim_id": c["claim_id"], "position": i, "phash": phash_to_bigint(h)}
            for c in data["claims"]
            for i, h in enumerate(c["photo_phashes"])
        ],
        "eval.fraud_labels": data["fraud_labels"],
        "labour_rates": [
            {"state": s, "rate_per_hour": r}
            for s, r in pricing["labour_rates_per_hour"].items()
        ],
        "segment_multipliers": [
            {"segment": s, "parts_multiplier": m}
            for s, m in pricing["segment_parts_multiplier"].items()
        ],
        "repair_operations": pricing["operations"],
        "pricing_settings": [
            {
                "key": "paint_materials_per_hour",
                "value": pricing["paint_materials_per_hour"],
            }
        ],
    }
    return {
        table: (list(records[0]), [tuple(r.values()) for r in records])
        for table, records in rows.items()
    }


def read_data(data_dir):
    names = ("customers", "vehicles", "policies", "claims", "fraud_labels", "pricing")
    return {n: json.loads((Path(data_dir) / f"{n}.json").read_text()) for n in names}


def load(conn, data):
    # Imported here so the pure helpers above can be tested without psycopg.
    from psycopg import sql

    order = (
        "customers",
        "vehicles",
        "policies",
        "claims",
        "claim_photos",
        "eval.fraud_labels",
        "labour_rates",
        "segment_multipliers",
        "repair_operations",
        "pricing_settings",
    )
    tables = to_rows(data)
    counts = {}
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            sql.SQL("TRUNCATE {} CASCADE").format(
                sql.SQL(", ").join(sql.Identifier(*t.split(".")) for t in TABLES)
            )
        )
        for table in order:
            columns, rows = tables[table]
            copy = sql.SQL("COPY {} ({}) FROM STDIN").format(
                sql.Identifier(*table.split(".")),
                sql.SQL(", ").join(map(sql.Identifier, columns)),
            )
            with cur.copy(copy) as cp:
                for row in rows:
                    cp.write_row(row)
            counts[table] = len(rows)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data", required=True, help="Folder written by generate_synthetic.py"
    )
    args = parser.parse_args()

    import psycopg

    if "DATABASE_URL" not in os.environ:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv())  # searches upward from this file to the repo's .env
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        for table, n in load(conn, read_data(args.data)).items():
            print(f"{table:>20}: {n}")


if __name__ == "__main__":
    main()
