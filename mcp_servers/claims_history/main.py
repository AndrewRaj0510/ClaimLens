"""Claims-history MCP server: a customer's past claims and duplicate-photo checks.

Tools (all read-only):
- get_claim_history       past claims for a customer, plus counts the fraud
                          checks use (claims in the last 12 months, total paid)
- find_duplicate_photos   compare new photos' perceptual hashes with every photo
                          on file; a re-used photo is a classic fraud signal

Perceptual hashes (64-bit pHash, computed by the agent) change little when a
photo is re-saved, resized or lightly cropped, unlike a file hash. Two photos
are near-duplicates when their Hamming distance is small. Random pairs average
32 differing bits (std 4), so the default threshold of 8 is about six standard
deviations away from chance.

Privacy: a match on another customer's claim returns the claim ID and
`same_customer: false`, never the other customer's ID or details. Fraud labels
live in the `eval` schema and are never read here.
"""

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Annotated

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.transport_security import TransportSecuritySettings
from mcp_types import ToolAnnotations
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

DATABASE_URL = os.getenv("DATABASE_URL", "")
ALLOWED_HOSTS = os.getenv("MCP_ALLOWED_HOSTS", "localhost:*,127.0.0.1:*").split(",")
MAX_MATCHES = 50

CustomerId = Annotated[
    str, Field(pattern=r"^C\d{5}$", description="Customer ID, e.g. C00042")
]
PHash = Annotated[
    str, Field(pattern=r"^[0-9a-f]{16}$", description="64-bit pHash as 16 hex chars")
]


def phash_to_bigint(phash):
    """16 hex chars (unsigned 64-bit) -> signed bigint, as stored in claim_photos."""
    value = int(phash, 16)
    return value - (1 << 64) if value >= 1 << 63 else value


class PastClaim(BaseModel):
    claim_id: str
    policy_id: str
    peril: str
    description: str
    damages: list[dict]
    incident_date: date
    reported_date: date
    claimed_amount: float
    status: str
    paid_amount: float


class ClaimHistory(BaseModel):
    customer_id: str
    as_of: date
    claims: list[PastClaim]
    claims_last_12_months: int
    total_paid: float


class PhotoMatch(BaseModel):
    photo_index: int  # which of the submitted hashes matched
    claim_id: str
    same_customer: bool
    distance: int
    incident_date: date


class DuplicateCheck(BaseModel):
    max_distance: int
    matches: list[PhotoMatch]


HISTORY_SQL = """
SELECT claim_id, policy_id, peril, description, damages, incident_date, reported_date,
       claimed_amount, status, paid_amount
FROM claims
WHERE customer_id = %s AND incident_date BETWEEN %s AND %s
ORDER BY incident_date DESC, claim_id
"""

# Hamming distance = popcount(a XOR b). One pass over claim_photos per submitted
# hash: fine for thousands of photos; millions would need a BK-tree or
# multi-index hashing.
DUPLICATES_SQL = f"""
SELECT q.i - 1, p.claim_id, c.customer_id = %(customer)s,
       bit_count((p.phash # q.h)::bit(64)) AS distance, c.incident_date
FROM unnest(%(hashes)s::bigint[]) WITH ORDINALITY AS q(h, i)
JOIN claim_photos p ON bit_count((p.phash # q.h)::bit(64)) <= %(max_distance)s
JOIN claims c USING (claim_id)
ORDER BY distance, q.i, p.claim_id
LIMIT {MAX_MATCHES}
"""


@dataclass
class ClaimsStore:
    pool: object  # psycopg_pool.ConnectionPool

    def history(self, customer_id, start, end) -> list[PastClaim]:
        from psycopg.rows import dict_row

        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            rows = cur.execute(HISTORY_SQL, (customer_id, start, end)).fetchall()
        return [PastClaim(**r) for r in rows]

    def duplicates(self, hashes, customer_id, max_distance) -> list[PhotoMatch]:
        params = {
            "hashes": [phash_to_bigint(h) for h in hashes],
            "customer": customer_id,
            "max_distance": max_distance,
        }
        with self.pool.connection() as conn:
            rows = conn.execute(DUPLICATES_SQL, params).fetchall()
        return [
            PhotoMatch(
                photo_index=i,
                claim_id=c,
                same_customer=bool(s),
                distance=d,
                incident_date=dt,
            )
            for i, c, s, d, dt in rows
        ]


@asynccontextmanager
async def lifespan(server):
    from psycopg_pool import ConnectionPool

    # check: see services/gateway/main.py (dead connections after a DB restart).
    pool = ConnectionPool(
        DATABASE_URL,
        min_size=1,
        max_size=4,
        open=True,
        check=ConnectionPool.check_connection,
    )
    try:
        yield ClaimsStore(pool)
    finally:
        pool.close()


def store(ctx: Context) -> ClaimsStore:
    return ctx.request_context.lifespan_context


def summarise(customer_id, claims, as_of) -> ClaimHistory:
    year_ago = as_of - timedelta(days=365)
    return ClaimHistory(
        customer_id=customer_id,
        as_of=as_of,
        claims=claims,
        claims_last_12_months=sum(c.incident_date > year_ago for c in claims),
        total_paid=round(sum(c.paid_amount for c in claims), 2),
    )


def get_claim_history(
    customer_id: CustomerId,
    ctx: Context,
    years: Annotated[int, Field(ge=1, le=10)] = 5,
    as_of: date | None = None,
) -> ClaimHistory:
    """A customer's past claims (most recent first) over the last `years` years,
    with the number of claims in the 12 months before `as_of` (default: today)
    and the total paid out."""
    as_of = as_of or date.today()
    claims = store(ctx).history(customer_id, as_of - timedelta(days=365 * years), as_of)
    return summarise(customer_id, claims, as_of)


def find_duplicate_photos(
    phashes: Annotated[list[PHash], Field(min_length=1, max_length=20)],
    customer_id: CustomerId,
    ctx: Context,
    max_distance: Annotated[int, Field(ge=0, le=16)] = 8,
) -> DuplicateCheck:
    """Check whether any submitted photo (by perceptual hash) matches a photo
    from an earlier claim. `same_customer` tells you whether the earlier claim
    was this customer's; a match on another customer's claim is a strong fraud signal."""
    matches = store(ctx).duplicates(phashes, customer_id, max_distance)
    return DuplicateCheck(max_distance=max_distance, matches=matches)


def create_server(lifespan=lifespan) -> MCPServer:
    server = MCPServer(
        "claims_history",
        instructions="Look up past insurance claims and check photos for re-use.",
        lifespan=lifespan,
    )
    read_only = ToolAnnotations(read_only_hint=True, open_world_hint=False)
    for fn in (get_claim_history, find_duplicate_photos):
        server.add_tool(fn, annotations=read_only)

    @server.custom_route("/health", methods=["GET"])
    async def health(request):
        return JSONResponse({"status": "ok", "service": "claims_history"})

    return server


server = create_server()

if __name__ == "__main__":
    server.run(
        "streamable-http",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(allowed_hosts=ALLOWED_HOSTS),
    )
