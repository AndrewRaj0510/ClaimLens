"""Pricing MCP server: repair cost estimates from parts and labour tables.

Tool (read-only):
- estimate_repair_cost   line items and a cost range for a list of damages on a
                         vehicle segment in a state

The arithmetic is done here, not by the LLM: a model asked to add up repair
costs will sometimes get it wrong, and an estimate has to be reproducible.

cost per damage = parts x segment multiplier
                + (labour hours + paint hours) x state labour rate
                + paint hours x paint materials rate

The range (-15% / +20%) reflects how much body-shop quotes vary for the same
job; it is a stated assumption, not a statistic.
"""

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp_types import ToolAnnotations
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

DATABASE_URL = os.getenv("DATABASE_URL", "")
ALLOWED_HOSTS = os.getenv("MCP_ALLOWED_HOSTS", "localhost:*,127.0.0.1:*").split(",")
RANGE_LOW, RANGE_HIGH = 0.85, 1.20

DamageClass = Literal[
    "dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"
]
Severity = Literal["minor", "moderate", "severe"]
Segment = Literal["economy", "midsize", "suv", "truck", "luxury"]
State = Annotated[
    str, Field(pattern=r"^[A-Z]{2}$", description="US state code, e.g. ME")
]


class Damage(BaseModel):
    damage_class: DamageClass
    severity: Severity


class LineItem(BaseModel):
    damage_class: str
    severity: str
    operation: str
    parts: float
    labour_hours: float
    paint_hours: float
    cost: float


class Estimate(BaseModel):
    currency: str
    labour_rate_per_hour: float
    line_items: list[LineItem]
    total: float
    low: float
    high: float
    assumptions: list[str]


@dataclass
class Tables:
    """The pricing tables, loaded once at startup (they're small and static)."""

    labour_rates: dict  # state -> rate
    multipliers: dict  # segment -> parts multiplier
    operations: dict  # (class, severity) -> (operation, parts, labour h, paint h)
    paint_materials_per_hour: float


def load_tables(conn) -> Tables:
    rates = dict(conn.execute("SELECT state, rate_per_hour FROM labour_rates"))
    multipliers = dict(
        conn.execute("SELECT segment, parts_multiplier FROM segment_multipliers")
    )
    operations = {
        (cls, sev): (op, parts, labour, paint)
        for cls, sev, op, parts, labour, paint in conn.execute(
            "SELECT damage_class, severity, operation, parts_usd, labour_hours, paint_hours"
            " FROM repair_operations"
        )
    }
    settings = dict(conn.execute("SELECT key, value FROM pricing_settings"))
    settings.setdefault(
        "paint_materials_per_hour", 0
    )  # empty tables are caught by the caller
    return Tables(
        labour_rates={k: float(v) for k, v in rates.items()},
        multipliers={k: float(v) for k, v in multipliers.items()},
        operations={
            k: (op, float(p), float(lh), float(ph))
            for k, (op, p, lh, ph) in operations.items()
        },
        paint_materials_per_hour=float(settings["paint_materials_per_hour"]),
    )


class TableCache:
    """Loads the tables on first use and keeps them. Lazy, so the server starts
    (and passes its healthcheck) before the pricing data has been loaded."""

    def __init__(self, database_url):
        self.database_url = database_url
        self.tables = None

    def get(self) -> Tables:
        if self.tables is None:
            import psycopg

            with psycopg.connect(self.database_url) as conn:
                tables = load_tables(conn)
            if not tables.labour_rates or not tables.operations:
                raise ToolError("Pricing tables are empty; load the pricing data first")
            self.tables = tables
        return self.tables


@asynccontextmanager
async def lifespan(server):
    yield TableCache(DATABASE_URL)


def estimate(tables: Tables, damages, segment, state) -> Estimate:
    assumptions = []
    if state in tables.labour_rates:
        rate = tables.labour_rates[state]
    else:
        rate = sum(tables.labour_rates.values()) / len(tables.labour_rates)
        assumptions.append(
            f"No labour rate for {state}; used the average, ${rate:.2f}/h."
        )
    multiplier = tables.multipliers[segment]
    items = []
    for d in damages:
        key = (d.damage_class, d.severity)
        if key not in tables.operations:
            raise ToolError(
                f"No repair operation priced for {d.damage_class} / {d.severity}"
            )
        op, parts, labour_h, paint_h = tables.operations[key]
        cost = (
            parts * multiplier
            + (labour_h + paint_h) * rate
            + paint_h * tables.paint_materials_per_hour
        )
        items.append(
            LineItem(
                damage_class=d.damage_class,
                severity=d.severity,
                operation=op,
                parts=round(parts * multiplier, 2),
                labour_hours=labour_h,
                paint_hours=paint_h,
                cost=round(cost, 2),
            )
        )
    total = round(sum(i.cost for i in items), 2)
    assumptions.append(
        f"Range is {RANGE_LOW:.0%}-{RANGE_HIGH:.0%} of the total, for variation between body shops."
    )
    return Estimate(
        currency="USD",
        labour_rate_per_hour=rate,
        line_items=items,
        total=total,
        low=round(total * RANGE_LOW, 2),
        high=round(total * RANGE_HIGH, 2),
        assumptions=assumptions,
    )


def estimate_repair_cost(
    damages: Annotated[list[Damage], Field(min_length=1, max_length=20)],
    vehicle_segment: Segment,
    state: State,
    ctx: Context,
) -> Estimate:
    """Estimate the repair cost of a list of damages (class and severity) for a
    vehicle segment in a US state. Returns line items, a total and a low/high range.
    Use this for any cost figure instead of estimating it yourself."""
    tables = ctx.request_context.lifespan_context.get()
    return estimate(tables, damages, vehicle_segment, state)


def create_server(lifespan=lifespan) -> MCPServer:
    server = MCPServer(
        "pricing",
        instructions="Estimate car repair costs from parts and labour tables.",
        lifespan=lifespan,
    )
    server.add_tool(
        estimate_repair_cost,
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )

    @server.custom_route("/health", methods=["GET"])
    async def health(request):
        return JSONResponse({"status": "ok", "service": "pricing"})

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
