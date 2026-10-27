import os
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from mcp import Client

from main import Damage, Tables, create_server, estimate, load_tables

pytestmark = pytest.mark.anyio

TABLES = Tables(
    labour_rates={"ME": 60.0, "CA": 90.0},
    multipliers={"economy": 1.0, "luxury": 2.0},
    operations={
        ("dent", "minor"): ("Paintless dent repair", 0.0, 1.5, 0.0),
        ("crack", "moderate"): ("Replace bumper cover", 380.0, 2.5, 3.0),
    },
    paint_materials_per_hour=45.0,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_cost_formula_worked_by_hand():
    est = estimate(
        TABLES, [Damage(damage_class="crack", severity="moderate")], "economy", "ME"
    )
    # 380 parts + (2.5 + 3.0) h x $60 + 3.0 h x $45 paint materials
    assert est.total == 380 + 5.5 * 60 + 3 * 45 == 845
    assert (est.low, est.high) == (round(845 * 0.85, 2), round(845 * 1.2, 2))


def test_segment_multiplies_parts_only():
    economy = estimate(
        TABLES, [Damage(damage_class="crack", severity="moderate")], "economy", "ME"
    )
    luxury = estimate(
        TABLES, [Damage(damage_class="crack", severity="moderate")], "luxury", "ME"
    )
    assert luxury.total - economy.total == 380  # parts doubled, labour unchanged


def test_line_items_sum_to_total():
    damages = [
        Damage(damage_class="crack", severity="moderate"),
        Damage(damage_class="dent", severity="minor"),
    ]
    est = estimate(TABLES, damages, "economy", "CA")
    assert len(est.line_items) == 2
    assert est.total == pytest.approx(sum(i.cost for i in est.line_items))
    assert est.line_items[1].cost == 1.5 * 90


def test_unknown_state_uses_average_rate_and_says_so():
    est = estimate(
        TABLES, [Damage(damage_class="dent", severity="minor")], "economy", "ZZ"
    )
    assert est.labour_rate_per_hour == 75.0
    assert "average" in est.assumptions[0]


class FixedTables:
    """Stands in for TableCache."""

    def __init__(self, tables):
        self.tables = tables

    def get(self):
        return self.tables


def server_with(tables):
    @asynccontextmanager
    async def lifespan(server):
        yield FixedTables(tables)

    return create_server(lifespan=lifespan)


async def test_tool_returns_structured_estimate():
    args = {
        "damages": [{"damage_class": "dent", "severity": "minor"}],
        "vehicle_segment": "economy",
        "state": "ME",
    }
    async with Client(server_with(TABLES)) as client:
        tools = (await client.list_tools()).tools
        result = await client.call_tool("estimate_repair_cost", args)
    assert [t.name for t in tools] == ["estimate_repair_cost"]
    assert tools[0].annotations.read_only_hint
    assert result.structured_content["total"] == 90.0


@pytest.mark.parametrize(
    "bad",
    [
        {"damages": []},  # nothing to price
        {
            "damages": [{"damage_class": "rust", "severity": "minor"}]
        },  # not a known class
        {"state": "Maine"},  # not a state code
        {"vehicle_segment": "spaceship"},
    ],
)
async def test_invalid_arguments_are_rejected(bad):
    args = {
        "damages": [{"damage_class": "dent", "severity": "minor"}],
        "vehicle_segment": "economy",
        "state": "ME",
        **bad,
    }
    async with Client(server_with(TABLES)) as client:
        result = await client.call_tool("estimate_repair_cost", args)
    assert result.is_error


async def test_missing_operation_is_a_clear_error():
    args = {
        "damages": [{"damage_class": "tire flat", "severity": "severe"}],
        "vehicle_segment": "economy",
        "state": "ME",
    }
    async with Client(server_with(TABLES)) as client:
        result = await client.call_tool("estimate_repair_cost", args)
    assert result.is_error
    assert "tire flat / severe" in result.content[0].text


TEST_DB = os.getenv("TEST_DATABASE_URL")
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


@pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to run")
def test_tables_from_db_agree_with_the_generator(monkeypatch):
    # The fraud labels call a claim "inflated" relative to the generator's
    # repair_cost(); the pricing server must compute the same number.
    import psycopg

    monkeypatch.syspath_prepend(str(SCRIPTS))
    from generate_synthetic import DAMAGE_CLASSES, SEVERITIES, generate, repair_cost
    from load_synthetic import load

    with psycopg.connect(TEST_DB) as conn:
        conn.execute((SCRIPTS.parent / "db" / "schema.sql").read_text())
        load(conn, generate(seed=1, n_customers=50))
        tables = load_tables(conn)
    for cls in DAMAGE_CLASSES:
        for sev in SEVERITIES:
            est = estimate(
                tables, [Damage(damage_class=cls, severity=sev)], "suv", "CA"
            )
            assert est.total == pytest.approx(
                repair_cost(cls, sev, "suv", "CA"), abs=0.01
            )


@pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to run")
def test_cache_refuses_empty_tables_and_retries_later():
    import psycopg

    from main import TableCache

    with psycopg.connect(TEST_DB) as conn:
        conn.execute((SCRIPTS.parent / "db" / "schema.sql").read_text())
        conn.execute(
            "TRUNCATE labour_rates, segment_multipliers, repair_operations, pricing_settings"
        )
    cache = TableCache(TEST_DB)
    with pytest.raises(Exception, match="empty"):
        cache.get()
    assert cache.tables is None  # nothing cached, so a later call can succeed
