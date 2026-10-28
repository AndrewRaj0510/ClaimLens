"""Cross-service contract tests.

Each service is its own Docker image, so a few small helpers are copied rather
than shared through a package. These tests load every copy and check they agree,
so a change to one that isn't made to the others fails here instead of in
production (e.g. an agent that rejects photos the gateway accepted).
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load(relpath, name):
    """Import a file under a unique module name, with its folder on sys.path
    (for its own sibling imports)."""
    path = ROOT / relpath
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path.parent))


PHASH_COPIES = {
    "loader": "scripts/load_synthetic.py",
    "claims_history": "mcp_servers/claims_history/main.py",
    "agent": "services/agent/adapters.py",
}


@pytest.mark.parametrize(
    "h",
    [
        "0000000000000000",
        "7fffffffffffffff",
        "8000000000000000",
        "ffffffffffffffff",
        "5807f4a4135ce2f3",
    ],
)
def test_every_phash_encoding_agrees(h):
    # The loader writes hashes, the agent writes new ones, claims_history queries
    # them: one different encoding and duplicates silently stop matching.
    values = {
        who: load(path, f"contract_{who}").phash_to_bigint(h)
        for who, path in PHASH_COPIES.items()
    }
    assert len(set(values.values())) == 1, values


def test_upload_limit_is_the_same_everywhere(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "15")
    limits = {
        "gateway": int(
            load("services/gateway/common.py", "contract_gw").MAX_UPLOAD_MB
            * 1024
            * 1024
        ),
        "vision": load("services/vision/analysis.py", "contract_vision").max_bytes(),
        "agent": load(
            "services/agent/adapters.py", "contract_agent_limits"
        ).PhotoStore.MAX_BYTES,
        "guardrails": load(
            "services/guardrails/main.py", "contract_guardrails"
        ).MAX_UPLOAD_BYTES,
    }
    assert set(limits.values()) == {15 * 1024 * 1024}, limits


def test_frontend_upload_checks_match_the_gateway():
    # The frontend warns before uploading (frontend/lib/photos.ts). It can't
    # read env vars, so it mirrors the defaults; if they drift, customers get
    # "fine" in the browser and 413/415 from the gateway, or the reverse.
    import re

    source = (ROOT / "frontend" / "lib" / "photos.ts").read_text()

    def const(name):
        return re.search(rf"export const {name} = (.+?);", source).group(1)

    gateway = load("services/gateway/common.py", "contract_gw_types")
    claims = load("services/gateway/claims.py", "contract_gw_claims")
    assert (
        set(re.findall(r'"([^"]+)"', const("ALLOWED_TYPES"))) == gateway.ALLOWED_TYPES
    )
    assert float(const("MAX_MB")) == gateway.MAX_UPLOAD_MB
    assert int(const("MAX_PHOTOS")) == claims.MAX_PHOTOS


MCP_SERVERS = {
    "policy": "mcp_servers/policy/main.py",
    "pricing": "mcp_servers/pricing/main.py",
    "claims_history": "mcp_servers/claims_history/main.py",
}
# A tool whose name sounds like it changes something. There must be none: the
# agent's only route to a decision is the human-approval interrupt.
WRITE_VERBS = re.compile(
    r"approve|pay|decide|reject|settle|update|delete|insert|write|create|set_|close"
)


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_every_mcp_tool_is_read_only_and_the_agent_allowlist_matches(
    anyio_backend,
):
    from contextlib import asynccontextmanager

    from mcp import Client

    @asynccontextmanager
    async def no_db(server):
        yield None

    tools = {}
    for name, path in MCP_SERVERS.items():
        server = load(path, f"contract_mcp_{name}").create_server(lifespan=no_db)
        async with Client(server) as client:
            listed = (await client.list_tools()).tools
        for t in listed:
            assert t.annotations and t.annotations.read_only_hint, (name, t.name)
            assert not WRITE_VERBS.search(t.name), (name, t.name)
            tools[(name, t.name)] = t
    allowed = load("services/agent/tools.py", "contract_agent_tools").ALLOWED
    for node, pairs in allowed.items():
        assert pairs <= set(tools), (node, pairs - set(tools))
