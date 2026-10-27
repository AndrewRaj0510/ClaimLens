"""Cross-service contract tests.

Each service is its own Docker image, so a few small helpers are copied rather
than shared through a package. These tests load every copy and check they agree,
so a change to one that isn't made to the others fails here instead of in
production (e.g. an agent that rejects photos the gateway accepted).
"""

import importlib.util
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
    }
    assert set(limits.values()) == {15 * 1024 * 1024}, limits
