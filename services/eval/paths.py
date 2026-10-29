"""Where the code under test lives, and how eval imports it.

The eval service measures the other services' own code, not copies of it: the
vision service's ONNX post-processing, the agent's severity rules and report
prompt, the pricing server's arithmetic. So it imports them from the repo (the
Docker image keeps the repo layout, see Dockerfile). Sibling service folders
are appended to sys.path, after eval's own, because several have a main.py;
the pricing server is loaded from its file under its own name for the same
reason.
"""

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(os.getenv("REPO_ROOT", Path(__file__).resolve().parent.parent.parent))
AGENT = ROOT / "services" / "agent"
VISION = ROOT / "services" / "vision"
GUARDRAILS = ROOT / "services" / "guardrails"
PRICING = ROOT / "mcp_servers" / "pricing"
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"
RESULTS = Path(
    os.getenv("EVAL_RESULTS_DIR", Path(__file__).resolve().parent / "results")
)

for path in (AGENT, VISION, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.append(str(path))


def load_file(name, path):
    """Import a module from a file under a name of our choosing."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pricing():
    """mcp_servers/pricing/main.py, as module `pricing_server`."""
    return load_file("pricing_server", PRICING / "main.py")
