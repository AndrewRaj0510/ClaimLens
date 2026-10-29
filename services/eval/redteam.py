"""Red-team suite: the Phase 4 set and the held-out set, against guardrails.

    python run.py redteam [--guardrails-url http://guardrails:8000]

Both sets are generated from seeded scripts when missing (no dataset needed),
then run with scripts/run_redteam.py's runner, against the running service or
the guardrails app in-process. Two numbers matter, and they mean different
things:
- dev: the Phase 4 set. The rules were fixed on it, so it's a regression
  test: nothing that was caught may slip.
- heldout: written for Phase 5 and never tuned on, so its catch rate is the
  honest estimate of what the guardrails catch.
"""

import json

import paths

SETS = {
    "dev": ("generate_redteam", paths.DATA / "redteam"),
    "heldout": ("generate_redteam_heldout", paths.DATA / "redteam-heldout"),
}


def ensure_cases(name):
    """Generate a set if its cases aren't on disk, or are older than the script
    that generates them (seeded, so an unchanged script gives the same set)."""
    import importlib

    module_name, folder = SETS[name]
    cases_file = folder / "cases.jsonl"
    script = paths.SCRIPTS / f"{module_name}.py"
    if cases_file.exists() and cases_file.stat().st_mtime >= script.stat().st_mtime:
        return folder
    generator = importlib.import_module(module_name)
    cases = generator.generate()
    with open(folder / "cases.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in cases)
    return folder


def misses(results):
    """What a reader needs from the per-case results: the ids that went wrong."""
    return {
        "missed": [
            r["id"]
            for r in results
            if r["attack"] and not r["caught"] and not r["known_limit"]
        ],
        "known_limits_missed": [
            r["id"]
            for r in results
            if r["attack"] and not r["caught"] and r["known_limit"]
        ],
        "known_limits_caught": [
            r["id"] for r in results if r["attack"] and r["caught"] and r["known_limit"]
        ],
        "false_positives": [
            r["id"] for r in results if not r["attack"] and not r["caught"]
        ],
    }


def run(guardrails_url=None, sets=("dev", "heldout")):
    import run_redteam

    out = {}
    with run_redteam.client_for(guardrails_url) as client:
        for name in sets:
            folder = ensure_cases(name)
            summary, results = run_redteam.run_cases(
                client, run_redteam.load_cases(folder), folder
            )
            (folder / "report.json").write_text(
                json.dumps({"summary": summary, "results": results}, indent=1)
            )
            out[name] = summary | misses(results)
    return out
