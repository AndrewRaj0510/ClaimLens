"""Eval gates: fail the build when a stored result falls below its baseline.

    python run.py gate [--require redteam,faithfulness]

baseline.json holds, per suite, dotted paths into that suite's result with a
floor ("min") or a ceiling ("max"). The floors are the measured values minus a
stated tolerance (see each entry's "why"), so a real regression fails and run-
to-run noise doesn't. Raising a baseline is a reviewed change to baseline.json,
like changing a test.

A suite without a result file is skipped, not failed, because some can't run
everywhere (detection needs CarDD; faithfulness needs an LLM key). CI passes
--require for the suites it just ran, so a suite that should have produced a
result and didn't is a failure.
"""

import json
import os
from pathlib import Path

import paths

BASELINE = Path(__file__).resolve().parent / "baseline.json"


def lookup(result, dotted):
    value = result
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def check(baseline, results, require=()):
    """Returns (passed, rows). rows: suite, metric, value, limit, status."""
    rows = []
    for suite, gates in baseline.items():
        if suite.startswith("_"):
            continue
        result = results.get(suite)
        if result is None:
            status = "fail" if suite in require else "skipped"
            rows.append(
                {
                    "suite": suite,
                    "metric": "(result)",
                    "value": None,
                    "limit": "",
                    "status": status,
                }
            )
            continue
        for metric, rule in gates.items():
            value = lookup(result, metric)
            if value is None:
                rows.append(
                    {
                        "suite": suite,
                        "metric": metric,
                        "value": None,
                        "limit": _limit(rule),
                        "status": "fail",
                    }
                )
                continue
            # bool is an int in Python; a flag or a dict here is a typo'd path.
            numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
            ok = numeric and (
                ("min" not in rule or value >= rule["min"])
                and ("max" not in rule or value <= rule["max"])
            )
            rows.append(
                {
                    "suite": suite,
                    "metric": metric,
                    "value": value,
                    "limit": _limit(rule),
                    "status": "pass" if ok else "fail",
                }
            )
    return all(r["status"] != "fail" for r in rows), rows


def _limit(rule):
    return " ".join(
        f"{'>=' if k == 'min' else '<='} {rule[k]}" for k in ("min", "max") if k in rule
    )


def load_results(results_dir=paths.RESULTS):
    out = {}
    for p in Path(results_dir).glob("*.json"):
        record = json.loads(p.read_text())
        if isinstance(record, dict) and "result" in record:  # run.py's records only
            out[record.get("suite", p.stem)] = record["result"]
    return out


def markdown(rows, passed):
    lines = [
        f"### Eval gates: {'passed' if passed else 'FAILED'}",
        "",
        "| suite | metric | value | limit | |",
        "|---|---|---|---|---|",
    ]
    icon = {"pass": "ok", "fail": "**FAIL**", "skipped": "skipped (no result)"}
    for r in rows:
        value = "" if r["value"] is None else r["value"]
        lines.append(
            f"| {r['suite']} | `{r['metric']}` | {value} | {r['limit']} | {icon[r['status']]} |"
        )
    return "\n".join(lines)


def main(require=()):
    baseline = json.loads(BASELINE.read_text())
    passed, rows = check(baseline, load_results(), require)
    text = markdown(rows, passed)
    print(text)
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0 if passed else 1
