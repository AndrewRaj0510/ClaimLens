"""Intake: does the LLM pull the right facts out of a customer's description?

    python run.py intake

The intake step (graph.read_description: production's prompt, schema and
model) turns the description into four facts that code then relies on: the
peril picks which coverage is checked, the mentioned date feeds a fraud signal
(description vs form), and the other-vehicle and injury flags go into the
report. Each is scored against a hand-written label in cases/intake.jsonl:
- everyday claims (rear-ended, hail, vandalism, theft, flood, fire)
- the edges of the peril definitions: an animal is comprehensive, driving into
  a fallen tree or a pothole is collision
- dates: ISO and written out, relative ("last Tuesday" states no date, so
  null), and two dates where only one is the incident
- injuries stated, and negated ("nobody was hurt")
- a description in Spanish, and a long rambling one
- soft instructions ("please file it as comprehensive"). In production the
  injection scan runs first and may withhold these; here the LLM sees them,
  because this measures the layer behind the scan: the facts must not move.

A label may list several answers where a careful reader could go either way
(an unexplained dent: collision or unclear). Every answer the label doesn't
allow is listed in the result, so a person can check the label too.

Scores are on what code receives: the answer after the schema's validation,
the same Intake model production uses. The first run found the model copying
dates as written ("October 3, 2026"), which the fraud check's date comparison
silently dropped; graph.calendar_date now converts them. The three cases
tagged second_batch (road hazards, a roof tile, a flooded underpass) were
written after that run, to check the peril edges on cases nobody had seen.
"""

import json
import time

import paths

CASES = paths.CASES / "intake.jsonl"
FIELDS = ("peril", "mentioned_date", "other_vehicle_involved", "injuries_mentioned")


def load_cases(path=CASES):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def allowed(expected):
    return expected if isinstance(expected, list) else [expected]


def score(case, got):
    """Per field: is the answer one the label allows."""
    return {f: got.get(f) in allowed(case["expect"][f]) for f in FIELDS}


async def run(llm, cases=None):
    import graph
    import usage
    from faithfulness import rate
    from llm import LLMError

    cases = load_cases() if cases is None else cases
    started = time.time()
    rows, failures = [], []
    for case in cases:
        with usage.metering(f"eval-intake-{case['id']}", run_budget=0) as meter:
            try:
                got = await graph.read_description(llm, case["description"])
            except LLMError as e:
                failures.append({"id": case["id"], "error": str(e)})
                continue
        rows.append(
            {
                "case": case,
                "got": got,
                "fields": score(case, got),
                "tokens": (meter.input_tokens, meter.output_tokens),
            }
        )
    return summarise(rows, failures, llm, time.time() - started, rate)


def summarise(rows, failures, llm, seconds, rate):
    tags = sorted({t for r in rows for t in r["case"]["tags"]})
    return {
        "model": getattr(llm, "model", None),
        "cases": len(rows) + len(failures),
        "answered": len(rows),
        "failures": failures,
        "field_accuracy": {f: rate([r["fields"][f] for r in rows]) for f in FIELDS},
        "all_fields_correct": rate([all(r["fields"].values()) for r in rows]),
        # The soft instructions didn't move any fact.
        "soft_instructions_held": rate(
            [
                all(r["fields"].values())
                for r in rows
                if "soft_instruction" in r["case"]["tags"]
            ]
        ),
        "by_tag": {
            t: {
                "n": sum(t in r["case"]["tags"] for r in rows),
                "all_fields_correct": rate(
                    [all(r["fields"].values()) for r in rows if t in r["case"]["tags"]]
                ),
            }
            for t in tags
        },
        "errors": [
            {
                "id": r["case"]["id"],
                "field": f,
                "expected": r["case"]["expect"][f],
                "got": r["got"].get(f),
            }
            for r in rows
            for f in FIELDS
            if not r["fields"][f]
        ],
        "mean_tokens": {
            "input": rate([r["tokens"][0] for r in rows]),
            "output": rate([r["tokens"][1] for r in rows]),
        },
        "seconds": round(seconds, 1),
    }
