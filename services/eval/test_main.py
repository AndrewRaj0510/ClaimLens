"""The eval API and the gate logic."""

import json
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

import faithfulness as fa
import gate
import redteam
from llm import LLMError
from main import create_app

BASELINE = {
    "_comment": "ignored",
    "redteam": {
        "heldout.catch_rate": {"min": 0.9, "why": "test"},
        "heldout.false_positive_rate": {"max": 0.05},
    },
    "faithfulness": {"judge_catch_rate": {"min": 0.8}},
}


def test_gate_passes_fails_and_skips():
    ok_result = {"heldout": {"catch_rate": 0.93, "false_positive_rate": 0.0}}
    passed, rows = gate.check(BASELINE, {"redteam": ok_result})
    assert passed
    assert {r["status"] for r in rows if r["suite"] == "redteam"} == {"pass"}
    assert [r["status"] for r in rows if r["suite"] == "faithfulness"] == ["skipped"]

    worse = {"heldout": {"catch_rate": 0.85, "false_positive_rate": 0.1}}
    passed, rows = gate.check(BASELINE, {"redteam": worse})
    assert not passed
    assert [r["status"] for r in rows if r["suite"] == "redteam"] == ["fail", "fail"]


def test_a_required_suite_without_a_result_fails():
    ok_result = {"heldout": {"catch_rate": 0.93, "false_positive_rate": 0.0}}
    passed, rows = gate.check(
        BASELINE, {"redteam": ok_result}, require=["faithfulness"]
    )
    assert not passed
    assert [r["suite"] for r in rows if r["status"] == "fail"] == ["faithfulness"]


def test_a_metric_missing_from_the_result_fails():
    passed, rows = gate.check(BASELINE, {"redteam": {"heldout": {"catch_rate": 0.95}}})
    assert not passed
    assert (
        rows[1]["metric"] == "heldout.false_positive_rate" and rows[1]["value"] is None
    )


def test_the_committed_baseline_is_well_formed():
    baseline = json.loads(gate.BASELINE.read_text())
    for suite, gates in baseline.items():
        if suite.startswith("_"):
            continue
        assert suite in ("redteam", "faithfulness", "detection")
        for metric, rule in gates.items():
            assert {"min", "max"} & set(rule), metric
            assert rule.get("why"), f"{suite}.{metric} needs a reason"


def test_the_committed_results_pass_their_gates():
    """The same check CI runs: results in the repo must pass the baseline in the repo."""
    passed, rows = gate.check(
        json.loads(gate.BASELINE.read_text()), gate.load_results()
    )
    assert passed, [r for r in rows if r["status"] == "fail"]


def test_misses_lists_what_went_wrong():
    results = [
        {"id": "a", "attack": True, "caught": False, "known_limit": False},
        {"id": "b", "attack": True, "caught": False, "known_limit": True},
        {"id": "c", "attack": True, "caught": True, "known_limit": True},
        {"id": "d", "attack": False, "caught": False, "known_limit": False},
        {"id": "e", "attack": True, "caught": True, "known_limit": False},
    ]
    assert redteam.misses(results) == {
        "missed": ["a"],
        "known_limits_missed": ["b"],
        "known_limits_caught": ["c"],
        "false_positives": ["d"],
    }


# ---- API ----


class FakeJudge:
    model = "fake"

    def __init__(self, down=False):
        self.down = down

    async def structured(self, system, user, schema):
        if self.down:
            raise LLMError("down")
        return schema.model_validate(
            {"statements": [{"text": "x", "verdict": "supported", "reason": ""}]}
        )


@pytest.fixture
def client(tmp_path):
    def make(down=False):
        @asynccontextmanager
        async def lifespan(app):
            app.state.judge = FakeJudge(down)
            yield

        baseline = tmp_path / "config" / "baseline.json"
        baseline.parent.mkdir(exist_ok=True)
        baseline.write_text(json.dumps(BASELINE))
        return TestClient(
            create_app(lifespan=lifespan, results_dir=tmp_path, baseline=baseline)
        )

    return make


def body():
    import graph

    facts = graph.report_facts(fa.build_cases(1)[0])
    report = {
        "customer_message": "Your claim has been approved.",
        "adjuster_summary": f"Policy {facts['policy_id']}. See clause C9.",
    }
    return {"facts": facts, "report": report}


def test_health(client):
    with client() as c:
        assert c.get("/health").json()["service"] == "eval"


def test_grounding_and_judge(client):
    with client() as c:
        g = c.post("/grounding", json=body()).json()
        assert g["clauses"] == ["C9"] and not g["ok"]
        j = c.post("/judge", json=body()).json()
        assert j["grounding"]["clauses"] == ["C9"]
        assert j["judge"]["faithful"]


def test_judge_down_is_503(client):
    with client(down=True) as c:
        assert c.post("/judge", json=body()).status_code == 503


def test_oversized_facts_are_refused(client):
    b = body()
    b["facts"]["padding"] = "x" * 60_000
    with client() as c:
        assert c.post("/grounding", json=b).status_code == 413


def test_results_and_gates(client, tmp_path):
    with client() as c:
        assert c.get("/results").json()["results"] == []
        assert c.get("/results/redteam").status_code == 404
        assert c.get("/results/secrets").status_code == 422
        (tmp_path / "redteam.json").write_text(
            json.dumps(
                {
                    "suite": "redteam",
                    "ran_at": "2026-10-29T00:00:00+00:00",
                    "commit": "abc",
                    "result": {},
                }
            )
        )
        listed = c.get("/results").json()["results"]
        assert listed[0]["suite"] == "redteam" and listed[0]["gates"] == "fail"
        assert c.get("/results/redteam").json()["commit"] == "abc"
        assert not c.get("/gates").json()["passed"]


def test_a_non_numeric_value_fails_instead_of_crashing():
    baseline = {"redteam": {"heldout": {"min": 0.5, "why": "a path to a dict"}}}
    passed, rows = gate.check(baseline, {"redteam": {"heldout": {"catch_rate": 0.9}}})
    assert not passed and rows[0]["status"] == "fail"
    passed, _ = gate.check(
        {"redteam": {"ok": {"min": 1, "why": "a flag"}}}, {"redteam": {"ok": True}}
    )
    assert not passed  # True >= 1 in Python; a flag isn't a metric


def test_ensure_cases_regenerates_when_the_generator_changed(tmp_path, monkeypatch):
    import os

    calls = []

    class Generator:
        @staticmethod
        def generate():
            calls.append(1)
            return [{"id": "x"}]

    script = tmp_path / "gen.py"
    script.write_text("# generator")
    monkeypatch.setattr(redteam.paths, "SCRIPTS", tmp_path)
    monkeypatch.setitem(redteam.SETS, "t", ("gen", tmp_path))
    monkeypatch.setitem(__import__("sys").modules, "gen", Generator)
    redteam.ensure_cases("t")
    redteam.ensure_cases("t")  # up to date: not regenerated
    assert len(calls) == 1
    newer = (tmp_path / "cases.jsonl").stat().st_mtime + 10
    os.utime(script, (newer, newer))
    redteam.ensure_cases("t")
    assert len(calls) == 2
