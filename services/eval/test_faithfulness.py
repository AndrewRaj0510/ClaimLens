"""Faithfulness suite without an LLM: seeded cases, the grounding check, the
planted errors, and the run loop with a fake writer and a fake judge."""

import json
import re

import pytest

import faithfulness as fa

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def facts_of(state):
    import graph

    return graph.report_facts(state)


def honest_report(facts):
    d = facts["decision"]
    outcome = "approved" if d["action"] in ("approve", "edit") else "not approved"
    clauses = ", ".join(c["id"] for c in facts["cited_clauses"])
    return {
        "customer_message": f"Your claim has been {outcome}. Payment: ${d['amount']:,.2f}.",
        "adjuster_summary": (
            f"Policy {facts['policy_id']}, incident {facts['incident_date']}. "
            f"Estimate ${facts['estimate']['total']:,.2f}, deductible "
            f"${facts['coverage']['deductible']:,}. Clauses {clauses}."
        ),
    }


def test_cases_are_seeded_and_cover_every_scenario():
    a, b = fa.build_cases(10, seed=3), fa.build_cases(10, seed=3)
    assert a == b
    assert [s["scenario"] for s in a[:5]] == list(fa.SCENARIOS)
    for s in a:
        facts = facts_of(s)
        assert set(facts) >= {"decision", "payable", "estimate", "cited_clauses"}
        assert len(facts["cited_clauses"]) == 2
        if s["scenario"] == "deductible":
            assert facts["payable"]["expected"] == 0
        if s["scenario"] == "deny":
            assert not facts["coverage"]["covered"] and facts["decision"]["amount"] == 0


def test_an_honest_report_is_grounded():
    for s in fa.build_cases(5):
        facts = facts_of(s)
        assert fa.grounding(honest_report(facts), facts)["ok"], s["scenario"]


def test_grounding_flags_what_the_facts_dont_contain():
    facts = facts_of(fa.build_cases(1)[0])
    report = honest_report(facts)
    total = facts["estimate"]["total"]
    bad = {
        **report,
        "customer_message": report["customer_message"]
        + f" We add ${total + 250:,.2f}. See clause C7. Incident on September 2, 2025."
        + " Policy P99999.",
    }
    g = fa.grounding(bad, facts)
    assert not g["ok"]
    assert g["amounts"] and g["clauses"] == ["C7"]
    assert g["dates"] == ["2025-09-02"] and g["ids"] == ["P99999"]
    # A rounded amount and a long-form date of the incident are fine.
    rounded = {
        **report,
        "customer_message": f"About ${round(total):,} on " + _long(facts),
    }
    assert fa.grounding(rounded, facts)["ok"]


def _long(facts):
    from datetime import date

    d = date.fromisoformat(facts["incident_date"])
    return d.strftime("%B ") + str(d.day) + d.strftime(", %Y")


def test_planted_errors_and_what_grounding_can_see():
    facts = facts_of(fa.build_cases(1)[0])  # an approval
    report = honest_report(facts)
    planted = dict(fa.corruptions(report, facts))
    assert set(planted) == {
        "wrong_amount",
        "wrong_date",
        "invented_clause",
        "invented_event",
        "wrong_outcome",
    }
    assert "denied" in planted["wrong_outcome"]["customer_message"]
    seen = {k: not fa.grounding(v, facts)["ok"] for k, v in planted.items()}
    # Code catches invented numbers, dates and IDs, not wrong words or events:
    # that's what the judge is for.
    assert seen == {
        "wrong_amount": True,
        "wrong_date": True,
        "invented_clause": True,
        "invented_event": False,
        "wrong_outcome": False,
    }


def test_flip_outcome():
    assert (
        fa.flip_outcome("Your claim was denied.", "reject")
        == "Your claim was approved."
    )
    assert fa.flip_outcome("We approved it.", "approve") == "We denied it."
    assert fa.flip_outcome("Thanks for waiting.", "approve") is None


def test_score():
    out = fa.score(
        {
            "statements": [
                {"text": "a", "verdict": "supported", "reason": ""},
                {"text": "b", "verdict": "unsupported", "reason": "not in facts"},
            ]
        }
    )
    assert out["faithfulness"] == 0.5 and not out["faithful"]
    assert fa.score({"statements": []})["faithful"]


class FakeWriter:
    model = "fake-writer"

    async def structured(self, system, user, schema):
        facts = json.loads(
            re.search(r"<facts>\n(.*)\n</facts>", user, re.DOTALL).group(1)
        )
        return schema.model_validate(honest_report(facts))


class FakeJudge:
    """Flags the planted sentences it recognises, like a decent judge would."""

    model = "fake-judge"
    MARKERS = ("will be sent", "records show", "clause C9", "police report", "denied")

    async def structured(self, system, user, schema):
        report = re.search(r"<report>\n(.*)\n</report>", user, re.DOTALL).group(1)
        facts = json.loads(
            re.search(r"<facts>\n(.*)\n</facts>", user, re.DOTALL).group(1)
        )
        statements = [{"text": "the outcome", "verdict": "supported", "reason": ""}]
        rejected = facts["decision"]["action"] == "reject"
        if self.MARKERS and rejected and "has been approved" in report:
            statements.append(
                {"text": "approved", "verdict": "contradicted", "reason": ""}
            )
        for m in self.MARKERS:
            if m in report:
                statements.append(
                    {"text": m, "verdict": "unsupported", "reason": "planted"}
                )
        return schema.model_validate({"statements": statements})


async def test_the_run_scores_reports_and_the_judge():
    out = await fa.run(FakeWriter(), FakeJudge(), n=5)
    assert out["reports"] == 5 and out["failures"] == []
    assert out["grounding_pass_rate"] == 1.0
    assert out["judge_faithful_rate"] == 1.0
    assert out["judge_catch_rate"] == 1.0
    assert out["grounding_catch_rate"] == pytest.approx(0.6, abs=0.1)
    assert out["by_planted_error"]["invented_event"]["grounding"] == 0.0
    assert out["writer_model"] == "fake-writer" and out["judge_model"] == "fake-judge"


async def test_a_judge_that_sees_nothing_is_caught_by_calibration():
    class BlindJudge(FakeJudge):
        MARKERS = ()

    out = await fa.run(FakeWriter(), BlindJudge(), n=3)
    assert out["judge_faithful_rate"] == 1.0  # looks perfect on real reports...
    assert out["judge_catch_rate"] == 0.0  # ...but finds none of the planted errors


async def test_failed_reports_are_counted_one_by_one():
    class DownWriter(FakeWriter):
        async def structured(self, system, user, schema):
            from llm import LLMError

            raise LLMError("down")

    out = await fa.run(DownWriter(), FakeJudge(), n=6)
    # 6 cases, two with the same scenario: all 6 are counted as failures.
    assert out["cases"] == 6 and out["reports"] == 0 and out["judged"] == 0
    assert len(out["failures"]) == 6
