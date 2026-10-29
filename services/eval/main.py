"""Eval service: offline metrics, the report judge, and the gates, over HTTP.

    GET  /health
    POST /grounding        a report checked against its facts by code (no LLM)
    POST /judge            the same, plus the LLM judge's statement-by-statement verdict
    GET  /results          the stored suite results (when, which commit, gate status)
    GET  /results/{suite}  one suite's full result
    GET  /gates            every gate against the stored results

The suites themselves are long batch jobs, so they run from the CLI (run.py),
in CI and in Colab, not in a request. This service serves their results and
makes the judge callable, e.g. to spot-check reports from production claims.
Internal only, like the agent.
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Path, Request
from pydantic import BaseModel, Field

import faithfulness as fa
import gate
import paths

log = logging.getLogger("eval")
SUITES = ("redteam", "faithfulness", "detection")


class ReportIn(BaseModel):
    customer_message: Annotated[str, Field(max_length=1500)]
    adjuster_summary: Annotated[str, Field(max_length=3000)]


class JudgeRequest(BaseModel):
    facts: dict
    report: ReportIn


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "judge"):  # tests set their own
        from llm import OllamaLLM

        app.state.judge = OllamaLLM(model=fa.JUDGE_MODEL, think=fa.JUDGE_THINK)
    yield


def create_app(lifespan=lifespan, results_dir=None, baseline=None):
    app = FastAPI(title="ClaimLens Eval", version="0.1.0", lifespan=lifespan)
    results = results_dir or paths.RESULTS
    baseline = baseline or gate.BASELINE

    def facts_size_ok(facts):
        if len(json.dumps(facts, default=str)) > 50_000:
            raise HTTPException(status_code=413, detail="Facts too large")

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "eval", "judge_model": fa.JUDGE_MODEL}

    @app.post("/grounding")
    def grounding(body: JudgeRequest):
        facts_size_ok(body.facts)
        return fa.grounding(body.report.model_dump(), body.facts)

    @app.post("/judge")
    async def judge(request: Request, body: JudgeRequest):
        from llm import LLMError

        facts_size_ok(body.facts)
        report = body.report.model_dump()
        out = {"grounding": fa.grounding(report, body.facts)}
        try:
            out["judge"] = await fa.judge(request.app.state.judge, body.facts, report)
        except LLMError as e:
            raise HTTPException(
                status_code=503, detail=f"Judge unavailable: {e}"
            ) from e
        return out

    def stored():
        return gate.load_results(results)

    @app.get("/results")
    def list_results():
        out = []
        for suite in SUITES:
            p = results / f"{suite}.json"
            if p.exists():
                record = json.loads(p.read_text())
                out.append({k: record[k] for k in ("suite", "ran_at", "commit")})
        passed, rows = gate.check(json.loads(baseline.read_text()), stored())
        for r in out:
            r["gates"] = _status([x for x in rows if x["suite"] == r["suite"]])
        return {"results": out, "gates_passed": passed}

    @app.get("/results/{suite}")
    def get_result(suite: Annotated[Literal[SUITES], Path()]):
        p = results / f"{suite}.json"
        if not p.exists():
            raise HTTPException(status_code=404, detail="No result for this suite yet")
        return json.loads(p.read_text())

    @app.get("/gates")
    def gates():
        passed, rows = gate.check(json.loads(baseline.read_text()), stored())
        return {"passed": passed, "gates": rows}

    return app


def _status(rows):
    if any(r["status"] == "fail" for r in rows):
        return "fail"
    return "pass" if rows else "none"


app = create_app()
