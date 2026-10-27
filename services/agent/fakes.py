"""Test doubles for the graph's dependencies. The fake tools still enforce the
real per-node allowlist, so tests exercise it."""

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pydantic import ValidationError

from graph import Deps
from llm import LLMError
from tools import ToolFailed, check_allowed

POLICY = {
    "policy_id": "P00001",
    "customer_id": "C00001",
    "state": "ME",
    "status": "active",
    "start_date": "2024-01-01",
    "end_date": "2026-12-31",
    "vehicle": {
        "vehicle_id": "V00001",
        "make": "Honda",
        "model": "Civic",
        "year": 2020,
        "segment": "economy",
    },
}
CLAUSES = [
    {
        "wording_id": "w1",
        "section": "PART D > INSURING AGREEMENT",
        "page_start": 7,
        "page_end": 7,
        "text": "We will pay for direct and accidental loss to your covered auto.",
        "score": 0.8,
    },
    {
        "wording_id": "w1",
        "section": "PART D > EXCLUSIONS",
        "page_start": 8,
        "page_end": 8,
        "text": "We will not pay for wear and tear.",
        "score": 0.7,
    },
]


def photo_bytes(seed=0, blur=0):
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (800, 600), (120, 130, 140))
    draw = ImageDraw.Draw(img)
    for _ in range(30):
        x, y = rng.integers(0, 800), rng.integers(0, 600)
        w, h = rng.integers(20, 200, size=2)
        draw.rectangle(
            [x, y, x + w, y + h],
            fill=tuple(int(c) for c in rng.integers(0, 255, size=3)),
        )
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def vision_result(*detections):
    return {
        "status": "done",
        "result": {
            "width": 800,
            "height": 600,
            "damage_area_pct": 0,
            "detections": [
                {"class_name": c, "area_pct": a, "confidence": 0.9}
                for c, a in detections
            ],
        },
    }


class FakeLLM:
    def __init__(self, *, down=False, tool_calls=(), opinion=None, intake=None):
        self.down = down
        self.tool_calls = list(tool_calls)  # scripted chat() replies, in order
        self.opinion = opinion
        self.intake = intake
        self.prompts = []

    async def structured(self, system, user, schema):
        self.prompts.append((schema.__name__, system, user))
        if self.down:
            raise LLMError("down")
        data = {
            "Intake": self.intake
            or {
                "peril": "collision",
                "summary": "Rear-ended at a light.",
                "mentioned_date": None,
                "other_vehicle_involved": True,
                "injuries_mentioned": False,
            },
            "Opinion": self.opinion
            or {
                "assessment": "covered",
                "reasoning": "Part D covers it.",
                "citations": ["C1"],
                "concerns": [],
            },
            "Report": {
                "customer_message": "Approved.",
                "adjuster_summary": "Approved per policy.",
            },
        }[schema.__name__]
        try:
            return schema.model_validate(data)
        except ValidationError as e:  # like OllamaLLM: bad output is an LLM failure
            raise LLMError("schema") from e

    async def chat(self, messages, tools):
        self.prompts.append(("chat", messages, tools))
        if self.down:
            raise LLMError("down")
        if self.tool_calls:
            return {"content": "", "tool_calls": [self.tool_calls.pop(0)]}
        return {"content": "Enough clauses.", "tool_calls": []}


class FakeTools:
    def __init__(
        self, *, policy=POLICY, coverage=None, history=None, matches=(), fail=()
    ):
        self.policy = policy
        self.coverage = coverage or {
            "policy_id": "P00001",
            "in_force": True,
            "peril_covered": True,
            "covered": True,
            "deductible": 250,
            "reasons": [],
        }
        self.history = history or {
            "claims_last_12_months": 0,
            "total_paid": 0.0,
            "claims": [],
        }
        self.matches = list(matches)
        self.fail = set(fail)  # tool names that answer with a tool error
        self.calls = []

    async def call(self, node, server, tool, args):
        check_allowed(node, server, tool)
        self.calls.append((node, tool, args))
        if tool in self.fail:
            raise ToolFailed(f"{tool} failed")
        if tool == "get_policy":
            if self.policy is None:
                raise ToolFailed("No policy")
            return self.policy
        if tool == "check_coverage":
            return self.coverage
        if tool == "search_clauses":
            return {"wording_id": "w1", "results": CLAUSES}
        if tool == "estimate_repair_cost":
            total = 300.0 * len(args["damages"])
            return {
                "currency": "USD",
                "total": total,
                "low": total * 0.85,
                "high": total * 1.2,
                "line_items": [],
                "labour_rate_per_hour": 58,
                "assumptions": [],
            }
        if tool == "get_claim_history":
            return self.history
        if tool == "find_duplicate_photos":
            return {"max_distance": 8, "matches": self.matches}
        raise AssertionError(tool)

    async def schema(self, node, server, tool):
        check_allowed(node, server, tool)
        return {
            "type": "function",
            "function": {"name": tool, "description": "", "parameters": {}},
        }


class FakePhotos:
    def __init__(self, objects):
        self.objects = objects

    async def get(self, key):
        return self.objects[key]


class FakeVision:
    def __init__(self, results):
        self.results = results

    async def result(self, job_id, timeout_s):
        return self.results.get(job_id)


class FakeRecorder:
    def __init__(self):
        self.recorded = []

    async def record(self, state):
        self.recorded.append(state)


def deps(*, llm=None, tools=None, photos=None, vision=None):
    return Deps(
        llm=llm or FakeLLM(),
        tools=tools or FakeTools(),
        photos=FakePhotos(photos or {}),
        vision=FakeVision(vision or {}),
        recorder=FakeRecorder(),
    )
