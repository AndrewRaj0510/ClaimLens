"""Test doubles for the graph's dependencies. The fake tools still enforce the
real per-node allowlist, so tests exercise it."""

import base64
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pydantic import ValidationError

from adapters import GuardrailsRejected, GuardrailsUnavailable
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
        self,
        *,
        policy=POLICY,
        coverage=None,
        history=None,
        matches=(),
        fail=(),
        searched=None,
    ):
        self.policy = policy
        self.searched = searched or {}  # query -> search results, else CLAUSES
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
            return {
                "wording_id": "w1",
                "results": self.searched.get(args["query"], CLAUSES),
            }
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


class FakeIndex:
    """In-memory ClaimIndex with the same ordering and filters."""

    def __init__(self):
        self.rows = {}
        self.clock = 0

    def _tick(self):
        self.clock += 1
        return f"2026-10-27T00:00:{self.clock:02d}+00:00"

    async def setup(self):
        pass

    async def add(self, claim_id, customer_id, policy_id):
        now = self._tick()
        self.rows.setdefault(
            claim_id,
            {
                "claim_id": claim_id,
                "customer_id": customer_id,
                "policy_id": policy_id,
                "status": "processing",
                "summary": {},
                "created_at": now,
                "updated_at": now,
            },
        )

    async def update(self, claim_id, status, summary=None):
        row = self.rows[claim_id]
        row["status"], row["updated_at"] = status, self._tick()
        if summary is not None:
            row["summary"] = summary

    async def list(self, customer_id=None, status=None, limit=50):
        rows = [
            dict(r)
            for r in self.rows.values()
            if customer_id in (None, r["customer_id"]) and status in (None, r["status"])
        ]
        return sorted(rows, key=lambda r: r["updated_at"], reverse=True)[:limit]


class FakeGuardrails:
    """Flags any text containing one of `trigger` words; per-photo results can
    be scripted by the photo's bytes. down=True: the service is unreachable."""

    def __init__(self, *, trigger=("ignore previous",), down=False, photos=None):
        self.trigger = trigger
        self.down = down
        self.photos = photos or {}  # bytes -> partial inspect result
        self.texts = []  # every text sent for scanning
        self.profiles = []
        self.inspected = 0

    def _scan(self, text):
        hit = any(t in text.lower() for t in self.trigger)
        return {"flagged": hit, "score": 0.99 if hit else 0.01, "rules": []}

    async def scan_texts(self, texts, profile="untrusted"):
        if self.down:
            raise GuardrailsUnavailable("/scan/text: ConnectError")
        self.texts += list(texts)
        self.profiles.append(profile)
        return [self._scan(t) for t in texts]

    async def inspect_photo(self, data, incident_date=None):
        if self.down:
            raise GuardrailsUnavailable("/images/inspect: ConnectError")
        if data == b"not an image":
            raise GuardrailsRejected("/images/inspect: HTTP 400")
        self.inspected += 1
        scripted = self.photos.get(data, {})
        ocr = scripted.get("ocr", [])
        text = "\n".join(r["text"] for r in ocr)
        return {
            "width": 800,
            "height": 600,
            "work_size": [800, 600],
            "ocr": ocr,
            "text_scan": self._scan(text) if text else None,
            "text_scan_mode": "rules+classifier",
            "redactions": {"faces": 0, "text_regions": len(ocr), "plates": 0},
            "metadata": {
                "camera": None,
                "software": None,
                "taken_at": None,
                "has_exif": False,
                "has_gps": False,
                "findings": scripted.get("findings", []),
            },
            "ai_generated": scripted.get("ai_generated"),
            "redacted_jpeg": base64.b64encode(b"redacted:" + data[:8]).decode(),
        }


class FakeVLM:
    def __init__(self, *, down=False, review=None):
        self.down = down
        self.review = review
        self.calls = []  # (user prompt, images)

    async def structured(self, system, user, schema, images=None):
        self.calls.append((user, list(images or [])))
        if self.down:
            raise LLMError("down")
        return schema.model_validate(
            self.review
            or {
                "matches_description": "yes",
                "damage_seen": ["dent on the rear bumper"],
                "concerns": [],
                "summary": "A dent on the rear bumper, as described.",
            }
        )


def deps(*, llm=None, tools=None, photos=None, vision=None, guardrails=None, vlm=None):
    return Deps(
        llm=llm or FakeLLM(),
        tools=tools or FakeTools(),
        photos=FakePhotos(photos or {}),
        vision=FakeVision(vision or {}),
        recorder=FakeRecorder(),
        guardrails=guardrails or FakeGuardrails(),
        vlm=vlm or FakeVLM(),
    )
