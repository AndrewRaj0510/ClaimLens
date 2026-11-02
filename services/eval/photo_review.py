"""Photo review: does the vision LLM notice when photos don't match the claim?

    python run.py photos [--guardrails-url URL]

The photo review (graph.review_photos: production's prompt, schema and vision
model) is the one step that compares the photos with the story. Only its
"no" changes anything (the claim goes to review with the reason); "unclear"
is shown to the adjuster but doesn't escalate. cases/photos.json pairs 11
freely licensed photos (Wikimedia Commons, downloaded here and pinned by
SHA-256) with descriptions:
- match: the description fits the photo; expect "yes"
- mismatch: the same photo with damage somewhere else on the car (the wrong
  photos for a claim); expect "no"
- no_damage: an undamaged car with a description of damage the photo would
  show; expect "no"

As in production, every photo goes through guardrails first (OCR, injection
scan, faces and text pixelated, EXIF removed) and the model only sees the
redacted copy; a photo the scan flags isn't shown. Guardrails runs in-process
like the red-team suite, or at --guardrails-url.

Scored: matches confirmed (yes on match), mismatches escalated (no on
mismatch and no_damage), mismatches passed (yes on them: the miss that
matters), false alarms (no on match), and for a photo with something an
adjuster should see (an old, rusted bumper), whether the concerns mention it.
"""

import base64
import hashlib
import json
import time
import urllib.error
import urllib.request

import paths

CASES = paths.CASES / "photos.json"
PHOTO_DIR = paths.DATA / "llm-eval" / "photos"
USER_AGENT = "ClaimLens-eval/1.0 (https://github.com/AndrewRaj0510/ClaimLens)"


def load(path=CASES):
    return json.loads(path.read_text(encoding="utf-8"))


def fetch(name, photo, folder=PHOTO_DIR, retries=4):
    """The photo's bytes: from disk if its hash matches, else downloaded."""
    path = folder / f"{name}.jpg"
    if path.exists():
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() == photo["sha256"]:
            return data
    folder.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(photo["url"], headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:  # Wikimedia rate-limits bursts
            if e.code != 429 or attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    digest = hashlib.sha256(data).hexdigest()
    if digest != photo["sha256"]:
        raise ValueError(f"{name}: SHA-256 {digest} is not the pinned one")
    path.write_bytes(data)
    return data


def redact(client, data):
    """Guardrails' inspection; the redacted JPEG, or None if it isn't safe to show."""
    r = client.post(
        "/images/inspect", files={"file": ("photo.jpg", data, "image/jpeg")}
    )
    r.raise_for_status()
    out = r.json()
    if out["text_scan"] and out["text_scan"]["flagged"]:
        return None
    return base64.b64decode(out["redacted_jpeg"])


def facts_for(case, shown):
    """What production's photo_review node gives the model."""
    return {
        "description": case["description"],
        "detected_damage": [
            {"damage_class": c, "severity": s} for c, s in case["detected"]
        ],
        "photos_shown": shown,
        "photos_submitted": 1,
    }


def score(case, got):
    answer = got["matches_description"] if got else None
    expect_yes = case["expect"] == "match"
    out = {
        "correct": answer == ("yes" if expect_yes else "no"),
        "false_alarm": expect_yes and answer == "no",
        "passed_mismatch": not expect_yes and answer == "yes",
    }
    words = case.get("concern_words")
    if words and got:
        text = " ".join(got["concerns"] + [got["summary"]]).lower()
        out["concern_raised"] = any(w in text for w in words)
    return out


async def run(vlm, guardrails_url=None, data=None):
    import graph
    import run_redteam
    import usage
    from faithfulness import rate
    from llm import LLMError

    data = load() if data is None else data
    started = time.time()
    redacted = {}
    with run_redteam.client_for(guardrails_url) as client:
        for name, photo in data["photos"].items():
            redacted[name] = redact(client, fetch(name, photo))
    rows, failures = [], []
    for case in data["cases"]:
        photo = redacted[case["photo"]]
        shown = [photo] if photo else []
        got = None
        with usage.metering(f"eval-photos-{case['id']}", run_budget=0) as meter:
            if shown:
                try:
                    got = await graph.review_photos(
                        vlm, facts_for(case, len(shown)), shown
                    )
                except LLMError as e:
                    failures.append({"id": case["id"], "error": str(e)})
                    continue
        rows.append(
            {
                "case": case,
                "got": got,
                "score": score(case, got),
                "tokens": (meter.input_tokens, meter.output_tokens),
            }
        )

    def where(*expect):
        return [r for r in rows if r["case"]["expect"] in expect]

    concern = [r for r in rows if "concern_raised" in r["score"]]
    return {
        "model": getattr(vlm, "model", None),
        "photos": len(data["photos"]),
        "photos_withheld": sorted(k for k, v in redacted.items() if v is None),
        "cases": len(data["cases"]),
        "answered": len(rows),
        "failures": failures,
        "accuracy": rate([r["score"]["correct"] for r in rows]),
        "match_confirmed": rate([r["score"]["correct"] for r in where("match")]),
        "mismatch_escalated": rate(
            [r["score"]["correct"] for r in where("mismatch", "no_damage")]
        ),
        "mismatch_passed": rate(
            [r["score"]["passed_mismatch"] for r in where("mismatch", "no_damage")]
        ),
        "false_alarm_rate": rate([r["score"]["false_alarm"] for r in where("match")]),
        "concern_raised": rate([r["score"]["concern_raised"] for r in concern]),
        "by_expect": {
            e: {
                "n": len(where(e)),
                "answers": {
                    a: sum(
                        (r["got"] or {}).get("matches_description") == a
                        for r in where(e)
                    )
                    for a in ("yes", "no", "unclear")
                },
            }
            for e in ("match", "mismatch", "no_damage")
        },
        "mean_tokens": {
            "input": rate([r["tokens"][0] for r in rows]),
            "output": rate([r["tokens"][1] for r in rows]),
        },
        "cases_detail": [
            {
                "id": r["case"]["id"],
                "expect": r["case"]["expect"],
                "answer": (r["got"] or {}).get("matches_description"),
                "ok": r["score"]["correct"],
                "summary": (r["got"] or {}).get("summary", "")[:300],
                "concerns": (r["got"] or {}).get("concerns", []),
            }
            for r in rows
        ],
        "seconds": round(time.time() - started, 1),
    }
