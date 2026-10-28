"""Fraud signals and a score, computed in code from tool results.

Each signal has a weight: roughly "how likely is fraud if only this signal
fired". They're combined with noisy-OR, score = 1 - prod(1 - w), so several
weak signals add up but no number of them exceeds 1, and one strong signal
(another customer's photo) is enough on its own to send a claim for review.

The weights are hand-set starting points. Phase 5 measures them against the
synthetic fraud labels (eval.fraud_labels) and recalibrates. The score never
denies a claim; it decides whether a human investigator looks first.
"""

from datetime import date

WEIGHTS = {
    "photo_reused_other_customer": 0.6,
    "photo_reused_same_customer": 0.3,
    "early_claim": 0.25,
    "late_report": 0.2,
    "frequent_claims": 0.25,
    "amount_inflated": 0.3,
    "date_mismatch": 0.15,
    # From services/guardrails (Phase 4). An injection attempt is someone
    # trying to talk the system into a decision: on its own it sends the claim
    # to review (medium), it doesn't prove the damage is fake.
    "injection_attempt": 0.4,
    "ai_generated_photo": 0.5,
    "photo_predates_incident": 0.35,
    "photo_edited": 0.15,
    "duplicate_in_claim": 0.1,
}
EARLY_CLAIM_DAYS = 30
LATE_REPORT_DAYS = 30
FREQUENT_CLAIMS_12M = 2
INFLATION_RATIO = 2.0
DATE_MISMATCH_DAYS = 3
LEVELS = ((0.6, "high"), (0.3, "medium"), (0.0, "low"))


def signals(
    *,
    incident_date: date,
    reported_date: date,
    policy_start: date,
    claims_last_12_months: int,
    duplicate_matches: list[dict],
    claimed_amount: float | None,
    estimate_high: float | None,
    mentioned_date: date | None,
    photo_checks: list[dict] = (),
    injection_sources: list[str] = (),
    duplicate_in_claim: bool = False,
):
    found = []
    mirrored = (
        " (a mirrored copy)"
        if any(m.get("mirrored") for m in duplicate_matches)
        else ""
    )
    if any(not m["same_customer"] for m in duplicate_matches):
        found.append(
            (
                "photo_reused_other_customer",
                f"A photo matches one from another customer's claim{mirrored}.",
            )
        )
    elif duplicate_matches:
        found.append(
            (
                "photo_reused_same_customer",
                f"A photo matches one from this customer's earlier claim{mirrored}.",
            )
        )
    days_in = (incident_date - policy_start).days
    if 0 <= days_in <= EARLY_CLAIM_DAYS:
        found.append(
            ("early_claim", f"Incident {days_in} days after the policy started.")
        )
    delay = (reported_date - incident_date).days
    if delay > LATE_REPORT_DAYS:
        found.append(("late_report", f"Reported {delay} days after the incident."))
    if claims_last_12_months >= FREQUENT_CLAIMS_12M:
        found.append(
            (
                "frequent_claims",
                f"{claims_last_12_months} claims in the last 12 months.",
            )
        )
    if (
        claimed_amount
        and estimate_high
        and claimed_amount > INFLATION_RATIO * estimate_high
    ):
        found.append(
            (
                "amount_inflated",
                f"Claimed ${claimed_amount:,.0f}, more than {INFLATION_RATIO:g}x the top of the estimate (${estimate_high:,.0f}).",
            )
        )
    if (
        mentioned_date
        and abs((mentioned_date - incident_date).days) > DATE_MISMATCH_DAYS
    ):
        found.append(
            (
                "date_mismatch",
                f"The description mentions {mentioned_date}, the form says {incident_date}.",
            )
        )
    if injection_sources:
        found.append(
            (
                "injection_attempt",
                "Text that tries to instruct the AI was found in: "
                + ", ".join(injection_sources)
                + ".",
            )
        )
    found += photo_signals(photo_checks)
    if duplicate_in_claim:
        found.append(
            ("duplicate_in_claim", "The same photo was submitted more than once.")
        )
    return [{"signal": s, "weight": WEIGHTS[s], "detail": d} for s, d in found]


def photo_signals(photo_checks):
    """One signal per kind, naming the photos it was seen in."""
    seen = {}
    for check in photo_checks:
        i = check["photo_index"] + 1  # named for people: 1-based
        codes = {f["code"] for f in check.get("metadata", {}).get("findings", [])}
        ai = check.get("ai_generated") or {}
        if ai.get("flagged") or "ai_metadata" in codes:
            seen.setdefault("ai_generated_photo", []).append(i)
        if "taken_before_incident" in codes:
            seen.setdefault("photo_predates_incident", []).append(i)
        if "edited" in codes:
            seen.setdefault("photo_edited", []).append(i)
    text = {
        "ai_generated_photo": "Photo {} looks AI-generated (detector or metadata).",
        "photo_predates_incident": "Photo {} was taken before the incident date.",
        "photo_edited": "Photo {} was saved by editing software.",
    }
    return [
        (signal, text[signal].format(", ".join(str(i) for i in photos)))
        for signal, photos in seen.items()
    ]


def score(found):
    remaining = 1.0
    for s in found:
        remaining *= 1 - s["weight"]
    value = round(1 - remaining, 3)
    level = next(name for cutoff, name in LEVELS if value >= cutoff)
    return {"score": value, "level": level, "signals": found}
