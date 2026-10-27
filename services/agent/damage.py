"""Turn per-photo detections into the claim's list of damages with a severity.

Severity comes from the damage's area as a % of the photo. That's a rough
proxy (a close-up makes a small dent look big), so the thresholds are
class-specific starting points to be calibrated in Phase 5, and the adjuster
sees the areas behind each severity.
"""

# class -> (minor below this %, moderate below this %, severe above)
SEVERITY_THRESHOLDS = {
    "dent": (1.0, 4.0),
    "scratch": (0.5, 2.0),  # scratches are thin: little area even when long
    "crack": (1.0, 4.0),
    # a chip vs a cracked window vs a shattered windshield
    "glass shatter": (3.0, 15.0),
    "lamp broken": (0.5, 2.0),
    "tire flat": (2.0, 6.0),
}
MIN_CONFIDENCE = 0.35


def severity(damage_class, area_pct):
    minor, moderate = SEVERITY_THRESHOLDS[damage_class]
    if area_pct < minor:
        return "minor"
    if area_pct < moderate:
        return "moderate"
    return "severe"


def aggregate(results):
    """results: one vision response per photo (or None if that photo failed).

    The same dent photographed twice is one damage, so detections are merged per
    class, keeping the largest area seen (the closest, clearest view)."""
    by_class = {}
    for photo_index, result in enumerate(results):
        for d in (result or {}).get("detections", []):
            if (
                d["confidence"] < MIN_CONFIDENCE
                or d["class_name"] not in SEVERITY_THRESHOLDS
            ):
                continue
            best = by_class.get(d["class_name"])
            if best is None or d["area_pct"] > best["area_pct"]:
                by_class[d["class_name"]] = {
                    "damage_class": d["class_name"],
                    "area_pct": d["area_pct"],
                    "confidence": d["confidence"],
                    "photo_index": photo_index,
                }
    damages = sorted(by_class.values(), key=lambda d: d["damage_class"])
    for d in damages:
        d["severity"] = severity(d["damage_class"], d["area_pct"])
    return damages
