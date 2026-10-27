"""Generate ClaimLens's synthetic insurance data (US market).

Everything except the damage photos and the policy wordings is synthetic:
customers, vehicles, policies, past claims, repair prices and fraud labels.
The same seed gives byte-identical output, so tests, evals and demos are
reproducible. Dates are relative to a fixed AS_OF date (not today) for the
same reason.

Output (one JSON array per table, except pricing):

    out/
      customers.json  vehicles.json  policies.json  claims.json
      fraud_labels.json   ground truth, kept out of claims.json so nothing that
                          serves claim history can leak it to the agent
      pricing.json        labour rates, segment multipliers, repair operations

Fraud is planted as known patterns (duplicate photos, a claim right after the
policy starts, an inflated amount, a late report, bursts of claims), and some
honest claims get a weak version of the same signals, so a fraud score has to
weigh evidence rather than match a single rule.

Stdlib only, like the CarDD converter.

Usage:
    python generate_synthetic.py --out ../data/synthetic [--seed 42] [--customers 2000]
"""

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

AS_OF = date(2026, 10, 1)
HISTORY_DAYS = 5 * 365

# Policy wordings ingested for RAG (Phase 2, pgvector). Each policy points at one.
WORDINGS = ("iso_pap_1998", "progressive_9611a", "geico_a30")

# Must match the vision model's class names (CarDD categories).
DAMAGE_CLASSES = (
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat",
)
SEVERITIES = ("minor", "moderate", "severe")

# All prices are invented, rough 2026 US ballparks: plausible, not quotes.
# Body-shop labour rate in USD per hour, by state.
LABOUR_RATES = {
    "CA": 88,
    "NY": 82,
    "WA": 78,
    "IL": 70,
    "FL": 64,
    "TX": 62,
    "GA": 60,
    "ME": 58,
    "OH": 56,
}
PAINT_MATERIALS_PER_HOUR = 45
SEGMENT_PARTS_MULTIPLIER = {
    "economy": 1.0,
    "midsize": 1.15,
    "suv": 1.3,
    "truck": 1.35,
    "luxury": 1.9,
}
SEGMENT_PREMIUM = {
    "economy": 1500,
    "midsize": 1700,
    "suv": 1900,
    "truck": 2000,
    "luxury": 2800,
}

# (damage class, severity) -> (operation, parts USD for an economy car, labour hours, paint hours)
REPAIR_OPERATIONS = {
    ("dent", "minor"): ("Paintless dent repair", 0, 1.5, 0.0),
    ("dent", "moderate"): ("Repair and refinish panel", 30, 4.0, 3.0),
    ("dent", "severe"): ("Replace and refinish panel", 450, 6.0, 4.0),
    ("scratch", "minor"): ("Buff and touch up", 20, 1.0, 0.5),
    ("scratch", "moderate"): ("Refinish panel", 40, 1.5, 3.0),
    ("scratch", "severe"): ("Refinish several panels", 80, 3.0, 6.0),
    ("crack", "minor"): ("Plastic repair and refinish", 25, 2.0, 2.5),
    ("crack", "moderate"): ("Replace bumper cover", 380, 2.5, 3.0),
    ("crack", "severe"): ("Replace bumper cover and reinforcement", 750, 4.5, 3.5),
    ("glass shatter", "minor"): ("Replace side window", 250, 1.5, 0.0),
    ("glass shatter", "moderate"): ("Replace windshield", 450, 2.0, 0.0),
    ("glass shatter", "severe"): (
        "Replace windshield and recalibrate ADAS camera",
        1100,
        3.0,
        0.0,
    ),
    ("lamp broken", "minor"): ("Replace lens or bulb", 120, 0.5, 0.0),
    ("lamp broken", "moderate"): ("Replace headlamp assembly", 350, 1.0, 0.0),
    ("lamp broken", "severe"): ("Replace LED adaptive headlamp", 1200, 1.5, 0.0),
    ("tire flat", "minor"): ("Repair puncture", 15, 0.5, 0.0),
    ("tire flat", "moderate"): ("Replace tire", 180, 0.7, 0.0),
    ("tire flat", "severe"): ("Replace tire and wheel", 550, 1.0, 0.0),
}

VEHICLES = (
    ("Toyota", "Corolla", "economy"),
    ("Honda", "Civic", "economy"),
    ("Hyundai", "Elantra", "economy"),
    ("Nissan", "Sentra", "economy"),
    ("Toyota", "Camry", "midsize"),
    ("Honda", "Accord", "midsize"),
    ("Nissan", "Altima", "midsize"),
    ("Toyota", "RAV4", "suv"),
    ("Honda", "CR-V", "suv"),
    ("Ford", "Explorer", "suv"),
    ("Chevrolet", "Equinox", "suv"),
    ("Ford", "F-150", "truck"),
    ("Chevrolet", "Silverado 1500", "truck"),
    ("Ram", "1500", "truck"),
    ("BMW", "3 Series", "luxury"),
    ("Mercedes-Benz", "C-Class", "luxury"),
    ("Lexus", "RX", "luxury"),
    ("Tesla", "Model 3", "luxury"),
)

FIRST_NAMES = [
    "James",
    "Mary",
    "Robert",
    "Patricia",
    "John",
    "Jennifer",
    "Michael",
    "Linda",
    "David",
    "Elizabeth",
    "William",
    "Barbara",
    "Richard",
    "Susan",
    "Joseph",
    "Jessica",
    "Thomas",
    "Sarah",
    "Carlos",
    "Maria",
    "Daniel",
    "Karen",
    "Wei",
    "Priya",
    "Ahmed",
    "Fatima",
    "Kenji",
    "Aisha",
    "Luis",
    "Sofia",
]
LAST_NAMES = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Rodriguez",
    "Martinez",
    "Hernandez",
    "Lopez",
    "Wilson",
    "Anderson",
    "Thomas",
    "Taylor",
    "Moore",
    "Jackson",
    "Martin",
    "Lee",
    "Nguyen",
    "Patel",
    "Kim",
    "Chen",
    "Singh",
    "Okafor",
    "Cohen",
    "Murphy",
    "Rossi",
    "Walker",
]

ROADS = ("I-95", "Route 1", "Main Street", "I-10", "Highway 101", "I-35")

# (peril, description template, damage classes the scenario can produce)
SCENARIOS = (
    (
        "collision",
        "Rear-ended while stopped at a light on {road}.",
        ("dent", "scratch", "crack", "lamp broken"),
    ),
    ("collision", "Backed into a pole in a parking lot.", ("dent", "scratch", "crack")),
    (
        "collision",
        "Side-swiped by a car changing lanes on {road}.",
        ("scratch", "dent"),
    ),
    ("collision", "Hit a curb hard turning into a driveway.", ("tire flat", "scratch")),
    (
        "collision",
        "Front-end collision in slow traffic on {road}.",
        ("crack", "lamp broken", "dent"),
    ),
    (
        "comprehensive",
        "A rock thrown up by a truck broke the windshield on {road}.",
        ("glass shatter",),
    ),
    (
        "comprehensive",
        "Car was keyed and a window smashed overnight.",
        ("scratch", "glass shatter"),
    ),
    (
        "comprehensive",
        "Hail storm while the car was parked outside.",
        ("dent", "glass shatter"),
    ),
    (
        "comprehensive",
        "Hit a deer on {road} at dusk.",
        ("dent", "lamp broken", "crack"),
    ),
)

FRAUD_RATE = 0.07
# pattern -> relative weight. Applied in this order: patterns that move the
# incident date come before the ones that depend on it (bursts, duplicates).
FRAUD_PATTERNS = {
    "early_claim": 0.2,
    "late_report": 0.15,
    "claim_burst": 0.15,
    "inflated_amount": 0.25,
    "duplicate_photo": 0.25,
}
# Thresholds that define each planted pattern. Tests check against these.
EARLY_CLAIM_MAX_DAYS = 20
LATE_REPORT_MIN_DAYS = 45
INFLATION_MIN_RATIO = 2.2
DUPLICATE_MAX_BIT_FLIPS = 4
BURST_WINDOW_DAYS = 365


def repair_cost(damage_class, severity, segment, state):
    """Expected cost in USD of one repair, from the pricing tables."""
    _, parts, labour_h, paint_h = REPAIR_OPERATIONS[(damage_class, severity)]
    rate = LABOUR_RATES[state]
    return (
        parts * SEGMENT_PARTS_MULTIPLIER[segment]
        + (labour_h + paint_h) * rate
        + paint_h * PAINT_MATERIALS_PER_HOUR
    )


def expected_cost(damages, segment, state):
    return sum(repair_cost(d["class"], d["severity"], segment, state) for d in damages)


def pricing_table():
    return {
        "currency": "USD",
        "labour_rates_per_hour": LABOUR_RATES,
        "paint_materials_per_hour": PAINT_MATERIALS_PER_HOUR,
        "segment_parts_multiplier": SEGMENT_PARTS_MULTIPLIER,
        "operations": [
            {
                "damage_class": cls,
                "severity": sev,
                "operation": op,
                "parts_usd": parts,
                "labour_hours": labour,
                "paint_hours": paint,
            }
            for (cls, sev), (op, parts, labour, paint) in REPAIR_OPERATIONS.items()
        ],
    }


def random_date(rng, start, end):
    return start + timedelta(days=rng.randint(0, (end - start).days))


def random_phash(rng):
    """A 64-bit perceptual hash as 16 hex chars. Synthetic history has no real
    photos, so hashes are random; a re-used photo is a near-copy (see below)."""
    return f"{rng.getrandbits(64):016x}"


def near_copy(rng, phash):
    """The same photo re-saved or lightly cropped: a few hash bits differ."""
    value = int(phash, 16)
    for bit in rng.sample(range(64), rng.randint(0, DUPLICATE_MAX_BIT_FLIPS)):
        value ^= 1 << bit
    return f"{value:016x}"


def make_vin(rng):
    # VIN alphabet (no I, O, Q). Not check-digit valid, so it can't match a real car.
    alphabet = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
    return "".join(rng.choice(alphabet) for _ in range(17))


def make_policy(rng, policy_id, customer, vehicle):
    start = random_date(
        rng,
        date.fromisoformat(customer["customer_since"]),
        AS_OF - timedelta(days=30),
    )
    # 365-day terms renewed until AS_OF (leap days ignored: it's synthetic).
    terms = (AS_OF - start).days // 365 + 1
    end = start + timedelta(days=365 * terms)
    status, cancelled = "active", None
    r = rng.random()
    if r < 0.08 and terms > 1:
        status = "lapsed"
        end = start + timedelta(days=365 * rng.randint(1, terms - 1))
    elif r < 0.12:
        status = "cancelled"
        cancelled = random_date(rng, start + timedelta(days=30), AS_OF)
        end = cancelled

    r = rng.random()
    collision = r < 0.75
    comprehensive = (
        r < 0.80
    )  # 75% full cover, 5% comprehensive only, 20% liability only
    segment = vehicle["segment"]
    premium = SEGMENT_PREMIUM[segment] * (1.0 if collision else 0.45)
    return {
        "policy_id": policy_id,
        "customer_id": customer["customer_id"],
        "vehicle_id": vehicle["vehicle_id"],
        "wording_id": rng.choice(WORDINGS),
        "state": customer["state"],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "status": status,
        "cancelled_date": cancelled.isoformat() if cancelled else None,
        "coverages": {
            "liability": True,
            "collision": collision,
            "comprehensive": comprehensive,
            "full_glass": comprehensive and rng.random() < 0.3,
            "rental_reimbursement": rng.random() < 0.4,
        },
        "collision_deductible": rng.choice((250, 500, 500, 1000))
        if collision
        else None,
        "comprehensive_deductible": rng.choice((100, 250, 500))
        if comprehensive
        else None,
        "annual_premium": round(premium * rng.uniform(0.8, 1.25)),
    }


def deductible_for(policy, peril, damages):
    if peril == "collision":
        return policy["collision_deductible"]
    if policy["coverages"]["full_glass"] and all(
        d["class"] == "glass shatter" for d in damages
    ):
        return 0
    return policy["comprehensive_deductible"]


def make_claim(rng, policy, vehicle, incident, report_delay_days):
    perils = [p for p in ("collision", "comprehensive") if policy["coverages"][p]]
    peril, template, pool = rng.choice([s for s in SCENARIOS if s[0] in perils])
    classes = rng.sample(pool, min(len(pool), rng.choice((1, 1, 2))))
    damages = [
        {
            "class": c,
            "severity": rng.choices(SEVERITIES, weights=(0.5, 0.35, 0.15))[0],
        }
        for c in classes
    ]
    cost = expected_cost(damages, vehicle["segment"], policy["state"])
    return {
        "policy_id": policy["policy_id"],
        "customer_id": policy["customer_id"],
        "vehicle_id": policy["vehicle_id"],
        "peril": peril,
        "description": template.format(road=rng.choice(ROADS)),
        "damages": damages,
        "incident_date": incident,
        "reported_date": min(incident + timedelta(days=report_delay_days), AS_OF),
        "claimed_amount": round(cost * rng.uniform(0.9, 1.15), 2),
        "photo_phashes": [random_phash(rng) for _ in range(rng.randint(1, 4))],
        "_patterns": [],
    }


def claim_window(policy):
    start = date.fromisoformat(policy["start_date"])
    end = min(date.fromisoformat(policy["end_date"]), AS_OF)
    return start, end


def plant_fraud(rng, claim, pattern, policy, vehicle):
    """Turn an honest claim into one showing `pattern`. Returns extra claims (bursts).
    Returns None if the claim can't carry the pattern (e.g. no room in the window)."""
    start, end = claim_window(policy)
    if pattern == "early_claim":
        incident = start + timedelta(days=rng.randint(1, EARLY_CLAIM_MAX_DAYS))
        if incident > end:
            return None
        delay = claim["reported_date"] - claim["incident_date"]
        claim["incident_date"] = incident
        claim["reported_date"] = min(incident + delay, AS_OF)
    elif pattern == "late_report":
        reported = claim["incident_date"] + timedelta(
            days=rng.randint(LATE_REPORT_MIN_DAYS, 120)
        )
        if reported > AS_OF:
            return None
        claim["reported_date"] = reported
    elif pattern == "inflated_amount":
        cost = expected_cost(claim["damages"], vehicle["segment"], policy["state"])
        claim["claimed_amount"] = round(cost * rng.uniform(INFLATION_MIN_RATIO, 4.0), 2)
    elif pattern == "claim_burst":
        # Two more claims on the same policy within the burst window.
        lo = claim["incident_date"]
        hi = min(lo + timedelta(days=BURST_WINDOW_DAYS - 1), end)
        if (hi - lo).days < 30:
            return None
        extra = []
        for _ in range(2):
            c = make_claim(
                rng, policy, vehicle, random_date(rng, lo, hi), rng.randint(0, 5)
            )
            c["_patterns"].append("claim_burst")
            extra.append(c)
        claim["_patterns"].append(pattern)
        return extra
    # duplicate_photo is applied after sorting, once "earlier claims" are known.
    claim["_patterns"].append(pattern)
    return []


def add_weak_signals(rng, claim, policy):
    """Honest claims that look a bit suspicious: realistic false-positive bait."""
    start, end = claim_window(policy)
    r = rng.random()
    if r < 0.04 and (end - start).days > 45:
        claim["incident_date"] = start + timedelta(days=rng.randint(21, 45))
        claim["reported_date"] = min(claim["incident_date"] + timedelta(days=2), AS_OF)
    elif r < 0.08:
        claim["reported_date"] = min(
            claim["incident_date"] + timedelta(days=rng.randint(10, 30)), AS_OF
        )
    elif r < 0.11:
        claim["claimed_amount"] = round(
            claim["claimed_amount"] * rng.uniform(1.2, 1.5), 2
        )


def plant_duplicate_photos(rng, claims):
    """Re-use a photo from an earlier claim by a different customer."""
    for i, claim in enumerate(claims):
        if "duplicate_photo" not in claim["_patterns"]:
            continue
        earlier = [
            c
            for c in claims[:i]
            if c["reported_date"] < claim["incident_date"]
            and c["customer_id"] != claim["customer_id"]
        ]
        if not earlier:
            claim["_patterns"].remove("duplicate_photo")
            continue
        source = rng.choice(earlier)
        claim["photo_phashes"][0] = near_copy(rng, rng.choice(source["photo_phashes"]))


def generate(seed=42, n_customers=2000):
    rng = random.Random(seed)
    customers, vehicles, policies = [], [], []
    for i in range(n_customers):
        first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        customer = {
            "customer_id": f"C{i + 1:05d}",
            "first_name": first,
            "last_name": last,
            "date_of_birth": (
                AS_OF - timedelta(days=rng.randint(18 * 365, 80 * 365))
            ).isoformat(),
            "state": rng.choice(sorted(LABOUR_RATES)),
            # example.com and 555-01xx are reserved for fiction, so these can't reach anyone.
            "email": f"{first.lower()}.{last.lower()}{i + 1}@example.com",
            "phone": f"(207) 555-01{rng.randint(0, 99):02d}",
            "customer_since": (
                AS_OF - timedelta(days=rng.randint(60, HISTORY_DAYS))
            ).isoformat(),
        }
        customers.append(customer)
        for _ in range(2 if rng.random() < 0.15 else 1):
            make, model, segment = rng.choice(VEHICLES)
            vehicle = {
                "vehicle_id": f"V{len(vehicles) + 1:05d}",
                "customer_id": customer["customer_id"],
                "vin": make_vin(rng),
                "make": make,
                "model": model,
                "year": rng.randint(2012, 2026),
                "segment": segment,
            }
            vehicles.append(vehicle)
            policies.append(
                make_policy(rng, f"P{len(policies) + 1:05d}", customer, vehicle)
            )

    vehicle_by_id = {v["vehicle_id"]: v for v in vehicles}
    claims = []
    for policy in policies:
        if not (
            policy["coverages"]["collision"] or policy["coverages"]["comprehensive"]
        ):
            continue  # liability-only drivers don't claim for their own car
        start, end = claim_window(policy)
        n = rng.choices((0, 1, 2, 3), weights=(0.62, 0.27, 0.09, 0.02))[0]
        for _ in range(n):
            claims.append(
                make_claim(
                    rng,
                    policy,
                    vehicle_by_id[policy["vehicle_id"]],
                    random_date(rng, start, end),
                    rng.choices(range(8), weights=(3, 4, 3, 2, 1, 1, 1, 1))[0],
                )
            )

    policy_by_id = {p["policy_id"]: p for p in policies}
    fraud_idx = set(rng.sample(range(len(claims)), round(FRAUD_RATE * len(claims))))
    extra = []
    for i, claim in enumerate(claims):
        policy = policy_by_id[claim["policy_id"]]
        vehicle = vehicle_by_id[claim["vehicle_id"]]
        if i not in fraud_idx:
            add_weak_signals(rng, claim, policy)
            continue
        picked = set(
            rng.choices(
                list(FRAUD_PATTERNS),
                weights=list(FRAUD_PATTERNS.values()),
                k=2 if rng.random() < 0.2 else 1,
            )
        )
        for pattern in [p for p in FRAUD_PATTERNS if p in picked]:
            added = plant_fraud(rng, claim, pattern, policy, vehicle)
            if added:
                extra += added
    claims += extra

    claims.sort(key=lambda c: (c["incident_date"], c["policy_id"]))
    plant_duplicate_photos(rng, claims)

    fraud_labels = []
    for n, claim in enumerate(claims, start=1):
        claim_id = f"CL{n:06d}"
        patterns = sorted(set(claim.pop("_patterns")))
        policy = policy_by_id[claim["policy_id"]]
        deductible = deductible_for(policy, claim["peril"], claim["damages"])
        # Historical outcome: investigators caught most planted fraud; some was paid.
        if patterns and rng.random() < 0.6:
            status, paid = "denied", 0.0
        elif rng.random() < 0.04:
            status, paid = "withdrawn", 0.0
        elif claim["claimed_amount"] <= deductible:
            status, paid = "below_deductible", 0.0
        else:
            status, paid = "paid", round(claim["claimed_amount"] - deductible, 2)
        claim.update(
            claim_id=claim_id,
            incident_date=claim["incident_date"].isoformat(),
            reported_date=claim["reported_date"].isoformat(),
            deductible=deductible,
            status=status,
            paid_amount=paid,
        )
        fraud_labels.append(
            {"claim_id": claim_id, "is_fraud": bool(patterns), "patterns": patterns}
        )

    field_order = [
        "claim_id",
        "policy_id",
        "customer_id",
        "vehicle_id",
        "peril",
        "description",
        "damages",
        "incident_date",
        "reported_date",
        "claimed_amount",
        "deductible",
        "status",
        "paid_amount",
        "photo_phashes",
    ]
    claims = [{k: c[k] for k in field_order} for c in claims]
    return {
        "customers": customers,
        "vehicles": vehicles,
        "policies": policies,
        "claims": claims,
        "fraud_labels": fraud_labels,
        "pricing": pricing_table(),
    }


def write(data, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in data.items():
        (out / f"{name}.json").write_text(json.dumps(rows, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="Output folder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--customers", type=int, default=2000)
    args = parser.parse_args()

    data = generate(args.seed, args.customers)
    write(data, args.out)
    n_fraud = sum(label["is_fraud"] for label in data["fraud_labels"])
    for name in ("customers", "vehicles", "policies", "claims"):
        print(f"{name:>10}: {len(data[name])}")
    print(f"     fraud: {n_fraud} ({n_fraud / len(data['claims']):.1%} of claims)")


if __name__ == "__main__":
    main()
