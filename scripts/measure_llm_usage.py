"""Measure what real claims cost in LLM tokens, end to end through the public API.

Files N claims as different customers, approves each as the adjuster, and
reads the agent's per-claim usage (GET /claims/{id} -> llm_usage, from
services/agent/usage.py) at the approval point and after the report.

    python scripts/measure_llm_usage.py --claims 8 --photos data/e2e/car.jpg ...

Needs a running stack with a real LLM key (kind: http://localhost:8080).
Logins come from the file kind's seed.sh writes; passwords are never printed.
Writes one JSON line per claim to --out. Each customer may file 10 claims an
hour (the gateway's limit), so claims rotate over customers.
"""

import argparse
import json
import random
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import httpx

DESCRIPTIONS = [
    (
        "I was stopped at a red light on Main Street when the car behind me failed "
        "to brake and hit my rear bumper. The bumper is dented and the paint is "
        "scratched. Nobody was hurt and we exchanged insurance details."
    ),
    (
        "While reversing out of a parking space at the supermarket I hit a concrete "
        "pillar I hadn't seen. The rear quarter panel has a dent and a long scratch."
    ),
    (
        "Someone keyed the driver's side of my car overnight while it was parked "
        "on the street outside my house. There are deep scratches along both doors."
    ),
    (
        "A tree branch fell on my car during last night's storm. The windshield is "
        "cracked and there is a dent on the hood."
    ),
    (
        "Another driver merged into my lane on the highway and sideswiped me. My "
        "front door and wing mirror are damaged and the headlight is broken."
    ),
    (
        "I hit a deep pothole at speed on the way to work. The front tyre went flat "
        "immediately and the front bumper scraped the road."
    ),
    (
        "My car was hit in a car park by a vehicle that drove off. The rear lamp is "
        "smashed and the bumper below it is cracked."
    ),
    (
        "Hail damaged the roof and bonnet of my car while it was parked outside "
        "the office. There are many small dents across the top panels."
    ),
]


def logins(path):
    adjuster, customers = None, []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "adjuster" and adjuster is None:
            adjuster = (parts[1], parts[2])
        elif len(parts) == 4 and parts[0] == "customer":
            customers.append((parts[1], parts[2]))
    if not adjuster or not customers:
        sys.exit(f"no adjuster or customer logins in {path}")
    return adjuster, customers


def session(base, username, password):
    c = httpx.Client(base_url=base, timeout=60)
    r = c.post("/auth/login", json={"username": username, "password": password})
    r.raise_for_status()
    return c


def wait(adj, claim_id, statuses, timeout_s=300):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        body = adj.get(f"/claims/{claim_id}").json()
        if body["status"] in statuses:
            return body
        time.sleep(1)
    raise TimeoutError(f"{claim_id} stuck in {body['status']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8080/api")
    ap.add_argument("--logins", default="data/kind-logins.txt")
    ap.add_argument("--claims", type=int, default=8)
    ap.add_argument("--photos", nargs="+", required=True)
    ap.add_argument("--out", default="llm-usage.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    (adj_user, adj_pw), customers = logins(args.logins)
    adj = session(args.base, adj_user, adj_pw)
    photos = [Path(p) for p in args.photos]
    out = open(args.out, "w", encoding="utf-8")  # noqa: SIM115
    for i in range(args.claims):
        user, pw = customers[i % len(customers)]
        cust = session(args.base, user, pw)
        policies = [
            p
            for p in cust.get("/me/policies").json()["policies"]
            if p["status"] == "active"
        ]
        if not policies:
            continue
        policy = policies[0]
        incident = min(
            date.today(), date.fromisoformat(policy["end_date"])
        ) - timedelta(days=2)
        chosen = rng.sample(photos, k=rng.randint(1, min(2, len(photos))))
        files = [("photos", (p.name, p.read_bytes(), "image/jpeg")) for p in chosen]
        t0 = time.time()
        r = cust.post(
            "/claims",
            data={
                "policy_id": policy["policy_id"],
                "incident_date": incident.isoformat(),
                "description": DESCRIPTIONS[i % len(DESCRIPTIONS)],
            },
            files=files,
        )
        r.raise_for_status()
        claim_id = r.json()["claim_id"]
        paused = wait(adj, claim_id, {"awaiting_approval", "awaiting_retake", "failed"})
        to_adjuster_s = time.time() - t0
        row = {
            "claim_id": claim_id,
            "photos": len(chosen),
            "paused_at": paused["status"],
            "to_adjuster_s": round(to_adjuster_s, 1),
            "sources": {
                s["node"]: s.get("source") for s in paused["trace"] if s.get("source")
            },
            "usage_before_decision": paused["llm_usage"],
        }
        if paused["status"] == "awaiting_approval":
            adj.post(
                f"/claims/{claim_id}/decision", json={"action": "approve", "notes": ""}
            )
            done = wait(adj, claim_id, {"completed", "failed"})
            row["usage_total"] = done["llm_usage"]
            row["final_status"] = done["status"]
        u = row.get("usage_total") or row["usage_before_decision"] or {}
        print(
            f"{i + 1:>2} {claim_id[:8]} {row['paused_at']:<17} {to_adjuster_s:5.1f}s "
            f"calls={u.get('calls')} in={u.get('input_tokens')} out={u.get('output_tokens')} "
            f"est=${u.get('est_cost_usd')}"
        )
        out.write(json.dumps(row) + "\n")
        out.flush()
        cust.close()


if __name__ == "__main__":
    main()
