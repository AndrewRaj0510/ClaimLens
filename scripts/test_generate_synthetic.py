from collections import defaultdict
from datetime import date, timedelta

import pytest

from generate_synthetic import (
    AS_OF,
    BURST_WINDOW_DAYS,
    DAMAGE_CLASSES,
    DUPLICATE_MAX_BIT_FLIPS,
    EARLY_CLAIM_MAX_DAYS,
    FRAUD_PATTERNS,
    INFLATION_MIN_RATIO,
    LATE_REPORT_MIN_DAYS,
    SEVERITIES,
    WORDINGS,
    expected_cost,
    generate,
    write,
)


@pytest.fixture(scope="module")
def data():
    return generate(seed=42)


@pytest.fixture(scope="module")
def index(data):
    return {
        "policy": {p["policy_id"]: p for p in data["policies"]},
        "vehicle": {v["vehicle_id"]: v for v in data["vehicles"]},
        "label": {f["claim_id"]: f for f in data["fraud_labels"]},
    }


def d(s):
    return date.fromisoformat(s)


def test_same_seed_gives_identical_files(tmp_path):
    write(generate(seed=7, n_customers=200), tmp_path / "a")
    write(generate(seed=7, n_customers=200), tmp_path / "b")
    for f in (tmp_path / "a").iterdir():
        assert f.read_bytes() == (tmp_path / "b" / f.name).read_bytes(), f.name


def test_different_seed_gives_different_data():
    assert generate(seed=1, n_customers=200) != generate(seed=2, n_customers=200)


def test_ids_link_up(data, index):
    customers = {c["customer_id"] for c in data["customers"]}
    for v in data["vehicles"]:
        assert v["customer_id"] in customers
    for p in data["policies"]:
        assert index["vehicle"][p["vehicle_id"]]["customer_id"] == p["customer_id"]
        assert p["wording_id"] in WORDINGS
    for c in data["claims"]:
        policy = index["policy"][c["policy_id"]]
        assert (c["customer_id"], c["vehicle_id"]) == (
            policy["customer_id"],
            policy["vehicle_id"],
        )
    assert [f["claim_id"] for f in data["fraud_labels"]] == [
        c["claim_id"] for c in data["claims"]
    ]


def test_claim_dates_are_consistent(data, index):
    for c in data["claims"]:
        p = index["policy"][c["policy_id"]]
        incident, reported = d(c["incident_date"]), d(c["reported_date"])
        assert d(p["start_date"]) <= incident <= min(d(p["end_date"]), AS_OF)
        assert incident <= reported <= AS_OF


def test_claims_only_for_covered_perils(data, index):
    for c in data["claims"]:
        assert index["policy"][c["policy_id"]]["coverages"][c["peril"]]
        for dmg in c["damages"]:
            assert dmg["class"] in DAMAGE_CLASSES
            assert dmg["severity"] in SEVERITIES


def test_paid_amount_respects_deductible(data):
    for c in data["claims"]:
        if c["status"] == "paid":
            assert c["claimed_amount"] > c["deductible"]
            assert c["paid_amount"] == round(c["claimed_amount"] - c["deductible"], 2)
        else:
            assert c["paid_amount"] == 0
        if c["status"] == "below_deductible":
            assert c["claimed_amount"] <= c["deductible"]


def test_fraud_rate_and_every_pattern_present(data):
    labels = data["fraud_labels"]
    rate = sum(f["is_fraud"] for f in labels) / len(labels)
    assert 0.05 <= rate <= 0.15
    seen = {p for f in labels for p in f["patterns"]}
    assert seen == set(FRAUD_PATTERNS)


def test_labels_are_not_in_claims(data):
    # The agent reads claim history; ground truth must stay in a separate table.
    for c in data["claims"]:
        assert not {"is_fraud", "patterns", "_patterns"} & c.keys()


def hamming(a, b):
    return (int(a, 16) ^ int(b, 16)).bit_count()


def test_planted_patterns_are_actually_present(data, index):
    claims_by_policy = defaultdict(list)
    for c in data["claims"]:
        claims_by_policy[c["policy_id"]].append(c)

    for c in data["claims"]:
        patterns = index["label"][c["claim_id"]]["patterns"]
        policy = index["policy"][c["policy_id"]]
        incident = d(c["incident_date"])
        if "early_claim" in patterns:
            assert (incident - d(policy["start_date"])).days <= EARLY_CLAIM_MAX_DAYS
        if "late_report" in patterns:
            assert (d(c["reported_date"]) - incident).days >= LATE_REPORT_MIN_DAYS
        if "inflated_amount" in patterns:
            vehicle = index["vehicle"][c["vehicle_id"]]
            cost = expected_cost(c["damages"], vehicle["segment"], policy["state"])
            assert c["claimed_amount"] >= INFLATION_MIN_RATIO * cost - 0.01
        if "claim_burst" in patterns:
            near = [
                o
                for o in claims_by_policy[c["policy_id"]]
                if abs((d(o["incident_date"]) - incident).days) < BURST_WINDOW_DAYS
            ]
            assert len(near) >= 3
        if "duplicate_photo" in patterns:
            assert any(
                o["customer_id"] != c["customer_id"]
                and d(o["reported_date"]) < incident
                and hamming(o_hash, c["photo_phashes"][0]) <= DUPLICATE_MAX_BIT_FLIPS
                for o in data["claims"]
                for o_hash in o["photo_phashes"]
            )


def test_honest_claims_stay_below_fraud_thresholds(data, index):
    # Weak signals on honest claims must not cross the planted-pattern thresholds,
    # otherwise the labels would contradict the data.
    for c in data["claims"]:
        if index["label"][c["claim_id"]]["is_fraud"]:
            continue
        delay = d(c["reported_date"]) - d(c["incident_date"])
        assert delay < timedelta(days=LATE_REPORT_MIN_DAYS)
        policy = index["policy"][c["policy_id"]]
        vehicle = index["vehicle"][c["vehicle_id"]]
        cost = expected_cost(c["damages"], vehicle["segment"], policy["state"])
        assert c["claimed_amount"] < INFLATION_MIN_RATIO * cost


def test_pricing_covers_every_class_and_severity(data):
    ops = {(o["damage_class"], o["severity"]) for o in data["pricing"]["operations"]}
    assert ops == {(c, s) for c in DAMAGE_CLASSES for s in SEVERITIES}


def test_contact_details_are_fictional(data):
    for c in data["customers"]:
        assert c["email"].endswith("@example.com")
        assert "555-01" in c["phone"]
