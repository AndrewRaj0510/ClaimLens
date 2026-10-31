"""Per-step results of a k6 step test, from its CSV output.

    python k8s/loadtest/analyze.py k8s/loadtest/results/<run>

Reads requests.csv.gz and the step schedule from claimlens.js, drops each
step's 10 s ramp, and prints a Markdown table per scenario: target and
achieved rate, latency percentiles, errors and dropped iterations (requests
k6 couldn't start because every VU was still waiting: the system was too
slow for the target rate).
"""

import csv
import gzip
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RAMP_S = 10


def schedule():
    js = (HERE / "claimlens.js").read_text()
    out = {}
    for name in ("browse", "detect"):
        rates = re.search(rf"{name}: \[([\d, ]+)\]", js).group(1)
        hold = re.search(rf'steps\(STEPS\.{name}, "(\d+)s"\)', js).group(1)
        out[name] = ([int(r) for r in rates.split(",")], int(hold))
    return out


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round(q / 100 * (len(values) - 1)))]


def ms(v):
    return "" if v is None else (f"{v / 1000:.2f} s" if v >= 1000 else f"{v:.0f} ms")


def main(run):
    rows = defaultdict(list)  # scenario -> [(t, metric, value, tags)]
    with gzip.open(Path(run) / "requests.csv.gz", "rt", newline="") as f:
        for r in csv.DictReader(f):
            if r["scenario"] in ("browse", "detect", "claims"):
                rows[r["scenario"]].append(
                    (int(r["timestamp"]), r["metric_name"], float(r["metric_value"]), r)
                )
    for name, (rates, hold) in schedule().items():
        data = rows.get(name)
        if not data:
            continue
        t0 = min(t for t, *_ in data)
        print(f"\n### {name}\n")
        latency = "end-to-end p50 | p95" if name == "detect" else "p50 | p95 | p99"
        print(f"| target /s | achieved /s | {latency} | errors | dropped |")
        print("|---|---|---|---|---|" + ("" if name == "detect" else "---|"))
        for i, rate in enumerate(rates):
            lo = t0 + i * (RAMP_S + hold) + RAMP_S
            hi = lo + hold
            win = [(m, v, tags) for t, m, v, tags in data if lo <= t < hi]
            iters = sum(v for m, v, _ in win if m == "iterations")
            failed = [v for m, v, _ in win if m == "http_req_failed"]
            dropped = sum(v for m, v, _ in win if m == "dropped_iterations")
            err = f"{100 * sum(failed) / len(failed):.1f}%" if failed else ""
            if name == "detect":
                e2e = [v for m, v, _ in win if m == "detect_total_ms"]
                lat = f"{ms(pct(e2e, 50))} | {ms(pct(e2e, 95))}"
            else:
                d = [v for m, v, _ in win if m == "http_req_duration"]
                lat = f"{ms(pct(d, 50))} | {ms(pct(d, 95))} | {ms(pct(d, 99))}"
            print(f"| {rate} | {iters / hold:.1f} | {lat} | {err} | {dropped:.0f} |")


if __name__ == "__main__":
    main(sys.argv[1])
