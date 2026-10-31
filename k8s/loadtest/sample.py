"""Sample the cluster every few seconds during a load test: replicas per
workload, the vision queue's lag, request rate and node CPU/memory, all from
Prometheus. Writes a CSV next to the k6 results, for the scaling timeline.

    python k8s/loadtest/sample.py --out k8s/loadtest/results/<run>/timeline.csv

It runs its own kubectl port-forward to Prometheus and re-opens it if it
drops (it does when Prometheus restarts), and stops when <out>.stop appears
(k8s/loadtest/run.sh creates it when k6 is done) or after --minutes.
"""

import argparse
import csv
import subprocess
import time
from pathlib import Path

import httpx

PROM = "http://localhost:9090/api/v1/query"
NS = 'namespace="claimlens"'
QUERIES = {
    "gateway": f'kube_deployment_status_replicas_available{{{NS},deployment="gateway"}}',
    "frontend": f'kube_deployment_status_replicas_available{{{NS},deployment="frontend"}}',
    "agent": f'kube_deployment_status_replicas_available{{{NS},deployment="agent"}}',
    "guardrails": f'kube_deployment_status_replicas_available{{{NS},deployment="guardrails"}}',
    "vision_workers": f'kube_deployment_status_replicas_available{{{NS},deployment="vision-worker"}}',
    "queue_lag": 'max(keda_scaler_metrics_value{scaledObject="vision-worker"})',
    "edge_rps": f"sum(rate(nginx_http_requests_total{{{NS}}}[1m]))",
    "vision_jobs_per_s": f"sum(rate(vision_jobs_total{{{NS}}}[30s]))",
    "gateway_cpu_vs_request": (
        f'sum(rate(container_cpu_usage_seconds_total{{{NS},container="gateway"}}[30s]))'
        f' / sum(kube_pod_container_resource_requests{{{NS},container="gateway",resource="cpu"}})'
    ),
    "node_cpu_cores": 'sum(rate(node_cpu_seconds_total{mode!="idle"}[30s]))',
    "node_memory_used_gib": "(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / 2^30",
    "node_swap_used_gib": "(node_memory_SwapTotal_bytes - node_memory_SwapFree_bytes) / 2^30",
}


def value(client, query):
    try:
        result = client.get(PROM, params={"query": query}).json()["data"]["result"]
        return round(float(result[0]["value"][1]), 3) if result else ""
    except (httpx.HTTPError, ValueError, KeyError, IndexError):
        return ""  # Prometheus restarting or the port-forward reconnecting


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=9)
    ap.add_argument("--every", type=float, default=10)
    args = ap.parse_args()
    stop = Path(args.out + ".stop")
    forward = None
    start = time.time()
    try:
        with httpx.Client(timeout=10) as client, open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", *QUERIES])
            while time.time() - start < args.minutes * 60 and not stop.exists():
                if forward is None or forward.poll() is not None:
                    forward = subprocess.Popen(
                        [
                            "kubectl",
                            "-n",
                            "monitoring",
                            "port-forward",
                            "svc/kube-prometheus-stack-prometheus",
                            "9090:9090",
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    time.sleep(2)
                w.writerow(
                    [
                        round(time.time() - start),
                        *(value(client, q) for q in QUERIES.values()),
                    ]
                )
                f.flush()
                time.sleep(args.every)
    finally:
        if forward:
            forward.terminate()
        stop.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
