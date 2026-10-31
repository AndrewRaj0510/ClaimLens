# ClaimLens on Kubernetes

Helm charts, the kind setup for a laptop, the platform pieces (edge,
autoscaling, monitoring) and the load test.

```
k8s/
  charts/
    claimlens-lib/     library chart: Deployment, Service, HPA, PDB, ServiceMonitor,
                       NetworkPolicy and StatefulSet templates, written once
    <service>/         one chart per service: values.yaml + a one-line template
    claimlens/         umbrella chart: all services, plus the edge (Gateway API),
                       default-deny NetworkPolicy, Grafana dashboards
      values-kind.yaml   laptop overrides
      values-cloud.yaml  starting point for a managed cluster
  platform/            values for the cluster-wide charts (NGINX Gateway Fabric,
                       KEDA, kube-prometheus-stack, metrics-server)
  kind/                cluster config, up.sh, seed.sh, down.sh
  scripts/             build-charts.sh, create-secret.sh
  loadtest/            k6 script, runner, results
```

## How a request flows

```
browser -> NGINX (Gateway API, NGINX Gateway Fabric)   rate limit on /api, 105 MB bodies
  -> frontend (Next.js)     /api/* proxied with a path + header allowlist
    -> gateway              sessions + per-account limits in Redis
      -> agent              LangGraph; run leases in Redis, state in Postgres
        -> guardrails, policy / pricing / claims-history (MCP), Ollama Cloud
      -> Redis stream vision:jobs -> vision workers (KEDA: scaled on the stream's lag)
Postgres + pgvector, Redis, SeaweedFS: StatefulSets with their own volumes
```

Every service is a Deployment behind a ClusterIP Service named after the
chart, so services reach each other by the same names as in docker-compose
(`http://gateway:8000`). Kubernetes Services spread requests over the
replicas; the edge is the only way in.

## On a laptop (kind)

Needs Docker (about 8 GB for Docker Desktop), kind, helm, kubectl, Python with
the scripts' requirements, `.env` (`cp .env.example .env`) and the models in
`data/models` (`best.onnx` from training, `python scripts/download_models.py`
for the rest).

```bash
k8s/kind/up.sh       # cluster, platform, images, ClaimLens (about 15 minutes the first time)
k8s/kind/seed.sh     # synthetic data, policy wordings, an adjuster and 3 demo customers
open http://localhost:8080
```

`up.sh` is idempotent: re-run it after a code change (it rebuilds and reloads
the images and restarts the Deployments), or `SKIP_BUILD=1 k8s/kind/up.sh`
after a chart change. `k8s/kind/down.sh` deletes the cluster.

Grafana:

```bash
kubectl -n monitoring port-forward svc/kube-prometheus-stack-grafana 3001:80
kubectl -n monitoring get secret grafana-admin -o jsonpath='{.data.admin-password}' | base64 -d
# http://localhost:3001, user admin; dashboard "ClaimLens overview"
```

## What's in the cluster

| Namespace | What | Why |
|---|---|---|
| `claimlens` | the 11 services (eval off by default), the Gateway and its NGINX pods | the app |
| `nginx-gateway` | NGINX Gateway Fabric 2.8 control plane | turns Gateway/HTTPRoute objects into NGINX config |
| `keda` | KEDA 2.21 | scales the vision workers on the Redis stream's lag |
| `monitoring` | kube-prometheus-stack 92.16 (Prometheus, Grafana, operator, kube-state-metrics, node-exporter) | metrics and dashboards |
| `kube-system` | metrics-server 0.9 | pod CPU/memory for the HPAs |

### Autoscaling

| Workload | Scaler | Signal | kind (min-max) | default (min-max) |
|---|---|---|---|---|
| frontend | HPA | CPU 70% of request | 1-3 | 2-4 |
| gateway | HPA | CPU 70% | 1-4 | 2-6 |
| agent | HPA | CPU 70%, 5 min scale-down window | 1-3 | 2-4 |
| guardrails | HPA | CPU 70% | 1-1 (memory) | 1-3 |
| policy, pricing, claims-history | HPA | CPU 70% | 1-2 | 1-3 |
| vision-worker | KEDA | stream lag, 5 per worker | 1-3 | 1-6 |

The agent can run several replicas because its run state is in Redis
(`services/agent/runstate.py`). On a scale-down it
drains running claims for up to 90 s before the pod stops.

### Security

- **Default deny:** no pod accepts traffic unless its chart names the caller
  (`networkPolicy.from`). kind's default CNI (kindnet) enforces NetworkPolicy.
- **Pods:** non-root (UID 1000; 999 for Postgres and Redis), no privilege
  escalation, all capabilities dropped, RuntimeDefault seccomp, no service
  account token.
- **Secrets:** one Secret from `.env` (`k8s/scripts/create-secret.sh`), and
  each pod gets only the keys its chart lists. Values never reach the charts
  or the terminal.
- **Edge:** per-IP rate limit on `/api` (20 r/s, burst 40), body limit, TLS
  and an HTTP->HTTPS redirect with `edge.tls.enabled` (values-cloud.yaml).

## Load test

```bash
k8s/loadtest/make-users.sh          # once: 20 load-test logins -> data/loadtest/users.json
k8s/loadtest/run.sh browse,detect   # reads + the vision queue
k8s/loadtest/run.sh claims          # full claims, with the agent's LLM key removed for the run
```

k6 runs in Docker on kind's network and calls the NGINX NodePort, so every
request takes the public path. Each run's folder in `results/` has the
per-step table (`steps.md`), k6's summary and a Prometheus timeline.

Results on one laptop node (20 CPUs, 7.6 GiB for the cluster and k6):

| Path | Capacity | What limits it | What scaled |
|---|---|---|---|
| Browse (3 reads per iteration) | 200 iterations/s = 600 API requests/s at p95 29 ms; at 400/s, 32% errors | the Next.js `/api` proxy's CPU, at its kind maximum of 3 pods | gateway 1→4, frontend 1→3, agent 1→3 (HPA) |
| Detect (uploads) | 3/s at p95 4.5 s (the scale-up itself); at most about 4.2 jobs/s, and 0 errors beyond it: requests wait and poll | 3 workers (the kind cap), about 0.6 s per job each | vision workers 1→3 (KEDA, within 25 s) |
| Claims (LLM steps off) | 121 at 30/min, all reached the adjuster, 0 failed requests | the per-customer submission limit | nothing needed to |

The first run, with both scenarios at once and up to 700 VUs, ran the node
out of memory: memory requests were below real use, so pods kept being
placed, the node swapped, and liveness probes restarted pods that were only
slow. That run (`results/2026-10-31-1420-browse-detect`) led to the memory,
probe, connection-pool and Redis-wait fixes listed under "Gotchas".

## Cloud

Not deployed yet; the provider isn't chosen. What's ready:

- CI pushes every image to GHCR on each commit to main
  (`ghcr.io/<owner>/claimlens/<name>:<sha>`).
- `values-cloud.yaml`: GHCR images, TLS on a host name, two replicas and a
  PodDisruptionBudget on the request path, bigger volumes.
- The platform charts install the same way as in `kind/up.sh`, minus the
  kind-specific values (NGINX gets a cloud LoadBalancer; the provider runs
  metrics-server).

Still to decide per provider: where the models live (a volume several nodes
can mount read-only, or an init container that downloads them), managed
Postgres/Redis or the in-cluster ones, and certificates (cert-manager).

## Gotchas found while building this

- **Helm library charts are copied into every chart that uses them**, and
  named templates are global across a release, so a stale copy in one chart
  silently overrides a fix in the others. `scripts/build-charts.sh` rebuilds
  every chart, every time.
- **Helm merges maps.** A default `modelsVolume` PVC plus kind's `hostPath`
  rendered one volume with both sources; the default is now empty, and a
  chart that needs it fails with a clear message.
- **ONNX Runtime sees the node's cores, not the pod's CPU limit**, so the
  charts set `ORT_THREADS` (`services/guardrails/text_scan.py` explains).
- **Git Bash paths:** kind.exe can't read `/tmp/...`; `up.sh` converts paths
  with `cygpath -m` on Windows.
- **Memory requests must match real use.** Below it, the scheduler (and
  KEDA) placed pods on a node that couldn't hold them; it swapped, and
  liveness probes then restarted pods that were only slow. Requests are now
  measured values, and liveness probes allow 5 s and three failures.
- **Every database pool checks connections** before use: a Postgres restart
  otherwise left a dead connection in each pool, and each one failed a request.
- **Blocking Redis waits are capped** (`DETECT_WAITERS`): 100+ uploads waiting
  in BLPOP took the gateway's whole Redis pool, sessions included.
- **`kind load` fails on multi-platform images** from Docker Desktop's
  containerd store (only one platform's layers are local), so Postgres,
  Redis and SeaweedFS are pulled by the node instead.
