#!/usr/bin/env bash
# Run the k6 load test against ClaimLens on kind and keep the results.
#
#   k8s/loadtest/make-users.sh                 # once: 20 load-test logins
#   k8s/loadtest/run.sh browse,detect          # the read path and the vision queue
#   k8s/loadtest/run.sh claims                 # full claims, with the LLM key removed
#
# k6 runs in Docker on kind's network and calls the NGINX NodePort directly
# (http://claimlens-control-plane:30080): the same edge, routes and policies
# a browser goes through. The summary lands in k8s/loadtest/results/.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
SCENARIO=${1:-browse,detect}
K6_IMAGE=${K6_IMAGE:-grafana/k6:1.4.2}
PHOTO=${PHOTO:-services/vision/fixtures/panel.jpg}
NAME=$(date +%Y-%m-%d-%H%M)-${SCENARIO//,/-}
OUT=k8s/loadtest/results/$NAME
mkdir -p "$OUT"
[[ -f data/loadtest/users.json ]] || { echo "no data/loadtest/users.json: run k8s/loadtest/make-users.sh" >&2; exit 1; }

win() {  # Docker Desktop on Windows wants C:/... for bind mounts
  if command -v cygpath >/dev/null; then cygpath -m "$1"; else echo "$1"; fi
}

# One client IP sends everything, so the per-IP limit on /api would turn the
# test into a test of the limit. Dry-run mode keeps it configured (NGINX logs
# what it would have rejected) without rejecting.
helm upgrade claimlens k8s/charts/claimlens -n claimlens --reuse-values \
  --set edge.rateLimit.dryRun=true >/dev/null
restore=(--set edge.rateLimit.dryRun=false)
if [[ ",$SCENARIO," == *",claims,"* ]]; then
  # Every LLM step falls back to its rule-based default without a key
  # (services/agent/graph.py): the load is ours, not Ollama Cloud's, and the
  # test doesn't spend the free tier's quota. Tracing goes off too, so a
  # hundred test claims don't land in Langfuse.
  helm upgrade claimlens k8s/charts/claimlens -n claimlens --reuse-values \
    --set agent.secretEnv.OLLAMA_API_KEY=null \
    --set agent.secretEnv.LANGFUSE_PUBLIC_KEY=null \
    --set agent.secretEnv.LANGFUSE_SECRET_KEY=null >/dev/null
  kubectl -n claimlens rollout status deployment/agent --timeout 5m
  for key in OLLAMA_API_KEY LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY; do
    restore+=(--set "agent.secretEnv.$key.key=$key" --set "agent.secretEnv.$key.optional=true")
  done
fi
trap 'helm upgrade claimlens k8s/charts/claimlens -n claimlens --reuse-values "${restore[@]}" >/dev/null' EXIT

# Where the cluster stands when the test starts, for the write-up.
kubectl -n claimlens get deploy,hpa,scaledobject -o wide > "$OUT/before.txt" 2>&1 || true

# A timeline from Prometheus every 10 s: replicas, queue lag, rates, node CPU
# and memory (k8s/loadtest/sample.py).
python k8s/loadtest/sample.py --out "$OUT/timeline.csv" --minutes 60 --every 10 &
SAMPLER=$!

MSYS_NO_PATHCONV=1 docker run --rm --network kind \
  -v "$(win "$ROOT/k8s/loadtest"):/scripts:ro" \
  -v "$(win "$ROOT/data/loadtest"):/data:ro" \
  -v "$(win "$ROOT/$PHOTO"):/photo.jpg:ro" \
  -v "$(win "$ROOT/$OUT"):/out" \
  -e SCENARIO="$SCENARIO" -e BASE_URL="${BASE_URL:-}" \
  -e CLAIMS_PER_MIN="${CLAIMS_PER_MIN:-}" -e CLAIMS_DURATION="${CLAIMS_DURATION:-}" \
  "$K6_IMAGE" run --quiet --summary-export /out/summary.json --out csv=/out/requests.csv.gz \
  /scripts/claimlens.js 2>&1 | tee "$OUT/k6.txt" || true  # a crossed threshold is a result, not a reason to stop

touch "$OUT/timeline.csv.stop"
wait $SAMPLER || true
kubectl -n claimlens get deploy,hpa,scaledobject -o wide > "$OUT/after.txt" 2>&1 || true
python k8s/loadtest/analyze.py "$OUT" | tee "$OUT/steps.md"
echo "results: $OUT"
