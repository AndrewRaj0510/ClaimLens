#!/usr/bin/env bash
# ClaimLens on kind, from nothing to http://localhost:8080.
#
#   k8s/kind/up.sh              # cluster, platform, images, ClaimLens
#   SKIP_BUILD=1 k8s/kind/up.sh # reuse the images already loaded
#   k8s/kind/seed.sh            # then: data, policy wordings, logins
#
# Needs: Docker, kind, helm, kubectl, .env (cp .env.example .env) and the
# models in data/models (best.onnx; python scripts/download_models.py for
# the rest). Idempotent: re-run it after a change.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
CLUSTER=claimlens
NS=claimlens

# Pinned versions of everything installed into the cluster.
NGF_VERSION=2.8.0
KEDA_VERSION=2.21.0
KPS_VERSION=92.16.1
METRICS_SERVER_VERSION=3.14.0

say() { printf '\n== %s\n' "$*"; }

for tool in docker kind helm kubectl; do
  command -v "$tool" >/dev/null || { echo "missing: $tool" >&2; exit 1; }
done
[[ -f .env ]] || { echo "no .env (cp .env.example .env)" >&2; exit 1; }
[[ -f data/models/best.onnx ]] || echo "warning: no data/models/best.onnx; vision jobs will get 503 results"

say "cluster"
if ! kind get clusters | grep -qx "$CLUSTER"; then
  MODELS_DIR="$ROOT/data/models"
  CONFIG=$(mktemp)
  # Git Bash on Windows: kind.exe and Docker Desktop want C:/..., not /c/...
  if command -v cygpath >/dev/null; then
    MODELS_DIR=$(cygpath -m "$MODELS_DIR")
    CONFIG=$(cygpath -m "$CONFIG")
  fi
  sed "s|\${MODELS_DIR}|$MODELS_DIR|" k8s/kind/cluster.yaml > "$CONFIG"
  kind create cluster --config "$CONFIG" --wait 120s
  rm -f "$CONFIG"
fi
kubectl config use-context "kind-$CLUSTER" >/dev/null

say "Gateway API CRDs (the version NGINX Gateway Fabric $NGF_VERSION supports)"
kubectl kustomize "https://github.com/nginx/nginx-gateway-fabric/config/crd/gateway-api/standard?ref=v$NGF_VERSION" | kubectl apply -f - >/dev/null

say "platform: metrics-server, Prometheus + Grafana, KEDA, NGINX Gateway Fabric"
helm repo add metrics-server https://kubernetes-sigs.github.io/metrics-server/ >/dev/null 2>&1 || true
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
helm repo add kedacore https://kedacore.github.io/charts >/dev/null 2>&1 || true
helm repo update >/dev/null

helm upgrade --install metrics-server metrics-server/metrics-server --version "$METRICS_SERVER_VERSION" \
  -n kube-system -f k8s/platform/metrics-server-kind.yaml --wait

kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f - >/dev/null
if ! kubectl -n monitoring get secret grafana-admin >/dev/null 2>&1; then
  # A random admin password, never printed; read it back with kubectl (k8s/README.md).
  password=$(python -c "import secrets; print(secrets.token_urlsafe(18))")
  kubectl -n monitoring create secret generic grafana-admin \
    --from-literal=admin-user=admin --from-literal=admin-password="$password" >/dev/null
  unset password
fi
helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack --version "$KPS_VERSION" \
  -n monitoring -f k8s/platform/kube-prometheus-stack.yaml --wait --timeout 10m

helm upgrade --install keda kedacore/keda --version "$KEDA_VERSION" \
  -n keda --create-namespace -f k8s/platform/keda.yaml --wait

helm upgrade --install ngf oci://ghcr.io/nginx/charts/nginx-gateway-fabric --version "$NGF_VERSION" \
  -n nginx-gateway --create-namespace -f k8s/platform/nginx-gateway-fabric-kind.yaml --wait

say "namespace, secret, database init scripts"
kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
k8s/scripts/create-secret.sh "$NS" .env
kubectl -n "$NS" create configmap postgres-initdb \
  --from-file=001_schema.sql=db/schema.sql \
  --from-file=002_roles.sql=db/roles.sql \
  --from-file=003_role_passwords.sh=db/role-passwords.sh \
  --dry-run=client -o yaml | kubectl apply -f -

if [[ -z "${SKIP_BUILD:-}" ]]; then
  say "images: build with compose, tag claimlens/<name>:dev, load into kind"
  docker compose build frontend gateway agent guardrails vision-worker eval policy pricing claims-history
  for svc in frontend gateway agent guardrails eval policy pricing claims-history; do
    docker tag "claimlens-$svc:latest" "claimlens/$svc:dev"
  done
  docker tag claimlens-vision-worker:latest claimlens/vision:dev
  kind load docker-image --name "$CLUSTER" \
    claimlens/frontend:dev claimlens/gateway:dev claimlens/agent:dev claimlens/guardrails:dev \
    claimlens/vision:dev claimlens/eval:dev claimlens/policy:dev claimlens/pricing:dev \
    claimlens/claims-history:dev
  # Third-party images (Postgres, Redis, SeaweedFS) are pulled by the node
  # instead: Docker Desktop's containerd store keeps only this machine's
  # platform of a multi-platform image, and `kind load` then fails on it.
fi

say "ClaimLens"
k8s/scripts/build-charts.sh
helm upgrade --install claimlens k8s/charts/claimlens -n "$NS" \
  -f k8s/charts/claimlens/values-kind.yaml --wait --timeout 15m
# A rollout picks up new images loaded under the same tag.
if [[ -z "${SKIP_BUILD:-}" ]]; then
  kubectl -n "$NS" rollout restart deployment >/dev/null
  kubectl -n "$NS" rollout status deployment --timeout 10m
fi

say "up"
kubectl -n "$NS" get pods
cat <<EOF

ClaimLens:  http://localhost:8080   (first run: k8s/kind/seed.sh)
Grafana:    kubectl -n monitoring port-forward svc/kube-prometheus-stack-grafana 3001:80
            -> http://localhost:3001, user admin, password:
            kubectl -n monitoring get secret grafana-admin -o jsonpath='{.data.admin-password}' | base64 -d
EOF
