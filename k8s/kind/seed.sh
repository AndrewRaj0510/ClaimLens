#!/usr/bin/env bash
# Load the data into the cluster's Postgres, the same way as for compose
# (README "Running it locally"): synthetic customers/policies/claims, the
# policy wordings for RAG, then logins. Runs the repo's scripts on this
# machine through a port-forward, so nothing extra runs in the cluster.
#
#   k8s/kind/seed.sh   # needs the venv: pip install -r scripts/requirements.txt -r mcp_servers/policy/requirements.txt
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
NS=claimlens
PORT=${SEED_PG_PORT:-5434}   # not 5433, which compose's Postgres uses
PYTHON=${PYTHON:-python}

kubectl -n "$NS" port-forward svc/postgres "$PORT:5432" >/dev/null &
FORWARD=$!
trap 'kill $FORWARD 2>/dev/null' EXIT
for _ in $(seq 30); do
  (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && break
  sleep 1
done

# The owner role, for loading. Read from .env without echoing it.
POSTGRES_PASSWORD=$(grep -E '^POSTGRES_PASSWORD=' .env | cut -d= -f2- | tr -d '\r')
export DATABASE_URL="postgresql://claimlens:${POSTGRES_PASSWORD}@127.0.0.1:${PORT}/claimlens"
unset POSTGRES_PASSWORD

[[ -f data/synthetic/customers.json ]] || "$PYTHON" scripts/generate_synthetic.py --out data/synthetic
"$PYTHON" scripts/load_synthetic.py --data data/synthetic
"$PYTHON" mcp_servers/policy/ingest.py --pdf-dir data/policies --download
# An adjuster (asks for a password, or set ADJUSTER_PASSWORD) and demo
# customers (their passwords are printed once).
"$PYTHON" scripts/create_users.py --adjuster "${ADJUSTER:-sam}" --demo-customers "${DEMO_CUSTOMERS:-3}"
