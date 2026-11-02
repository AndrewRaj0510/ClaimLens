#!/usr/bin/env bash
# Create 20 demo customers with one random password for the load test, and
# write their logins to data/loadtest/users.json (git-ignored). They're the
# 101st-120th eligible customers, so seed.sh's demo logins keep their
# passwords. Nothing is printed: create_users.py's output goes straight into
# the file.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PORT=${SEED_PG_PORT:-5434}
PYTHON=${PYTHON:-python}
mkdir -p data/loadtest

kubectl -n claimlens port-forward svc/postgres "$PORT:5432" >/dev/null &
FORWARD=$!
trap 'kill $FORWARD 2>/dev/null' EXIT
for _ in $(seq 30); do
  (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && break
  sleep 1
done

POSTGRES_PASSWORD=$(grep -E '^POSTGRES_PASSWORD=' .env | cut -d= -f2- | tr -d '\r')
export DATABASE_URL="postgresql://claimlens:${POSTGRES_PASSWORD}@127.0.0.1:${PORT}/claimlens"
unset POSTGRES_PASSWORD
DEMO_PASSWORD=$("$PYTHON" -c "import secrets; print(secrets.token_urlsafe(18))")
export DEMO_PASSWORD

"$PYTHON" scripts/create_users.py --demo-customers "${USERS:-20}" --demo-offset 100 \
  | "$PYTHON" -c "
import json, os, sys
users = [{'username': line.split()[1], 'password': os.environ['DEMO_PASSWORD']}
         for line in sys.stdin if line.startswith('customer')]
json.dump(users, open('data/loadtest/users.json', 'w'))
print(len(users), 'load-test logins in data/loadtest/users.json')
"
