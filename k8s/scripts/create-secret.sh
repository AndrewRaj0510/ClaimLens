#!/usr/bin/env bash
# Create (or update) the claimlens-env Secret from .env, with only the keys
# the charts use. Values are never printed; they go straight into the
# Secret's YAML on kubectl's stdin. Empty values are left out, so optional
# keys (OLLAMA_API_KEY, the Langfuse keys) are simply absent.
#
#   k8s/scripts/create-secret.sh [namespace] [env file]
#
# For a real cluster, prefer a secret manager (External Secrets, Sealed
# Secrets); this is the same Secret by hand.
set -euo pipefail
NAMESPACE=${1:-claimlens}
ENV_FILE=${2:-$(dirname "$0")/../../.env}

KEYS=(
  POSTGRES_PASSWORD
  DB_PASSWORD_GATEWAY DB_PASSWORD_AGENT DB_PASSWORD_POLICY DB_PASSWORD_PRICING DB_PASSWORD_CLAIMS_HISTORY
  REDIS_PASSWORD
  S3_ACCESS_KEY S3_SECRET_KEY
  OLLAMA_API_KEY LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY
)
pattern="^($(IFS='|'; echo "${KEYS[*]}"))="

[[ -f "$ENV_FILE" ]] || { echo "no $ENV_FILE (cp .env.example .env)" >&2; exit 1; }
{
  echo "apiVersion: v1"
  echo "kind: Secret"
  echo "metadata: {name: claimlens-env, namespace: $NAMESPACE}"
  echo "type: Opaque"
  echo "data:"
  grep -E "$pattern" "$ENV_FILE" | tr -d '\r' | while IFS='=' read -r key value; do
    if [[ -n "$value" ]]; then
      printf '  %s: %s\n' "$key" "$(printf %s "$value" | base64 | tr -d '\n')"
    fi
  done
} | kubectl apply -f -
