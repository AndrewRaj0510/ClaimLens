#!/bin/sh
# Sets the per-service role passwords (db/roles.sql) from the environment.
# Runs once, on a fresh volume, after 002_roles.sql: docker-compose.yml mounts
# it as 003_role_passwords.sh and passes DB_PASSWORD_* in. For an existing
# volume, run: python scripts/apply_db_roles.py
set -eu

set_password() {
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
         -v role="$1" -v password="$2" <<'SQL'
ALTER ROLE :"role" PASSWORD :'password';
SQL
}

set_password claimlens_gateway "$DB_PASSWORD_GATEWAY"
set_password claimlens_agent "$DB_PASSWORD_AGENT"
set_password claimlens_policy "$DB_PASSWORD_POLICY"
set_password claimlens_pricing "$DB_PASSWORD_PRICING"
set_password claimlens_claims_history "$DB_PASSWORD_CLAIMS_HISTORY"
