-- Per-service database roles: each service can reach only the tables it uses,
-- so a bug or an injection in one service can't read logins, fraud labels or
-- another service's data.
--
--   claimlens_gateway         SELECT auth.users, policies, vehicles
--   claimlens_agent           its own schema "agent" (LangGraph checkpoints,
--                             agent_claims); INSERT into claims, claim_photos
--   claimlens_policy          SELECT policies, vehicles, policy_wordings, policy_chunks
--   claimlens_pricing         SELECT the four pricing tables
--   claimlens_claims_history  SELECT claims, claim_photos
--
-- Nobody but the owner (claimlens: scripts, ingest, migrations) sees eval.*.
--
-- Runs after schema.sql on a fresh volume (docker-compose.yml mounts it as
-- 002_roles.sql) and is applied to an existing database by
-- scripts/apply_db_roles.py. Idempotent. No passwords here: they're set from
-- .env by db/role-passwords.sh (fresh volume) or the script.

DO $$
DECLARE
    r text;
BEGIN
    FOREACH r IN ARRAY ARRAY['claimlens_gateway', 'claimlens_agent', 'claimlens_policy',
                             'claimlens_pricing', 'claimlens_claims_history'] LOOP
        IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = r) THEN
            EXECUTE format('CREATE ROLE %I LOGIN', r);
        END IF;
    END LOOP;
    -- Only these roles (and the owner) may connect at all.
    EXECUTE format('REVOKE ALL ON DATABASE %I FROM PUBLIC', current_database());
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO claimlens_gateway, claimlens_agent, '
                   'claimlens_policy, claimlens_pricing, claimlens_claims_history',
                   current_database());
END $$;

-- Postgres 15+ already denies CREATE in public to everyone but the owner;
-- said explicitly so an older or changed default can't widen it.
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA auth, eval FROM PUBLIC;

-- gateway: logins and the customer's own policies (/me/policies).
GRANT USAGE ON SCHEMA auth TO claimlens_gateway;
GRANT SELECT ON auth.users TO claimlens_gateway;
GRANT SELECT ON policies, vehicles TO claimlens_gateway;

-- agent: owns a schema for the tables it creates itself, and only inserts
-- decided claims (and their photo hashes) into the shared tables.
CREATE SCHEMA IF NOT EXISTS agent AUTHORIZATION claimlens_agent;
ALTER ROLE claimlens_agent SET search_path = agent, public;
-- INSERT only, not SELECT: ClaimRecorder uses ON CONFLICT DO NOTHING without
-- a conflict target, which (measured) needs no read access; naming the
-- target column would need SELECT on it.
GRANT INSERT ON claims, claim_photos TO claimlens_agent;

-- Before Phase 4 the agent created its tables in public as the owner; move
-- them (data included) into its schema and hand them over.
DO $$
DECLARE
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['checkpoints', 'checkpoint_blobs', 'checkpoint_writes',
                             'checkpoint_migrations', 'agent_claims'] LOOP
        IF to_regclass('public.' || t) IS NOT NULL AND to_regclass('agent.' || t) IS NULL THEN
            EXECUTE format('ALTER TABLE public.%I SET SCHEMA agent', t);
        END IF;
        IF to_regclass('agent.' || t) IS NOT NULL THEN
            EXECUTE format('ALTER TABLE agent.%I OWNER TO claimlens_agent', t);
        END IF;
    END LOOP;
END $$;

-- MCP servers: read only, and only their own tables.
GRANT SELECT ON policies, vehicles, policy_wordings, policy_chunks TO claimlens_policy;
GRANT SELECT ON labour_rates, segment_multipliers, repair_operations, pricing_settings
    TO claimlens_pricing;
GRANT SELECT ON claims, claim_photos TO claimlens_claims_history;
