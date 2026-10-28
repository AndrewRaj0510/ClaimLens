-- ClaimLens database schema. Mounted into the Postgres container's
-- docker-entrypoint-initdb.d, so it runs once when the data volume is created.
-- Idempotent (IF NOT EXISTS) so it can also be applied by hand.
--
-- Loaded by:
--   scripts/load_synthetic.py     customers ... claims, pricing, eval.fraud_labels
--   mcp_servers/policy/ingest.py  policy_wordings, policy_chunks
--   scripts/create_users.py       auth.users
-- The agent creates its own tables: LangGraph's checkpoints and agent_claims.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS customers (
    customer_id     text PRIMARY KEY,
    first_name      text NOT NULL,
    last_name       text NOT NULL,
    date_of_birth   date NOT NULL,
    state           char(2) NOT NULL,
    email           text NOT NULL,
    phone           text NOT NULL,
    customer_since  date NOT NULL
);

CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_id   text PRIMARY KEY,
    customer_id  text NOT NULL REFERENCES customers,
    vin          text NOT NULL UNIQUE,
    make         text NOT NULL,
    model        text NOT NULL,
    year         int NOT NULL,
    segment      text NOT NULL
);

-- No foreign key to policy_wordings: wordings are ingested separately and a
-- policy should still load if its wording PDF hasn't been ingested yet.
CREATE TABLE IF NOT EXISTS policies (
    policy_id                 text PRIMARY KEY,
    customer_id               text NOT NULL REFERENCES customers,
    vehicle_id                text NOT NULL REFERENCES vehicles,
    wording_id                text NOT NULL,
    state                     char(2) NOT NULL,
    start_date                date NOT NULL,
    end_date                  date NOT NULL,
    status                    text NOT NULL CHECK (status IN ('active', 'lapsed', 'cancelled')),
    cancelled_date            date,
    liability                 boolean NOT NULL,
    collision                 boolean NOT NULL,
    comprehensive             boolean NOT NULL,
    full_glass                boolean NOT NULL,
    rental_reimbursement      boolean NOT NULL,
    collision_deductible      int,
    comprehensive_deductible  int,
    annual_premium            int NOT NULL
);
CREATE INDEX IF NOT EXISTS policies_customer_idx ON policies (customer_id);

CREATE TABLE IF NOT EXISTS claims (
    claim_id        text PRIMARY KEY,
    policy_id       text NOT NULL REFERENCES policies,
    customer_id     text NOT NULL REFERENCES customers,
    vehicle_id      text NOT NULL REFERENCES vehicles,
    peril           text NOT NULL CHECK (peril IN ('collision', 'comprehensive')),
    description     text NOT NULL,
    damages         jsonb NOT NULL,  -- [{"class": "dent", "severity": "minor"}, ...]
    incident_date   date NOT NULL,
    reported_date   date NOT NULL,
    claimed_amount  numeric(10, 2) NOT NULL,
    deductible      int NOT NULL,
    status          text NOT NULL,
    paid_amount     numeric(10, 2) NOT NULL
);
CREATE INDEX IF NOT EXISTS claims_customer_idx ON claims (customer_id);
CREATE INDEX IF NOT EXISTS claims_policy_idx ON claims (policy_id);

-- One row per photo. The 64-bit perceptual hash is stored as a signed bigint
-- so the duplicate check runs in SQL: bit_count((phash # $1)::bit(64)) is the
-- Hamming distance. A sequential scan is fine at this size; at millions of
-- photos you'd use multi-index hashing or a BK-tree.
CREATE TABLE IF NOT EXISTS claim_photos (
    claim_id  text NOT NULL REFERENCES claims ON DELETE CASCADE,
    position  int NOT NULL,
    phash     bigint NOT NULL,
    PRIMARY KEY (claim_id, position)
);

-- Pricing (all USD).
CREATE TABLE IF NOT EXISTS labour_rates (
    state          char(2) PRIMARY KEY,
    rate_per_hour  numeric(8, 2) NOT NULL
);
CREATE TABLE IF NOT EXISTS segment_multipliers (
    segment           text PRIMARY KEY,
    parts_multiplier  numeric(4, 2) NOT NULL
);
CREATE TABLE IF NOT EXISTS repair_operations (
    damage_class  text NOT NULL,
    severity      text NOT NULL,
    operation     text NOT NULL,
    parts_usd     numeric(8, 2) NOT NULL,
    labour_hours  numeric(4, 1) NOT NULL,
    paint_hours   numeric(4, 1) NOT NULL,
    PRIMARY KEY (damage_class, severity)
);
CREATE TABLE IF NOT EXISTS pricing_settings (
    key    text PRIMARY KEY,
    value  numeric NOT NULL
);

-- Policy wordings for RAG.
CREATE TABLE IF NOT EXISTS policy_wordings (
    wording_id   text PRIMARY KEY,
    title        text NOT NULL,
    insurer      text NOT NULL,
    form_number  text NOT NULL,
    source_url   text NOT NULL,
    sha256       char(64) NOT NULL,
    pages        int NOT NULL
);

-- Vector search only: a full-text column for hybrid search was tried and
-- removed after measuring (see mcp_servers/policy/test_retrieval_eval.py).
CREATE TABLE IF NOT EXISTS policy_chunks (
    wording_id   text NOT NULL REFERENCES policy_wordings ON DELETE CASCADE,
    chunk_index  int NOT NULL,
    section      text NOT NULL,
    page_start   int NOT NULL,
    page_end     int NOT NULL,
    content      text NOT NULL,
    embedding    vector(384) NOT NULL,  -- BAAI/bge-small-en-v1.5
    PRIMARY KEY (wording_id, chunk_index)
);
-- HNSW: approximate nearest neighbours, fast at any size. At 221 chunks a
-- sequential scan would do, but this is what scales.
CREATE INDEX IF NOT EXISTS policy_chunks_embedding_idx
    ON policy_chunks USING hnsw (embedding vector_cosine_ops);

-- Ground truth for evaluation only. A separate schema so services can be
-- granted the public schema without ever seeing the labels (Phase 4 adds
-- per-service database roles).
CREATE SCHEMA IF NOT EXISTS eval;
CREATE TABLE IF NOT EXISTS eval.fraud_labels (
    claim_id  text PRIMARY KEY REFERENCES claims ON DELETE CASCADE,
    is_fraud  boolean NOT NULL,
    patterns  text[] NOT NULL
);

-- Logins, read by the gateway. Their own schema, so Phase 4's per-service
-- roles can give the gateway this and read access to policies, nothing more.
-- No foreign key to customers on purpose: load_synthetic.py reloads with
-- TRUNCATE ... CASCADE, which empties every referencing table, so an FK would
-- delete the adjuster accounts on every data reload. A customer login whose
-- customer is gone simply sees no policies.
CREATE SCHEMA IF NOT EXISTS auth;
CREATE TABLE IF NOT EXISTS auth.users (
    username       text PRIMARY KEY,  -- lower case; customers log in with their email
    password_hash  text NOT NULL,     -- scrypt, format in services/gateway/passwords.py
    role           text NOT NULL CHECK (role IN ('customer', 'adjuster')),
    display_name   text NOT NULL,
    customer_id    text,
    CHECK ((role = 'customer') = (customer_id IS NOT NULL))
);
