-- Glasses data model. Postgres + pgvector.
--
-- Every table carries provenance — which encounter produced this row — because
-- that is what makes merge, split, deletion and debugging possible. The schema
-- looks over-normalized and that is the point: split needs per-embedding and
-- per-fact provenance, and there is no way to add it after the fact.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ---------------------------------------------------------------------------
-- Encounters: the unit everything else hangs off.
-- ---------------------------------------------------------------------------

CREATE TYPE encounter_status AS ENUM ('provisional', 'confirmed', 'expired');

CREATE TABLE encounter (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id   text NOT NULL,
    started_at  timestamptz NOT NULL,
    ended_at    timestamptz,
    status      encounter_status NOT NULL DEFAULT 'provisional',
    location    text,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX encounter_started_at_idx ON encounter (started_at DESC);
CREATE INDEX encounter_status_idx ON encounter (status) WHERE status = 'provisional';

-- ---------------------------------------------------------------------------
-- People.
-- ---------------------------------------------------------------------------

CREATE TYPE person_status AS ENUM ('provisional', 'confirmed', 'merged');

CREATE TABLE person (
    id                      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Null until a name is actually heard. A person is useful before they are
    -- named; forcing a placeholder here makes the empty case invisible.
    display_name            text,
    status                  person_status NOT NULL DEFAULT 'provisional',
    confidence              real NOT NULL DEFAULT 0.0,
    created_from_encounter  uuid REFERENCES encounter (id) ON DELETE SET NULL,
    -- Consent is recorded here, not assumed. v1 enrolls only people who have
    -- explicitly agreed; see the privacy section of the PRD.
    consented_at            timestamptz,
    consent_note            text,
    created_at              timestamptz NOT NULL DEFAULT now(),
    updated_at              timestamptz NOT NULL DEFAULT now(),
    deleted_at              timestamptz
);

CREATE INDEX person_status_idx ON person (status) WHERE deleted_at IS NULL;

-- ---------------------------------------------------------------------------
-- Biometric templates.
-- ---------------------------------------------------------------------------

CREATE TABLE face_template (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    person_id           uuid NOT NULL REFERENCES person (id) ON DELETE CASCADE,
    embedding           vector(512) NOT NULL,
    -- model_name + model_version are what let the gallery be re-embedded
    -- without being lost. Every query filters on both: without that, a model
    -- swap silently mixes incomparable vectors in one index and recognition
    -- quietly collapses.
    model_name          text NOT NULL,
    model_version       text NOT NULL,
    quality_score       real NOT NULL,
    source_encounter_id uuid REFERENCES encounter (id) ON DELETE SET NULL,
    created_at          timestamptz NOT NULL DEFAULT now(),
    deleted_at          timestamptz
);

-- Cosine, matching the `<=>` operator the gallery search uses.
CREATE INDEX face_template_embedding_idx
    ON face_template USING hnsw (embedding vector_cosine_ops);
CREATE INDEX face_template_person_idx ON face_template (person_id);
CREATE INDEX face_template_model_idx ON face_template (model_name, model_version);

CREATE TABLE voice_template (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    person_id           uuid NOT NULL REFERENCES person (id) ON DELETE CASCADE,
    embedding           vector(192) NOT NULL,  -- ECAPA-TDNN
    model_name          text NOT NULL,
    model_version       text NOT NULL,
    source_encounter_id uuid REFERENCES encounter (id) ON DELETE SET NULL,
    -- The wearer gets exactly one of these, enrolled once at setup. It turns
    -- diarization from a hard problem into a binary one.
    is_wearer           boolean NOT NULL DEFAULT false,
    created_at          timestamptz NOT NULL DEFAULT now(),
    deleted_at          timestamptz
);

CREATE UNIQUE INDEX voice_template_one_wearer_idx
    ON voice_template (is_wearer) WHERE is_wearer AND deleted_at IS NULL;
CREATE INDEX voice_template_embedding_idx
    ON voice_template USING hnsw (embedding vector_cosine_ops);
CREATE INDEX voice_template_person_idx ON voice_template (person_id);

-- ---------------------------------------------------------------------------
-- Who was in which encounter. The voice-to-face binding lives here as a
-- weighted hypothesis, not on `person` as a fact: each encounter that agrees
-- strengthens it, and a contradiction reopens it.
-- ---------------------------------------------------------------------------

CREATE TYPE binding_method AS ENUM (
    'temporal_cooccurrence',  -- exactly one non-wearer face visible while they spoke
    'voiceprint',             -- matched an already-bound voiceprint
    'face_match',             -- matched the gallery, no voice involved
    'manual'                  -- a human confirmed or corrected it
);

CREATE TABLE encounter_person (
    encounter_id        uuid NOT NULL REFERENCES encounter (id) ON DELETE CASCADE,
    person_id           uuid NOT NULL REFERENCES person (id) ON DELETE CASCADE,
    binding_confidence  real NOT NULL,
    binding_method      binding_method NOT NULL,
    created_at          timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (encounter_id, person_id)
);

CREATE INDEX encounter_person_person_idx ON encounter_person (person_id);

-- ---------------------------------------------------------------------------
-- Raw media. The highest-risk data in the system.
-- ---------------------------------------------------------------------------

CREATE TYPE media_kind AS ENUM ('frame', 'audio');

CREATE TABLE media_asset (
    id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    encounter_id         uuid NOT NULL REFERENCES encounter (id) ON DELETE CASCADE,
    kind                 media_kind NOT NULL,
    storage_key          text NOT NULL,
    captured_at          timestamptz NOT NULL,
    -- Microseconds from the ESP's monotonic clock, synced once against SNTP at
    -- boot. This is what binds a voice to a face: which face was visible while
    -- a particular sentence was spoken. A per-stream or drifting clock
    -- destroys that alignment silently.
    device_clock_us      bigint NOT NULL,
    quality_score        real,
    -- Enforced by a deletion job, not by good intentions. A column plus a job
    -- is the only version of a retention policy that actually exists.
    retention_expires_at timestamptz NOT NULL,
    deleted_at           timestamptz
);

CREATE INDEX media_asset_encounter_idx ON media_asset (encounter_id);
CREATE INDEX media_asset_retention_idx
    ON media_asset (retention_expires_at) WHERE deleted_at IS NULL;

-- ---------------------------------------------------------------------------
-- Facts.
-- ---------------------------------------------------------------------------

CREATE TABLE fact (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Nullable on purpose. When two or more non-wearer faces are visible at
    -- once, the facts attach to the encounter and wait. Guessing here is how a
    -- profile ends up asserting that your colleague has three children who
    -- actually belong to someone they were standing next to.
    person_id     uuid REFERENCES person (id) ON DELETE CASCADE,
    encounter_id  uuid NOT NULL REFERENCES encounter (id) ON DELETE CASCADE,
    kind          text NOT NULL,  -- name, employer, role, commitment, interest, ...
    value         text NOT NULL,
    -- The transcript span this came from. A fact with no evidence is a fact
    -- you cannot debug, and this is what makes a wrong profile correctable.
    evidence_span text,
    evidence_start_ms integer,
    evidence_end_ms   integer,
    confidence    real NOT NULL,
    extracted_at  timestamptz NOT NULL DEFAULT now(),
    superseded_by uuid REFERENCES fact (id) ON DELETE SET NULL
);

CREATE INDEX fact_person_idx ON fact (person_id) WHERE superseded_by IS NULL;
CREATE INDEX fact_encounter_idx ON fact (encounter_id);
CREATE INDEX fact_unassigned_idx ON fact (encounter_id) WHERE person_id IS NULL;

-- ---------------------------------------------------------------------------
-- Merge log. Beyond the seven tables in the PRD, and required by it: merge
-- "records the merge so it can be undone". Built now rather than as a
-- migration script written in a panic later.
-- ---------------------------------------------------------------------------

CREATE TABLE identity_merge (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    kept_id       uuid NOT NULL REFERENCES person (id) ON DELETE CASCADE,
    merged_id     uuid NOT NULL REFERENCES person (id) ON DELETE CASCADE,
    -- Every row reassigned by the merge, so the split can put them back.
    moved_face_templates  uuid[] NOT NULL DEFAULT '{}',
    moved_voice_templates uuid[] NOT NULL DEFAULT '{}',
    moved_facts           uuid[] NOT NULL DEFAULT '{}',
    reason        text,
    merged_at     timestamptz NOT NULL DEFAULT now(),
    undone_at     timestamptz
);

CREATE INDEX identity_merge_kept_idx ON identity_merge (kept_id) WHERE undone_at IS NULL;
