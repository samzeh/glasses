-- Glasses data model. Postgres + TimescaleDB + pgvector — one database.
--
-- Every table carries provenance — which encounter produced this row — because
-- that is what makes merge, split, deletion and debugging possible. The schema
-- looks over-normalized and that is the point: split needs per-embedding and
-- per-fact provenance, and there is no way to add it after the fact.
--
-- Two shapes of data live here, and the reason they live together is that the
-- join between them is the product. The relational half is the profile: who a
-- person is, which encounters they appeared in, what was learned about them.
-- The stream half is what the sensors saw microsecond by microsecond —
-- `face_observation` and `speech_observation`, hypertables at the bottom of
-- this file. Answering "who was in front of me while this sentence was
-- spoken" is a join across the two, and holding them in separate systems
-- turns that join into application code that reconciles two clocks.

CREATE EXTENSION IF NOT EXISTS timescaledb;
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
    -- 512-d for WeSpeaker CAM++. Note this is the same width as
    -- face_template: nothing but model_name keeps the two apart, which
    -- is why every search filters on it.
    embedding           vector(512) NOT NULL,
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

-- ---------------------------------------------------------------------------
-- Observation streams. Hypertables.
--
-- Everything above is a profile: a few rows per person, read constantly,
-- written rarely. These two are the opposite — append-only, ordered by time,
-- never updated, and the volume grows with wear time rather than with the
-- number of people. One frame per second per visible face and one row per
-- speech segment is already millions of rows a year from a single device.
--
-- They are a hypertable each rather than a plain table because the queries
-- are all "this encounter, this time range": chunk exclusion turns those into
-- a scan of one chunk instead of the whole stream, and chunks are also the
-- unit that compression and retention operate on. Plain Postgres partitioning
-- would give the first of those three and none of the rest.
--
-- Two clocks on every row, deliberately:
--
--   * `*_device_clock_us` is the ESP's free-running monotonic counter, and it
--     is the authority. Binding a voice to a face is interval arithmetic on
--     these integers (see glasses.encounter.types.Interval), and it must not
--     be done on wall time: an NTP step mid-encounter would reorder two
--     observations and attach a sentence to the wrong face.
--   * `observed_at` is that same instant mapped through the clock's single
--     wall anchor. It exists because it is what humans, time_bucket and
--     chunk exclusion need. It is derived, never read back for binding.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS face_observation (
    encounter_id          uuid NOT NULL REFERENCES encounter (id) ON DELETE CASCADE,
    observed_at           timestamptz NOT NULL,
    start_device_clock_us bigint NOT NULL,
    end_device_clock_us   bigint NOT NULL,
    -- Null for a stranger, which is most faces. The observation is still
    -- worth storing: the trigger counts detections rather than identities,
    -- and an unidentified face is what makes a binding ambiguous.
    person_id             uuid REFERENCES person (id) ON DELETE SET NULL,
    confidence            real NOT NULL,
    -- Frames in the span. Kept because a face present at t=10, 11 and 12 is
    -- one observation, and dwell counts detections, not spans.
    frame_count           integer NOT NULL DEFAULT 1,
    is_primary            boolean NOT NULL DEFAULT true,
    centred               boolean NOT NULL DEFAULT true,
    CONSTRAINT face_observation_span_ordered
        CHECK (end_device_clock_us >= start_device_clock_us)
);

CREATE TABLE IF NOT EXISTS speech_observation (
    encounter_id          uuid NOT NULL REFERENCES encounter (id) ON DELETE CASCADE,
    observed_at           timestamptz NOT NULL,
    start_device_clock_us bigint NOT NULL,
    end_device_clock_us   bigint NOT NULL,
    person_id             uuid REFERENCES person (id) ON DELETE SET NULL,
    -- The anonymous within-recording diarization label. Survives when the
    -- identity does not, which is what lets a later encounter name a voice
    -- retrospectively.
    speaker               integer,
    confidence            real NOT NULL,
    -- The wearer's own speech. Separated at the row level because every
    -- downstream question is about partner speech: the wearer's voice is the
    -- loudest thing the mic hears and the biggest source of facts being
    -- filed against the wrong person.
    is_wearer             boolean NOT NULL DEFAULT false,
    text                  text,
    CONSTRAINT speech_observation_span_ordered
        CHECK (end_device_clock_us >= start_device_clock_us)
);

-- One day per chunk. At this volume that is a few megabytes a chunk, which is
-- the size compression wants and small enough that a day-range query touches
-- one of them.
SELECT create_hypertable(
    'face_observation',
    by_range('observed_at', INTERVAL '1 day'),
    if_not_exists => true
);
SELECT create_hypertable(
    'speech_observation',
    by_range('observed_at', INTERVAL '1 day'),
    if_not_exists => true
);

-- Encounter-scoped lookups are the hot path; the time dimension is already
-- indexed by the hypertable itself.
CREATE INDEX IF NOT EXISTS face_observation_encounter_idx
    ON face_observation (encounter_id, observed_at DESC);
CREATE INDEX IF NOT EXISTS face_observation_person_idx
    ON face_observation (person_id, observed_at DESC) WHERE person_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS speech_observation_encounter_idx
    ON speech_observation (encounter_id, observed_at DESC);
CREATE INDEX IF NOT EXISTS speech_observation_person_idx
    ON speech_observation (person_id, observed_at DESC) WHERE person_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Continuous aggregates: the live view.
--
-- The demo question is "what is happening right now, per minute" — how much
-- was seen, how many distinct people, how much of the talking was theirs.
-- Asked against the raw stream that is a scan that grows without bound; asked
-- against these it is a handful of rows, and the work is done incrementally as
-- the data arrives rather than at query time.
--
-- `materialized_only = false` is set on purpose. The default since
-- TimescaleDB 2.13 hides buckets the refresh job has not reached yet, which
-- for a one-minute bucket means the current minute — exactly the minute a live
-- demo is looking at. With real-time aggregation on, the view unions the
-- materialized history with a fresh scan of the newest chunk.
--
-- Note what is *not* in here: count(DISTINCT person_id). Continuous
-- aggregates cannot hold a DISTINCT aggregate, because incremental
-- materialization needs a partial state that merges, and a distinct count's
-- does not. So the aggregate groups by person_id — one row per person per
-- minute — and the distinct count is taken one level up in
-- `encounter_minute`, over rows the aggregate has already collapsed.
-- ---------------------------------------------------------------------------

CREATE MATERIALIZED VIEW IF NOT EXISTS face_activity_per_minute
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket(INTERVAL '1 minute', observed_at) AS bucket,
    encounter_id,
    person_id,
    count(*)                                        AS observations,
    sum(frame_count)                                AS frames,
    avg(confidence)                                 AS mean_confidence,
    max(confidence)                                 AS peak_confidence,
    sum(end_device_clock_us - start_device_clock_us) AS visible_us
FROM face_observation
GROUP BY bucket, encounter_id, person_id
WITH NO DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS speech_activity_per_minute
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT
    time_bucket(INTERVAL '1 minute', observed_at) AS bucket,
    encounter_id,
    person_id,
    is_wearer,
    count(*)                                        AS segments,
    avg(confidence)                                 AS mean_confidence,
    sum(end_device_clock_us - start_device_clock_us) AS speech_us
FROM speech_observation
GROUP BY bucket, encounter_id, person_id, is_wearer
WITH NO DATA;

-- Refresh the last two hours every minute, leaving the two most recent
-- buckets to real-time aggregation. start_offset bounds the work: without it
-- a refresh would reconsider the whole stream.
SELECT add_continuous_aggregate_policy(
    'face_activity_per_minute',
    start_offset => INTERVAL '2 hours',
    end_offset   => INTERVAL '2 minutes',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => true
);
SELECT add_continuous_aggregate_policy(
    'speech_activity_per_minute',
    start_offset => INTERVAL '2 hours',
    end_offset   => INTERVAL '2 minutes',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => true
);

-- The view a live demo actually renders: both sensors, one row per encounter
-- per minute. Each side is aggregated before the join, because joining the
-- two per-person aggregates directly would multiply faces by voices.
CREATE OR REPLACE VIEW encounter_minute AS
WITH faces AS (
    SELECT
        bucket,
        encounter_id,
        sum(observations)                                 AS face_observations,
        -- One row per person per minute already, so counting the non-null
        -- person_ids is the distinct identified-face count. Strangers arrive
        -- in the person_id IS NULL row and are counted as observations only.
        count(person_id)                                  AS distinct_faces,
        sum(visible_us) FILTER (WHERE person_id IS NULL)  AS stranger_visible_us,
        sum(visible_us)                                   AS face_visible_us
    FROM face_activity_per_minute
    GROUP BY bucket, encounter_id
),
voices AS (
    SELECT
        bucket,
        encounter_id,
        sum(segments)                                  AS speech_segments,
        count(DISTINCT person_id)                      AS distinct_voices,
        sum(speech_us) FILTER (WHERE NOT is_wearer)    AS partner_speech_us,
        sum(speech_us) FILTER (WHERE is_wearer)        AS wearer_speech_us
    FROM speech_activity_per_minute
    GROUP BY bucket, encounter_id
)
SELECT
    coalesce(f.bucket, v.bucket)               AS bucket,
    coalesce(f.encounter_id, v.encounter_id)   AS encounter_id,
    coalesce(f.face_observations, 0)           AS face_observations,
    coalesce(f.distinct_faces, 0)              AS distinct_faces,
    coalesce(f.face_visible_us, 0) / 1e6       AS face_visible_s,
    coalesce(f.stranger_visible_us, 0) / 1e6   AS stranger_visible_s,
    coalesce(v.speech_segments, 0)             AS speech_segments,
    coalesce(v.distinct_voices, 0)             AS distinct_voices,
    coalesce(v.partner_speech_us, 0) / 1e6     AS partner_speech_s,
    coalesce(v.wearer_speech_us, 0) / 1e6      AS wearer_speech_s
FROM faces f
FULL OUTER JOIN voices v
    ON f.bucket = v.bucket AND f.encounter_id = v.encounter_id;

-- ---------------------------------------------------------------------------
-- Compression and retention.
--
-- Both are policies on the hypertable rather than cron jobs somewhere else,
-- which is the only version of either that stays true. Segmenting by
-- encounter_id matters: it is the column every query filters on, and a
-- compressed batch is only skippable if the filter can be evaluated without
-- decompressing it.
--
-- Nothing here compresses during a demo, because nothing in a demo is seven
-- days old. To see it work, compress by hand:
--
--   SELECT compress_chunk(c) FROM show_chunks('face_observation') c;
--
-- `glasses db stats` then reports the before/after.
--
-- Retention is 90 days on the raw stream while the continuous aggregates are
-- kept indefinitely: per-minute history is what the profile needs, and the
-- microsecond stream behind it stops being evidence anyone will re-examine
-- long before it stops costing storage. The deletion path for a person's data
-- is separate and immediate; this is the floor, not the policy.
-- ---------------------------------------------------------------------------

ALTER TABLE face_observation SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'encounter_id',
    timescaledb.compress_orderby = 'observed_at DESC'
);
ALTER TABLE speech_observation SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'encounter_id',
    timescaledb.compress_orderby = 'observed_at DESC'
);

SELECT add_compression_policy('face_observation', INTERVAL '7 days', if_not_exists => true);
SELECT add_compression_policy('speech_observation', INTERVAL '7 days', if_not_exists => true);

SELECT add_retention_policy('face_observation', INTERVAL '90 days', if_not_exists => true);
SELECT add_retention_policy('speech_observation', INTERVAL '90 days', if_not_exists => true);
