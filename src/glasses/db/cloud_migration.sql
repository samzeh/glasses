-- Additive upgrade for databases initialized before cloud enrollment.
ALTER TABLE face_template ADD COLUMN IF NOT EXISTS source text NOT NULL DEFAULT '';
ALTER TABLE voice_template ADD COLUMN IF NOT EXISTS source text NOT NULL DEFAULT '';
ALTER TABLE voice_template ADD COLUMN IF NOT EXISTS quality_score real NOT NULL DEFAULT 0;
-- A wearer can have multiple voiceprints. PgStore serializes wearer enrollment
-- and rejects a second wearer identity, rather than limiting it to one vector.
DROP INDEX IF EXISTS voice_template_one_wearer_idx;
CREATE INDEX IF NOT EXISTS voice_template_wearer_idx
    ON voice_template (is_wearer) WHERE is_wearer AND deleted_at IS NULL;

-- Original local files are archived byte-for-byte before local cleanup.
CREATE TABLE IF NOT EXISTS local_import_file (
    source_path text PRIMARY KEY,
    sha256 text NOT NULL,
    content bytea NOT NULL,
    person_ids uuid[] NOT NULL DEFAULT '{}',
    imported_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE local_import_file ADD COLUMN IF NOT EXISTS person_ids uuid[] NOT NULL DEFAULT '{}';
-- Older imports archived every file in a single transaction. Link each batch
-- to the people listed in its archived index so `forget` also erases originals.
UPDATE local_import_file AS target
SET person_ids = batch.person_ids
FROM (
    SELECT imported_at, array_agg(DISTINCT (p->>'id')::uuid) AS person_ids
    FROM local_import_file
    CROSS JOIN LATERAL jsonb_array_elements(
        CASE WHEN source_path LIKE '%/index.json'
             THEN convert_from(content, 'UTF8')::jsonb->'people'
             ELSE '[]'::jsonb END
    ) AS p
    GROUP BY imported_at
) AS batch
WHERE target.imported_at = batch.imported_at AND cardinality(target.person_ids) = 0;
