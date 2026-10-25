-- Voice Notes uses the same encounter table as the glasses application.
-- Run the glasses `db init` command first on a new database.
CREATE TABLE IF NOT EXISTS public.voice_note (
    encounter_id    uuid PRIMARY KEY REFERENCES public.encounter(id) ON DELETE CASCADE,
    transcript      text NOT NULL,
    summary         text NOT NULL,
    audio_mime_type text NOT NULL,
    audio_data      bytea,
    audio_bytes     integer NOT NULL CHECK (audio_bytes >= 0),
    duration_ms     integer CHECK (duration_ms >= 0),
    model           text NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT voice_note_audio_size_matches CHECK (
        audio_data IS NULL OR octet_length(audio_data) = audio_bytes
    )
);

CREATE INDEX IF NOT EXISTS voice_note_created_at_idx
    ON public.voice_note (created_at DESC);
