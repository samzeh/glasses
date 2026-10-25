import pg from 'pg';

const { Pool } = pg;
const connectionString = process.env.DATABASE_URL?.trim();
if (!connectionString) {
  throw new Error('DATABASE_URL is required to save voice notes in TigerData.');
}

const databaseUrl = new URL(connectionString);
// The glasses project's TigerData URL uses libpq's sslmode=require. Match its
// TLS semantics here; a supplied sslrootcert still enables CA verification.
if (databaseUrl.searchParams.get('sslmode') === 'require' &&
    !databaseUrl.searchParams.has('uselibpqcompat')) {
  databaseUrl.searchParams.set('uselibpqcompat', 'true');
}

export const pool = new Pool({
  connectionString: databaseUrl.toString(), max: 3, connectionTimeoutMillis: 5000,
});
pool.on('error', (error) => console.error('[database] idle connection error:', error.message));

export async function checkDatabase() {
  await pool.query('SELECT encounter_id FROM public.voice_note LIMIT 0');
}

// Postgres accepts a uuid with or without its hyphens, and the Python hub
// generates the unhyphenated form. Checked here so a junk id becomes "no id
// supplied" rather than a thrown cast that loses the note.
const UUID = /^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$/i;

async function existingEncounter(client, encounterId) {
  if (typeof encounterId !== 'string' || !UUID.test(encounterId.trim())) return null;
  const found = await client.query(
    'SELECT id FROM public.encounter WHERE id = $1',
    [encounterId.trim()],
  );
  return found.rowCount ? found.rows[0].id : null;
}

// `encounterId` links the note to an encounter the hub already recorded, so one
// conversation is one row rather than a video encounter and an audio encounter
// that nothing joins. An id that does not exist falls back to creating one:
// a note that arrives before, or without, its encounter is still a note.
export async function saveVoiceNote({
  transcript, summary, audio, mimeType, durationMs, model, keepAudio, encounterId = null,
}) {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    let id = await existingEncounter(client, encounterId);
    const linked = Boolean(id);
    if (!id) {
      const endedAt = new Date();
      const startedAt = new Date(endedAt.getTime() - (durationMs ?? 0));
      const encounter = await client.query(
        `INSERT INTO public.encounter (device_id, started_at, ended_at, status)
         VALUES ($1, $2, $3, 'confirmed') RETURNING id`,
        ['voice-notes-web', startedAt, endedAt],
      );
      id = encounter.rows[0].id;
    }
    // One note per encounter: the table says so, and a resubmit after a
    // network failure should replace the note rather than fail the upload.
    await client.query(
      `INSERT INTO public.voice_note
       (encounter_id, transcript, summary, audio_mime_type, audio_data, audio_bytes, duration_ms, model)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
       ON CONFLICT (encounter_id) DO UPDATE SET
         transcript = EXCLUDED.transcript, summary = EXCLUDED.summary,
         audio_mime_type = EXCLUDED.audio_mime_type, audio_data = EXCLUDED.audio_data,
         audio_bytes = EXCLUDED.audio_bytes, duration_ms = EXCLUDED.duration_ms,
         model = EXCLUDED.model`,
      [id, transcript, summary, mimeType, keepAudio ? audio : null,
        audio.length, durationMs, model],
    );
    await client.query('COMMIT');
    return { encounterId: id, linked };
  } catch (error) {
    await client.query('ROLLBACK');
    throw error;
  } finally {
    client.release();
  }
}
