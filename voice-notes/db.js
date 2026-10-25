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

export async function saveVoiceNote({ transcript, summary, audio, mimeType, durationMs, model, keepAudio }) {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    const endedAt = new Date();
    const startedAt = new Date(endedAt.getTime() - (durationMs ?? 0));
    const encounter = await client.query(
      `INSERT INTO public.encounter (device_id, started_at, ended_at, status)
       VALUES ($1, $2, $3, 'confirmed') RETURNING id`,
      ['voice-notes-web', startedAt, endedAt],
    );
    const encounterId = encounter.rows[0].id;
    await client.query(
      `INSERT INTO public.voice_note
       (encounter_id, transcript, summary, audio_mime_type, audio_data, audio_bytes, duration_ms, model)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`,
      [encounterId, transcript, summary, mimeType, keepAudio ? audio : null,
        audio.length, durationMs, model],
    );
    await client.query('COMMIT');
    return encounterId;
  } catch (error) {
    await client.query('ROLLBACK');
    throw error;
  } finally {
    client.release();
  }
}
