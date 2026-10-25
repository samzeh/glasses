// Holds the Gemini API key, saves each note in TigerData, and serves the front end.
//   POST /api/summarize  { mimeType, audio: <base64>, durationMs, encounterId? }
//                                                      -> { summary, transcript, encounterId }
//   POST /api/speak      { text }                      -> { audio: <base64 wav> }
//   POST /api/vad        { event: 'start'|'end' }       -> the hub's trigger state
//   GET  /api/trigger                                   -> the hub's trigger state
// Run: npm run db:init, then npm run dev (both keys in .env).
//
// The last two exist because the phone cannot reach the Python hub directly.
// Its page is served over HTTPS through a tunnel, and a browser on an HTTPS
// origin may not call plain HTTP — so the phone talks to this server on its
// own origin and this server forwards over loopback.

import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { checkDatabase, pool, saveVoiceNote } from './db.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT ?? 8787);
const KEY = process.env.GEMINI_API_KEY ?? '';
const KEEP_AUDIO = process.env.VOICE_NOTES_STORE_AUDIO !== 'false';

const ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/interactions';
// Overridable so a model rename does not require a code edit.
const TEXT_MODEL = process.env.GEMINI_TEXT_MODEL ?? 'gemini-3.8-flash';
const TTS_MODEL = process.env.GEMINI_TTS_MODEL ?? 'gemini-3.8-flash-tts';
const TTS_VOICE = process.env.GEMINI_TTS_VOICE ?? 'Kore';

// Inline audio must fit the 20MB whole-request limit. Base64 inflates by 4/3,
// so cap the decoded audio well under that.
const MAX_BODY_BYTES = 14 * 1024 * 1024;

// Where auto_hub.py's trigger bridge listens. Loopback: it has no
// authentication and must not be reachable off this machine.
const TRIGGER_URL = (process.env.TRIGGER_URL ?? 'http://127.0.0.1:8788').replace(/\/$/, '');
// The phone polls this. A hub that is slow or gone must not stall the page,
// so the timeout is short and a failure answers "unavailable".
const TRIGGER_TIMEOUT_MS = 2000;

async function callTrigger(path, init) {
  const response = await fetch(`${TRIGGER_URL}${path}`, {
    ...init,
    signal: AbortSignal.timeout(TRIGGER_TIMEOUT_MS),
  });
  if (!response.ok) throw new Error(`trigger ${response.status}`);
  return response.json();
}

// `available: false` is the phone's cue to stay idle rather than record into
// a void: without the hub there is no camera, so nothing can be confirmed.
async function handleTrigger(res) {
  try {
    return sendJson(res, 200, { available: true, ...(await callTrigger('/state')) });
  } catch (error) {
    return sendJson(res, 200, { available: false, reason: String(error.message ?? error) });
  }
}

async function handleVad(req, res) {
  const { event } = await readJsonBody(req);
  if (event !== 'start' && event !== 'end') {
    return sendJson(res, 400, { error: "event must be 'start' or 'end'" });
  }
  try {
    const state = await callTrigger('/speech', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ event }),
    });
    return sendJson(res, 200, { available: true, ...state });
  } catch (error) {
    return sendJson(res, 200, { available: false, reason: String(error.message ?? error) });
  }
}

const PROMPT = `You are summarizing a voice recording.

Return ONLY a JSON object, no markdown fences, with exactly these two keys:
"transcript": a verbatim transcript of the speech in the audio.
"summary": a concise summary of 3 to 6 sentences, written to be READ ALOUD. Use plain spoken sentences. No bullet points, no headings, no markdown, no parentheses, no abbreviations, no emoji.

If the audio contains no intelligible speech, set "summary" to "I could not hear any speech in that recording." and set "transcript" to "".`;

// Gemini accepts a bare audio MIME type. Browsers hand us codec parameters,
// and Safari reports audio/mp4, which the API knows as audio/m4a.
function normalizeMime(raw) {
  const base = String(raw || '').split(';')[0].trim().toLowerCase();
  if (base === 'audio/mp4' || base === 'video/mp4') return 'audio/m4a';
  if (base === 'audio/x-m4a') return 'audio/m4a';
  const allowed = new Set([
    'audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/aiff', 'audio/aac',
    'audio/ogg', 'audio/flac', 'audio/m4a', 'audio/opus', 'audio/webm',
  ]);
  return allowed.has(base) ? base : 'audio/webm';
}

async function callGemini(body) {
  const res = await fetch(ENDPOINT, {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-goog-api-key': KEY },
    body: JSON.stringify(body),
  });
  const raw = await res.text();
  if (!res.ok) {
    const err = new Error(`Gemini ${res.status}: ${raw.slice(0, 600)}`);
    err.status = res.status;
    throw err;
  }
  try {
    return JSON.parse(raw);
  } catch {
    throw new Error(`Gemini returned non-JSON: ${raw.slice(0, 300)}`);
  }
}

// Walk every step/content block rather than assuming a fixed index, so a
// thinking or tool step in front of the answer does not break extraction.
function collectText(payload) {
  let out = '';
  for (const step of payload?.steps ?? []) {
    for (const block of step?.content ?? []) {
      if (block?.type === 'text' && typeof block.text === 'string') out += block.text;
    }
  }
  return out.trim();
}

function collectAudio(payload) {
  if (typeof payload?.output_audio?.data === 'string') return payload.output_audio.data;
  for (const step of payload?.steps ?? []) {
    for (const block of step?.content ?? []) {
      const looksAudio = block?.type === 'audio' || String(block?.mime_type ?? '').startsWith('audio/');
      if (looksAudio && typeof block.data === 'string') return block.data;
    }
  }
  return null;
}

// The model is asked for bare JSON but may still fence it. Never let a parse
// failure lose the content: fall back to speaking whatever came back.
function parseSummary(text) {
  const cleaned = text.replace(/^\s*```(?:json)?/i, '').replace(/```\s*$/, '').trim();
  const start = cleaned.indexOf('{');
  const end = cleaned.lastIndexOf('}');
  if (start !== -1 && end > start) {
    try {
      const obj = JSON.parse(cleaned.slice(start, end + 1));
      const summary = typeof obj.summary === 'string' ? obj.summary.trim() : '';
      const transcript = typeof obj.transcript === 'string' ? obj.transcript.trim() : '';
      if (summary) return { summary, transcript };
    } catch { /* fall through */ }
  }
  return { summary: cleaned, transcript: '' };
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(Object.assign(new Error('Recording too large. Keep it under about ten minutes.'), { status: 413 }));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString('utf8')));
      } catch {
        reject(Object.assign(new Error('Malformed request body.'), { status: 400 }));
      }
    });
    req.on('error', reject);
  });
}

function sendJson(res, status, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(body),
    'cache-control': 'no-store',
  });
  res.end(body);
}

async function handleSummarize(req, res) {
  const { audio, mimeType, durationMs, encounterId } = await readJsonBody(req);
  if (typeof audio !== 'string' || audio.length < 100 ||
      !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(audio)) {
    return sendJson(res, 400, { error: 'No audio received.' });
  }
  const recording = Buffer.from(audio, 'base64');
  const recordingDuration = Number.isInteger(durationMs) && durationMs >= 0 && durationMs <= 300_000
    ? durationMs : null;
  const normalizedMime = normalizeMime(mimeType);
  const payload = await callGemini({
    model: TEXT_MODEL,
    input: [
      { type: 'text', text: PROMPT },
      { type: 'audio', data: audio, mime_type: normalizedMime },
    ],
  });
  const text = collectText(payload);
  if (!text) return sendJson(res, 502, { error: 'The model returned an empty response.' });
  const { summary, transcript } = parseSummary(text);
  let saved;
  try {
    saved = await saveVoiceNote({
      transcript, summary, audio: recording, mimeType: normalizedMime,
      durationMs: recordingDuration, model: TEXT_MODEL, keepAudio: KEEP_AUDIO,
      encounterId: typeof encounterId === 'string' ? encounterId : null,
    });
  } catch (error) {
    console.error('[database] could not save note:', error.message);
    return sendJson(res, 503, { error: 'Could not save this note to TigerData. Please retry.' });
  }
  const where = saved.linked ? `linked to encounter ${saved.encounterId}` : 'own encounter';
  console.log(`[summarize] ok, ${summary.length} chars of summary, ${where}`);
  sendJson(res, 200, { summary, transcript, encounterId: saved.encounterId, linked: saved.linked });
}

async function handleSpeak(req, res) {
  const { text } = await readJsonBody(req);
  if (typeof text !== 'string' || !text.trim()) {
    return sendJson(res, 400, { error: 'No text to speak.' });
  }
  const payload = await callGemini({
    model: TTS_MODEL,
    input: [{ type: 'user_input', content: [{ type: 'text', text: text.trim() }] }],
    response_format: { type: 'audio', mime_type: 'audio/wav', sample_rate: 24000 },
    generation_config: { speech_config: [{ voice: TTS_VOICE }] },
  });
  const data = collectAudio(payload);
  // A failure here is not fatal: the browser falls back to its own voice.
  if (!data) return sendJson(res, 503, { error: 'No audio in the model response.' });
  console.log('[speak] ok, wav returned');
  sendJson(res, 200, { audio: data, mimeType: 'audio/wav' });
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host ?? 'localhost'}`);
  try {
    if (req.method === 'GET' && url.pathname === '/api/health') {
      try {
        await checkDatabase();
        return sendJson(res, 200, { ok: true, keyConfigured: Boolean(KEY), databaseReady: true });
      } catch {
        return sendJson(res, 503, { ok: false, keyConfigured: Boolean(KEY), databaseReady: false });
      }
    }
    if (req.method === 'GET' && url.pathname === '/api/trigger') {
      return await handleTrigger(res);
    }
    if (req.method === 'POST' && url.pathname === '/api/vad') {
      return await handleVad(req, res);
    }
    if (req.method === 'POST' && (url.pathname === '/api/summarize' || url.pathname === '/api/speak')) {
      if (!KEY) return sendJson(res, 500, { error: 'GEMINI_API_KEY is not set on the server.' });
      return url.pathname === '/api/summarize'
        ? await handleSummarize(req, res)
        : await handleSpeak(req, res);
    }
    if (req.method === 'GET' && (url.pathname === '/' || url.pathname === '/index.html')) {
      const html = await readFile(path.join(HERE, 'public', 'index.html'));
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' });
      return res.end(html);
    }
    sendJson(res, 404, { error: 'Not found' });
  } catch (err) {
    const status = err?.status ?? 500;
    console.error('[error]', err?.message ?? err);
    if (!res.headersSent) sendJson(res, status, { error: err?.message ?? 'Server error' });
  }
});

try {
  await checkDatabase();
  server.listen(PORT, () => {
    console.log(`\n  Voice notes demo running at http://localhost:${PORT}`);
    if (!KEY) console.log('  WARNING: GEMINI_API_KEY is not set. Requests will fail until you set it.');
    console.log('  Saving notes in TigerData.');
    console.log('');
  });
} catch (error) {
  console.error('TigerData is unavailable or voice_note is missing. Run npm run db:init.');
  console.error(error.message);
  await pool.end();
  process.exitCode = 1;
}
