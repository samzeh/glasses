# Voice Notes

Record a voice note on your phone, get a spoken summary back, and keep the
recording, transcript, and summary in the glasses project's TigerData database.
One Gemini call transcribes and summarizes the audio, and a second turns the
summary into speech.

No build step. Node.js 20+, the `pg` package, Gemini, and TigerData are required.

Part of the `glasses` project. This phone web app shares the TigerData database
with the Python CLI, but does not need a Python environment to run.

## Run it

1. Get a Gemini API key and a TigerData `DATABASE_URL`. On a new database, run
   `glasses db init` first to create the shared `encounter` table.

2. Create `.env`:

   ```bash
   cd voice-notes
   cp .env.example .env
   # then put your GEMINI_API_KEY and DATABASE_URL in it
   ```

3. Install the PostgreSQL driver, create the Voice Notes table, and start the server:

   ```bash
   npm install
   npm run db:init
   npm run dev
   ```

   Open <http://localhost:8787>. It works on your laptop immediately, because
   `localhost` counts as a secure context.

## Get it on your phone

The microphone will not open over plain HTTP, and a local network address like
`192.168.1.20` does not count as secure. You need a real HTTPS URL. Pick one:

**Cloudflare tunnel**, nothing to sign up for:

```bash
brew install cloudflared
cloudflared tunnel --url http://localhost:8787
```

It prints a `https://something.trycloudflare.com` URL. Open that on your phone.

**Tailscale**, if you already use it:

```bash
tailscale serve --https=443 http://localhost:8787
```

This is the step that trips people up at hackathons. Do it before you are on
stage.

## How it works

```
phone browser  --MediaRecorder-->  base64 audio
      |
      v
POST /api/summarize   ->  Gemini 3.8 Flash (audio in, JSON out)
      |                         ->  encounter + voice_note in TigerData
      |                         ->  summary + transcript in browser
      |
      v
POST /api/speak       ->  Gemini 3.8 Flash TTS  ->  24 kHz WAV
      |
      v
played in an <audio> element
```

Gemini accepts audio directly, so there is no separate transcription service.
The server commits an `encounter` and linked `voice_note` in one database
transaction before it reports success. `voice_note` stores the transcript,
summary, original audio bytes, MIME type, duration, and model. Set
`VOICE_NOTES_STORE_AUDIO=false` in `.env` to keep only text and metadata. The
Gemini speech output is played in the browser and is not stored.

Voice Notes does not send its notes to Elasticsearch. The existing `glasses ask`
command searches Elastic's separate transcript and fact indices, so it will
not find these Voice Notes until a sync or search integration is added.

To inspect saved notes in TigerData:

```sql
SELECT e.id, e.started_at, v.summary, v.transcript,
       v.audio_mime_type, v.audio_bytes, octet_length(v.audio_data) AS stored_audio_bytes
FROM public.voice_note AS v
JOIN public.encounter AS e ON e.id = v.encounter_id
ORDER BY v.created_at DESC;
```

## Cost

Gemini may be free within its free tier; TigerData storage and usage depend on
your service plan. Flash allows 1,500 requests per day, which is
1,500 recordings. Each note uses two requests, so the real ceiling is about 750
notes per day, and the per-minute limit of 10 requests is the one you would hit
during a demo if you hammered it.

The one catch: free-tier Gemini content is used to improve Google's products.
Do not record anything confidential.

## Things worth knowing before you demo

- **Recording is capped at five minutes** and stops itself. Inline audio has to
  fit a 20 MB request.
- **iOS will not autoplay the summary.** The app warms up both audio channels on
  the record tap, which usually gets around it, but the "Play summary" button is
  always there as the reliable path. Tap it if nothing happens.
- **Safari records to a different format** than Chrome. The server maps Safari's
  `audio/mp4` to the `audio/m4a` type Gemini expects, so both work.
- **Backgrounding the tab stops the recording.** Keep the phone awake and the
  browser in front.
- **If the voice sounds robotic**, the model's text-to-speech call failed and the
  browser's built-in voice took over. Check the server log for a `[speak]` line.

## Files

| File | What it does |
|---|---|
| `server.js` | HTTP server, holds the API key, proxies Gemini, saves notes |
| `db.js` / `db-init.js` / `schema.sql` | TigerData connection, write transaction, and table setup |
| `public/index.html` | The entire front end: recorder, level meter, playback |
