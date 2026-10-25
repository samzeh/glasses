# Voice Notes

Record a voice note on your phone, get a spoken summary back. One Gemini call
transcribes and summarizes the audio, a second turns the summary into speech.

No dependencies. No build step. Two files.

Part of the `glasses` project. This is the voice side: a self-contained web app
that runs on a phone, independent of the Python CLI and the ESP32 firmware. It
shares nothing with them and needs no Python environment.

## Run it

1. Get a free API key at <https://aistudio.google.com/apikey>. No credit card.

2. Create `.env`:

   ```bash
   cd voice-notes
   cp .env.example .env
   # then put your key in it
   ```

3. Start the server:

   ```bash
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
POST /api/summarize   ->  Gemini 3.8 Flash (audio in, JSON out)  ->  summary + transcript
      |
      v
POST /api/speak       ->  Gemini 3.8 Flash TTS  ->  24 kHz WAV
      |
      v
played in an <audio> element
```

Gemini accepts audio directly, so there is no separate transcription service.
That is what makes the whole thing free.

## Cost

Nothing, within the free tier. Flash allows 1,500 requests per day, which is
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
| `server.js` | HTTP server, holds the API key, proxies both Gemini calls |
| `public/index.html` | The entire front end: recorder, level meter, playback |
