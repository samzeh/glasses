# glasses

AR glasses that recognize people and recall what you know about them.

Spec: [Face Recognition Pipeline PRD](https://claude.ai/artifact/WMPP82KGZY6X5vqPi84Rhw).

## Quick start

```sh
uv venv
source .venv/bin/activate              # or prefix every command with .venv/bin/
uv pip install -e '.[face,voice]'      # ~400 MB of wheels

glasses check                          # downloads the models on first run, ~280 MB
glasses enroll --name "Sam" --consent "myself" --camera
glasses watch
```

> **Do not run plain `uv sync` here.** It installs only the default
> dependencies and prunes everything else, which removes every model stack in
> this project — you will get `ModuleNotFoundError: No module named
> 'sherpa_onnx'` or `cv2`. Use `uv sync --all-extras`, or the
> `uv pip install -e '.[face,voice]'` above.

That is the whole loop: teach it your face, then let it look at you.

`check` prints dependency versions, weights digests, and what is in the store.
If it is clean, everything below works. On macOS the first camera command
triggers a permission prompt — if it never appears, enable your terminal under
System Settings → Privacy & Security → Camera.

## What you are doing

It is a contacts list keyed by face. Two steps, and the first is the one
people skip: **nothing can be recognised until it has been shown.**

1. **Enroll** — show it a face and give it a name. It stores a few 512-number
   vectors per person. Not the photos: the images are read and dropped.
2. **Identify** — show it a face and ask who that is. It compares against
   everyone enrolled and answers with a name, "possibly a name", or unknown.

Recognition models run on your machine. With `DATABASE_URL` configured,
people and embeddings are stored and searched in TigerData. Without it, the
gallery uses a local directory. Transcription is a hosted call by default — see
[Voice](#voice-who-is-speaking), and `--local` if that is not what you want.

### Step 1 — enroll yourself, from the webcam

No photo files needed:

```sh
glasses enroll --name "Sam" --consent "myself" --camera
```

A window opens. Look at the camera, then slowly turn your head left, right, up
and down. It captures 6 *different* views and stops. Near-identical frames are
skipped deliberately — six copies of one angle is worse than three genuinely
different ones, because the matcher averages duplicates into the same number a
single template would have given.

From photos instead, for someone who isn't in front of you:

```sh
glasses enroll --name "Dan Kim" --consent "asked him on 2026-10-23" photos/dan/
```

`--consent` is required and the note is stored with the person. That is the
PRD's posture, not politeness: enroll only people who have actually agreed.

### Step 2 — see if it knows you

```sh
glasses watch
```

Live webcam with a box round each face and a name above it:

| box | meaning |
|---|---|
| green | matched someone enrolled |
| amber | saw a face, but the frame failed the quality gate |
| grey | no face found |

The latency counter is top-left. `q` quits and prints p50/p95.

Or over still images:

```sh
glasses identify some-photos/
```

```
sam-at-desk.jpg
  primary  Sam (0.612)  iod=138px blur=492 yaw=+5 pitch=-11
  18 ms
stranger.jpg
  primary  unknown best=0.104  iod=132px blur=320 yaw=-16 pitch=-10
  17 ms
```

The number after the name is cosine similarity. Above 0.40 is a match, 0.35 to
0.40 is "possibly", below is unknown. `iod` is the face width in pixels, and
it is the figure that decides whether any of this can work on real hardware.

### If it says "unknown" when it should know you

Normal, and worth reading rather than fighting. Check in this order:

- `glasses people` — are you actually enrolled, with more than one template?
- Is the frame being rejected before it is ever matched? `identify` prints the
  failing gate terms. `pitch` dominating means the pose constant needs
  calibrating, not that your face is wrong.
- Is `iod` under 80px? Then the face is too small for the embedder and the
  answer is to get closer or raise the resolution, not to lower the threshold.

## Voice: who is speaking

The second identity channel, plus transcription on top of it. Speech-to-text
alone cannot do this: an STT API transcribes sound, it does not track people
across recordings. Identity comes from the gallery; the words come from
**Cartesia `ink-whisper`** when `CARTESIA_API_KEY` is set, and from Whisper
running locally when it is not.

Each speaker segment is transcribed **on its own** — one request per segment
— rather than transcribing the whole file and aligning word timestamps
against speaker turns. The segment already carries an identity, so its text
inherits it; there is no alignment step and therefore no bug where a sentence
lands on the wrong person because two clocks disagreed by 200 ms. Cartesia
returns word timings and they are deliberately ignored: the segment's own
start and end are the timing.

The local recogniser stays wired up rather than becoming dead code. A key
expires, or a conference network refuses to resolve anything, and either way
a slightly worse transcript beats no transcript — so a failed call falls back
to local Whisper mid-run and says so.

Those defaults apply to the standalone `glasses transcribe conversation.wav` command. The
`meet --remember` and `remember` memory flows use Gemini for transcription
and fact extraction.

Nothing extra to install: the `cartesia` extra resolves to the same voice
stack, because the API call is one multipart POST and `urllib` already does
that. Segmentation and speaker embedding still run locally — Cartesia
replaces Whisper, not the pipeline.

```sh
glasses voice-check            # fetch the voice models, ~36 MB, and say
                               # which recogniser a transcript would use

export CARTESIA_API_KEY=sk_car_…   # Cartesia transcribes; unset means local

# record yourself for 20 seconds — no audio files needed
glasses enroll-voice --name "Sam" --consent "myself" --wearer --mic

glasses enroll-voice --name "Dan" --consent "asked him" --mic --seconds 30
glasses identify-voice --mic              # listen for 10s and say who it heard
glasses transcribe conversation.wav       # who said what, attributed
glasses transcribe conversation.wav --local   # keep it all on this machine
glasses identify-voice conversation.wav --diarize --speakers 2
```

You do not have to export anything: a `.env` beside the repo is read at
startup, searched upwards from wherever you ran the command, so
`CARTESIA_API_KEY=sk_car_…` in a file is enough. A variable already set in
the shell always wins over the file. This matters more than it looks —
without the key the transcript still arrives, quietly produced by the local
recogniser, and the only sign is the `transcribing with …` line.

`--mic` records with a live level meter, so you can see it is hearing you
rather than discovering it was muted twenty seconds later. `--save out.wav`
keeps the recording; from files instead, pass paths in place of `--mic`.

**Enroll yourself as `--wearer` first.** Your voice is the loudest thing the
mic ever hears and is partly bone-conducted, so it is both the easiest to
recognise and the biggest source of mis-attribution — without it, your own
speech becomes facts about the person opposite. Enrolled once, it turns
open-set diarization into a binary question.

Segments need **3+ seconds of continuous speech**. That is the dominant term
in the whole pipeline: published work shows EER rising by roughly half going
from 3.6 s to 2.1 s, before any noise. A conversation is mostly backchannels,
so expect most of it to be thrown away, and that is correct.

### Calibrate the threshold before trusting a name

```sh
glasses voice-compare a.wav b.wav c.wav     # are these voices separable at all?
glasses voice-eval recordings/              # <person>/<recording>.wav -> a threshold
```

`accept_threshold` defaults to 0.60, which has two weak supports: it is
sherpa-onnx's own example default, and it is correct on the three-recording
real-speech sample this was tested against (genuine 0.658, impostor 0.584).
That is not a calibration. `voice-eval` does it properly on your voices.

## Recall: short memories about people

The separate [Voice Notes web app](voice-notes/README.md) saves recordings,
transcripts, and summaries in TigerData. Those notes are not part of the
`glasses ask` memory store.

Gemini transcribes a conversation in memory, then selects facts
that will help in a later conversation. The transcript is discarded after
extraction. The saved file is `glasses-store/encounters.json`, keyed by person
ID so two people with the same name remain separate.

Facts are grouped by category: name, age, school, employer, role, occupation,
industry, hobbies, interests, family, plans, favorites, projects, goals,
commitments, location, and relationships. A `topic` refines a category
(for example, favorite movie or a plan for next summer). Each value is a short
phrase with confidence and the date it was learned. Meeting date and location
are saved separately. The OLED chooses one hobby, favorite, project, or work
fact for its compact display.

Put your Gemini key in a `.env` file beside this repository:

```sh
GEMINI_API_KEY=your-key
```

The CLI and `vision_hub.py` load this file at startup. Gemini 3.8 Flash is
used by default for both extraction and questions.

For a fresh install, load a clearly marked synthetic person first. This
tests organized storage and Gemini recall without recording anyone:

```sh
glasses demo-person
glasses person "Dan Kim" --kind plan
glasses ask --person "Dan Kim" "What should I follow up on?"
glasses demo-person --clear
```

The demo profile has no face template, so it will not appear from camera
recognition. To test the OLED's face-triggered path, enroll a consenting
person (you can use your own face), record a meeting with `--remember`, then
scan that face again.

```sh
glasses meet --remember --seconds 60 --location "Cal Hacks"
glasses remember conversation.wav --location "Cal Hacks"
glasses person "Person You Met"              # organized facts, no API call
glasses ask --person "Person You Met" "What should I follow up on?"
```

`meet` uses camera and microphone together: lip sync attributes speech to a
visible face, Gemini extracts selected facts, and the face ID becomes the
memory key. `remember` works from audio and uses enrolled voice identities.
`ask --person` sends only that person's saved facts to Gemini.
`--dry-run` previews extracted facts without saving a profile. Audio is only
saved if you explicitly request an audio file.

### ESP32 OLED recall

For the ESP32 and a 128×64 SSD1306 OLED, install the Python hardware extra
(`uv pip install -e '.[hardware]'`) and the Arduino **Adafruit GFX** and
**Adafruit SSD1306** libraries. Flash
`arduino_smart_glasses.ino/test_arduino.ino`, then run:

```sh
export GLASSES_SERIAL_PORT=/dev/cu.usbmodemXXXX  # COM5 on Windows
export GLASSES_CAMERA_INDEX=0                    # select your camera
export GLASSES_LOCATION='Cal Hacks'              # optional meeting location
python vision_hub.py
```

Hold the touch pad to start a conversation and hold again to end it. A short
tap starts or stops live recognition. When a saved face appears, the OLED
shows name, first meeting date, first meeting location (or `Unknown`), and
one saved fact (or `None saved`). Text scrolls down one pixel every 160 ms.
Set `GLASSES_PREVIEW=1` to open camera debug windows with a GUI OpenCV build.

Older versions could index full transcripts in Elastic. This version does
not write them. Existing Elastic indices are not changed automatically; if
you used that earlier feature, delete those old indices separately when you
no longer need them.

## Storage: TigerData, one database for both halves

The store above is a directory, and that is right for M0 and wrong for
everything after it. The real system of record is Postgres —
**TimescaleDB + pgvector in one instance** — because this project has exactly
the two shapes of data that are usually split across two systems, and the
join between them *is* the product:

- **Relational profiles.** A few rows per person, read constantly, written
  rarely: who they are, which encounters they appeared in, what was learned
  about them, and their 512-d templates. The gallery search is a pgvector
  query against the same rows.
- **High-frequency streams.** `face_observation` and `speech_observation`,
  one row per visible face per span and per speech segment, on the ESP's
  microsecond clock. Append-only, and growing with wear time rather than with
  the number of people.

"Who was in front of me while this sentence was spoken" is a join across the
two. Hold them in separate systems and that join becomes application code
reconciling two clocks — which is the one bug this whole design is arranged to
avoid.

### Connect to Tiger Cloud

Use a Tiger Cloud development service with the TimescaleDB add-on. Copy its
PostgreSQL connection URL from the connection panel into the local `.env`:

```dotenv
DATABASE_URL=postgresql://tsdbadmin:YOUR_PASSWORD@YOUR_HOST:YOUR_PORT/tsdb?sslmode=require
```

Keep `.env` local; it is ignored by git. The app reads it automatically, and
an exported `DATABASE_URL` takes precedence. The Tiger MCP connection lets
the assistant manage the database; the app needs this URL separately.

```sh
.venv/bin/python -m pip install -e '.[db]'
.venv/bin/glasses db init
.venv/bin/glasses db stats
```

`db init` applies the schema to a new database. For an existing database it
applies the cloud enrollment upgrade without resetting saved data. Run it
after updating this branch. On first setup it reports both observation hypertables and the
two real-time continuous aggregates. Use `db stats` to verify the connection,
row counts, and live per-minute view. The schema enables compression after
7 days and raw observation retention of 90 days.

Enrollment, recognition, and voice use the TigerData gallery automatically
when `DATABASE_URL` is set. Structured memories for Gemini recall currently
live in `glasses-store/encounters.json`; an explicit `--store path` selects a
different local memory directory.

To move an existing local gallery and captured files into TigerData:

```sh
.venv/bin/glasses db migrate-local
.venv/bin/glasses db migrate-local --remove-local
```

The first command imports and verifies without deleting. The second verifies
again and removes the migrated local files after the database commit. IDs,
consent, timestamps, model versions, quality scores, and embeddings are
preserved. Original gallery files and files under `data/` are archived
byte-for-byte in `local_import_file`, including photos. Model weights and
configuration stay on your machine. Migration stops on conflicting data.
Use `--local-store path` and `--media-dir path` for other source directories.
`glasses forget "Sam"` also deletes archived source files associated with that
person. Shared source archives are removed in full; other people's canonical
database records stay intact.

The former Elasticsearch transcript path is no longer part of the active CLI.

### Start a local database

```sh
docker run -d --name glasses-tiger -p 55432:5432 \
    -e POSTGRES_PASSWORD=glasses -e POSTGRES_DB=glasses \
    timescale/timescaledb-ha:pg17

export DATABASE_URL=postgresql://postgres:glasses@127.0.0.1:55432/glasses
uv pip install -e '.[db]'

glasses db init      # tables, hypertables, aggregates, compression, retention
glasses db stats     # what is in there, and the live per-minute view
```

`timescaledb-ha:pg17` is the image to use rather than plain `postgres`: it
ships TimescaleDB, pgvector and pgvectorscale together, which is the whole
claim being made here. `glasses db init` can be rerun safely to apply cloud
enrollment upgrades; `--reset` drops everything first.

### What the schema does with it

| | |
|---|---|
| hypertables | `face_observation`, `speech_observation`, one day per chunk |
| continuous aggregates | `face_activity_per_minute`, `speech_activity_per_minute` |
| the live view | `encounter_minute` — both sensors, one row per encounter per minute |
| compression | segmented by `encounter_id`, after 7 days |
| retention | raw stream dropped after 90 days, the per-minute history kept |

Three decisions in there are worth the words:

**Two clocks on every observation row, and only one of them is the
authority.** `start_device_clock_us` is the ESP's monotonic counter and is what
binding does interval arithmetic on. `observed_at` is the same instant mapped
through the clock's single wall anchor, and it exists because `time_bucket`,
chunk exclusion and humans all need it. Binding never reads it back: an NTP
step mid-encounter would reorder two observations and file a sentence against
the wrong face.

**The aggregates group by `person_id` and count distinct one level up.** A
continuous aggregate cannot hold a `count(DISTINCT …)` — incremental
materialization needs a partial state that merges, and a distinct count's does
not. So each aggregate holds one row per person per minute, and
`encounter_minute` counts *those*, over data already collapsed by a factor of
thousands.

**Real-time aggregation is on (`materialized_only = false`).** The default
since TimescaleDB 2.13 hides buckets the refresh job has not reached yet —
which, at a one-minute bucket, means the current minute. That is the one
minute a live demo is looking at.

Nothing compresses during a demo, because nothing in a demo is seven days old.
To watch it work:

```sql
SELECT compress_chunk(c) FROM show_chunks('face_observation') c;
```

`glasses db stats` then reports the chunk count and the before/after size.

### Running the Postgres tests

The default suite is numpy-only: no network, no Docker, no driver. The
Postgres tests skip themselves unless `DATABASE_URL` is set, and when it is
they run against the real thing in a schema of their own
(`glasses_test_<pid>`), dropped afterwards:

```sh
DATABASE_URL=postgresql://postgres:glasses@127.0.0.1:55432/glasses \
    .venv/bin/python -m pytest tests/test_pgvector_gallery.py
```

That is also how `PgVectorGallery` stopped being the untested half of the
gallery. Running it found a real one: `hnsw.ef_search` defaults to 40, so a
`LIMIT 64` — and 64 is `MatchConfig.search_limit`, the default every match
uses — came back with 40 rows the moment the planner chose the vector index.
No error, just a quietly truncated candidate list feeding the top-3
aggregation. The gallery now raises `ef_search` to the requested limit and
turns on pgvector 0.8's iterative scan, which is also what makes the
`(model_name, model_version)` filter safe: an HNSW scan finds the nearest rows
*before* the filter is applied, so a gallery part-way through a re-embedding
could otherwise return nothing for the new model while thousands of its
templates sat in the table.

## What is built

Face recognition, voice attribution, the encounter clock, Gemini fact
extraction and recall, the local structured memory store, and an optional
TigerData gallery are built. ESP32-S3 firmware lives in
`arduino_smart_glasses.ino/` alongside `vision_hub.py`.

Still missing from the PRD: the dwell-and-speech trigger is written
(`encounter/session.py`) but nothing calls it, and nothing writes the
observation hypertables yet. The OLED shows a short local summary; broader
recall remains in the terminal.

```
src/glasses/
  face/
    config.py      thresholds, in one place, with CALIBRATE markers
    types.py       the values passed between stages
    detect.py      1. detect          SCRFD, + primary-subject choice
    align.py       2. align           Umeyama similarity transform -> 112x112
    quality.py     3. quality gate    score, size, blur, pose, luminance
    embed.py       4. embed           ArcFace buffalo_l, 512-d
    gallery.py     ---                in-memory and pgvector backends
    match.py       5. match           cosine, mean of a person's top 3
                   6. resolve         match / low confidence / unknown
    models.py      loading ONNX nets, and digesting the weights
    pipeline.py    the six, wired, with per-stage timings
  evaluation/
    verification.py  TAR at a fixed FAR, swept over gallery size
  db/
    schema.sql     the full data model: profiles, and the stream hypertables
    connect.py     the connection, and applying the schema one statement at a time
    store.py       PgStore — the same operations as store.py, on Tiger
  voice/
    config.py      thresholds; min_duration_s is the one that matters
    audio.py       WAV/ffmpeg loading, resampling, level and SNR
    segment.py     VAD, and pyannote diarization, behind one Protocol
    quality.py     the gate: duration, level, SNR, clipping
    embed.py       WeSpeaker CAM++, 512-d
    models.py      model downloads; no HF token, no torch
    pipeline.py    segment -> merge -> gate -> embed -> wearer -> match
    transcribe.py  one Transcriber Protocol, per speaker segment
    cartesia.py    Cartesia ink-whisper: the default, one POST per segment
  encounter/
    clock.py       one monotonic microsecond clock, shared by both sensors
    types.py       Interval, FaceObservation, SpeechObservation, Encounter
    session.py     the dwell + partner-speech trigger state machine
  identity/        shared by both channels: Embedding, gallery, matcher
  search/
    config.py      ELASTIC_URL, ELASTIC_API_KEY, index names; nothing hardcoded
    client.py      JSON over HTTP on the stdlib, behind one Transport seam
    mappings.py    the two index mappings, dynamic: strict
    documents.py   TranscriptSegment, PersonFact, deterministic ids
    indexer.py     one-way sync, per-document bulk error reporting
    agentbuilder.py  the 6 tools, the agent's instructions, converse
    seed.py        four synthetic encounters, every doc source=seed
  store.py       file-backed people + templates, for M0
  cli.py         glasses check / enroll / identify / watch / bench
  voice_cli.py   the voice subcommands
  search_cli.py  Gemini-backed glasses ask and local person lookup
  db_cli.py      glasses db init / stats
```

Faces and voices share `identity/` entirely — the gallery, the top-3
matcher, the three-way resolve and the `model_version` discipline are
modality-agnostic. The voice pipeline added no matching code of its own.

Everything holding real logic runs on numpy alone. Only `detect.py` and
`embed.py` need model weights, and both sit behind Protocols, so the suite runs
without downloading anything. `search/` adds no dependency either: it is
four verbs against a JSON API, so it uses `urllib` and every request goes
through one `Transport`, which is the seam the tests replace.

## Measured latency

Apple Silicon, 1280×886 frame, 6 faces detected, 2 through the gate:

| stage | CPU | CoreML | budget |
|---|---|---|---|
| detect | 44 ms | 9.6 ms | |
| align | 0.6 ms | 0.4 ms | |
| quality gate | 1.1 ms | 0.7 ms | |
| embed (2 faces) | 149 ms | 6.5 ms | |
| match | 0.1 ms | 0.1 ms | 50 ms |
| **total p50** | **192 ms** | **17.6 ms** | **400 ms** |

The budget is the PRD's server-side share; upload and render are not included.

**CoreML is the default where available, and it is free.** Same ArcFace
ResNet-50 weights, so there is no accuracy trade — measured cosine between CPU
and CoreML embeddings of the same crop is 0.9998, three orders of magnitude
below the gap the 0.40 accept threshold cares about, and a gallery built on one
provider matches probes from the other. It costs ~5 s of graph compilation at
first inference, once per process. `GLASSES_PROVIDERS=CPUExecutionProvider`
pins it back, which is what to do if two machines must agree bit for bit.

At 20× headroom, speed is not the constraint. Accuracy on real ESP frames is.
The smaller `buffalo_s` pack (MobileFaceNet, 13 MB) embeds in 10 ms on plain
CPU if you ever need it, but it trades accuracy for a budget that is already
met — the opposite of the direction this needs to move. The PRD's AdaFace swap
is the one worth making, and it is about low-quality imagery, not speed.

## Three things that are load-bearing

**The quality gate runs before the embedder, not after.** Rejecting a bad frame
is cheaper than recovering from the bad embedding it produces, and the gate is
what buys the accuracy target. It should reject 60–80% of ESP frames; the
capture burst exists so that is affordable. If the pass rate comes out far
below 20%, the camera settings are wrong, not the gate.

**Every embedding carries its model name and version, and every search filters
on both.** The gallery will be re-embedded at least once — probably buffalo_l
to AdaFace. Without this, a swap silently mixes incomparable vectors in one
index and recognition collapses in a way that reads as a threshold problem.

**A person is scored by the mean of their top 3 template similarities, never by
a centroid.** A centroid built from one good frontal and one bad profile is
worse than either, because the mean vector describes a face nobody has. Mean
rather than max of the top 3 is a further choice: max turns a single lucky
template into an identity, which is where false accepts come from.

## The numbers in `config.py` are not findings

They are starting points, and several are marked CALIBRATE. The PRD is right
that published thresholds are measured on clean benchmark photos and will not
transfer to an OV2640 behind a plastic lens. In particular
`min_blur_variance = 60.0` is a placeholder from webcam-grade imagery and
`accept_threshold = 0.40` is the PRD's starting value, not a measured one.

`nose_frontal_fraction` is the one to fix first, and it has already bitten
once. Deriving it from the ArcFace canonical reference gives 0.495 and looks
principled — that reference *is* the aligner's definition of frontal. It is
still wrong, because the constant belongs to whatever produces the landmarks,
and SCRFD puts its nose keypoint near 0.66. With 0.495 the estimator read
near-frontal faces as pitched down 45°, and the pitch gate threw them away.

Measure it on your own frames before trusting it:

```sh
glasses calibrate-pose path/to/frames/     # use 50+ roughly frontal faces
```

That also reports how much apparent pitch comes from face shape alone. On the
tiny sample available here it is ±20°, which is most of the ±30° gate — so if
`pitch` dominates your rejections, suspect this term before the camera.

Calibrating needs a held-out set of a few hundred frames from the real glasses
camera across real conditions, labelled by person. `glasses.evaluation` sweeps
the threshold against such a set and reports the operating point at a fixed
FAR, at each gallery size. Track false-accept rate against gallery size from
the first measurement: a threshold tuned at ten people produces steady
misidentifications at two hundred, and it degrades gradually enough that you
will not notice without the chart.

## Known gaps

- Yaw and pitch come from a landmark heuristic whose constants are calibrated
  on 6 faces. A weak-perspective PnP solve against a canonical 3D 5-point
  model is the principled upgrade, once there is a held-out set to score it on.
- `watch` and `meet` have both been run against a live camera.
- The voice threshold is supported by three recordings of two people, in
  Chinese, through an English-trained model. Run `voice-eval` before relying
  on a name.
- `DiarizingSegmenter` runs and is wired up, but has not been checked against
  a recording with known speaker turns — only the single-speaker path has.
- The Cartesia call has been made for real: `ink-whisper` returned a correct
  transcript of a 10 s clip in 400 ms. That exercised `CartesiaTranscriber`
  directly, not `glasses transcribe` end to end, and one recording is not a
  WER measurement — the far-field case the PRD worries about is still
  unmeasured.
- ReDimNet is the voice equivalent of the AdaFace swap: least cross-domain
  degradation of the current crop, which is what far-field audio needs.
- `PgVectorGallery` and cloud migration are covered by the database integration
  tests, which require `DATABASE_URL` and run in an isolated schema.
- Gemini extraction uses structured output. The selected facts are written
  locally; older Elastic indices are not read or updated by this flow.
- The retention and compression *policies* exist and are scheduled; neither
  has been observed firing, because nothing here is seven days old. The
  manual `compress_chunk` path is tested, which proves the configuration is
  usable rather than merely present.
- The continuous aggregates bucket at one minute, which is right for a demo
  and probably wrong for a year of wear. A second, coarser aggregate rolled
  up from the first is the normal answer and is not written.
- No anti-spoofing and no re-identification across clothing changes. Both are
  deliberately excluded from v1.

## Privacy

`schema.sql` carries `person.consented_at`, `media_asset.retention_expires_at`
and a soft-delete column on every table holding biometric data, because the
PRD puts the consent and deletion paths in M0 rather than retrofitting them.
The deletion job that enforces retention is not written yet. Until it is, do
not capture anyone.

The `meet` and `remember` paths send temporary speech segments and their
attributed text to Gemini for transcription and fact extraction. Later
questions send only saved structured memories to Gemini. The separate
`glasses transcribe` command can send speech to Cartesia when its key is set;
`--local` keeps that command's transcription local. Frames, face embeddings,
and the gallery remain local.
