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

Everything is local. There is nothing to upload and no account: the models run
on your machine and the gallery is a directory you can delete.

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
Whisper, running locally.

Each speaker segment is transcribed **on its own** rather than transcribing
the whole file and aligning word timestamps against speaker turns. The
segment already carries an identity, so its text inherits it — there is no
alignment step and therefore no bug where a sentence lands on the wrong
person because two clocks disagreed by 200 ms.

```sh
glasses voice-check            # fetch the voice models, ~36 MB

# record yourself for 20 seconds — no audio files needed
glasses enroll-voice --name "Sam" --consent "myself" --wearer --mic

glasses enroll-voice --name "Dan" --consent "asked him" --mic --seconds 30
glasses identify-voice --mic              # listen for 10s and say who it heard
glasses transcribe conversation.wav       # who said what, attributed
glasses identify-voice conversation.wav --diarize --speakers 2
```

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

## Recall: ask about the people you have met

Recognition answers *who is in front of me*. Recall answers the other
question — *who have I met who works in biotech?* — and it is a search
problem, not a recognition one. The transcripts already exist, attributed
per speaker; what was missing was somewhere to ask them from.

That somewhere is Elastic, with an [Elastic Agent
Builder](https://www.elastic.co/docs/explore-analyze/ai-features/elastic-agent-builder)
agent in front of it:

```sh
export ELASTIC_URL='https://my-project.es.us-east-1.aws.elastic.cloud'
export ELASTIC_API_KEY='…'

glasses search-setup                  # create the two indices
glasses agent-setup                   # apply the tools and the agent
glasses search-seed                   # four synthetic encounters to ask about
glasses ask "who have I met who works in biotech"
```

The shape of the answer — the agent writes the prose, so the wording will
differ:

```
? who have I met who works in biotech

  Dan Kim — Helix Bio, "a biotech company doing protein design". Cal Hacks,
  24 Oct. He also said he would send you the benchmark dataset and a preprint.

  Marcus Webb — research scientist at Verida Therapeutics, "biotech,
  immunology side". Cal Hacks, 24 Oct, about an hour later.

  · glasses_find_people_by_fact(kind='industry', value='*biotech*')
  · glasses_get_person_facts(person='*dan*')

  2.1s  ·  conversation 9f3a…
```

The tool calls are printed on purpose. An answer that names the queries
behind it is one the audience can check, and it is the first thing to read
when the answer is wrong.

### Elastic is a derived index, not the system of record

```
Postgres/TigerData          Elastic                Agent Builder
(the truth)          →   (a searchable copy)  →   (6 tools + 1 agent)
  people, encounters       transcript_segment       glasses ask "…"
  facts, templates         person_fact
```

One-way sync, always. Nothing reads an answer out of Elastic and stores it
as truth, and the index can be dropped and rebuilt from the source at any
time — which is what makes it safe to put an LLM in front of. An agent
querying a derived copy can be wrong about what it found; an agent
querying the system of record can be wrong about what *is*.

Two document types, both with deterministic ids so that re-running the
sync overwrites rather than duplicates:

| | fields |
|---|---|
| `transcript_segment` | text, person, encounter, start/end ms, `is_wearer`, location, time |
| `person_fact` | kind, value, person, encounter, `evidence_span`, confidence |

`is_wearer` is carried through everything and the agent is instructed
about it explicitly: your own speech must never become a fact about the
person opposite. `person_id` is nullable on both, matching
`db/schema.sql` — an unidentified speaker and an unassigned fact are real
states, and they stay searchable rather than being resolved by a guess.

### The agent's tools are code, not clicks

`src/glasses/search/agentbuilder.py` holds the six tool definitions and
the agent's instructions. `glasses agent-setup` applies them and is
idempotent: edit the file, run it again. `--dry-run` prints exactly what
would be sent and needs no credentials at all, which is how the
definitions get reviewed:

```sh
glasses agent-setup --dry-run
```

Two `index_search` tools do the open questions — Elastic picks keyword,
semantic or hybrid retrieval itself, which beats a hand-written full-text
query. Four `esql` tools do the precise ones (one person's facts, one
person's utterances, who was at a place, who matches a fact), because
those are exact filters that should return the same rows every time, with
the evidence columns rather than a relevance-ranked blob.

### Seed data is obviously synthetic

A recall demo needs a past, and the honest way to get one is to fabricate
it openly rather than to record strangers so a demo looks better. Every
seeded document carries `source: "seed"`, every person id starts
`seed-person-` and every encounter id `seed-encounter-`:

```sh
glasses search-seed            # 26 utterances, 22 facts, 4 encounters
glasses search-check           # capture and seed counted separately
glasses search-seed --clear    # removes every source=seed document, and
                               # cannot touch a real capture
```

Real recordings go in the same indices as `source: "capture"`:

```sh
glasses search-index conversation.wav --location "Cal Hacks"
```

Facts are not extracted from real audio yet — the `fact` table and its
mapping exist, the extractor does not — so `search-index` writes segments
only. Segments alone already answer "what did Dan say he'd send me?".

### Runbook: pointing it at an Elastic Serverless project

Agent Builder is GA on Serverless and in 9.3 (preview in 9.2). **This path
has not been run against a live project** — everything in `glasses.search`
is tested against a fake HTTP transport, and the API shapes come from the
Elastic docs. Expect to debug the first run.

1. Create an **Elasticsearch Serverless** project at
   [cloud.elastic.co](https://cloud.elastic.co).
2. From the project's connection details, copy the **Elasticsearch
   endpoint** (`https://….es.….elastic.cloud`) and create an **API key**.
3. Export both. Agent Builder is a *Kibana* API, so the commands also need
   the Kibana endpoint; it is derived by swapping `.es.` for `.kb.`, which
   is right on Elastic Cloud. Set it yourself if that is wrong for your
   deployment:

   ```sh
   export ELASTIC_URL='https://my-project.es.us-east-1.aws.elastic.cloud'
   export ELASTIC_API_KEY='…'
   export ELASTIC_KIBANA_URL='https://my-project.kb.us-east-1.aws.elastic.cloud'
   ```

4. `glasses search-check` — prints both endpoints, the key redacted, the
   Elasticsearch version, what is in each index, and whether the agent
   exists. Fix anything it reports before going on.
5. `glasses search-setup` then `glasses agent-setup` then
   `glasses search-seed`.
6. `glasses ask "who did I meet at Cal Hacks"`.

Optional: `glasses search-setup --semantic` adds a `semantic_text` field
and `glasses search-seed --semantic` populates it. It needs an inference
endpoint in the project, so it is off by default — a demo that can fail on
stage is not an improvement.

Other variables: `ELASTIC_INDEX_PREFIX` (default `glasses`) and
`ELASTIC_AGENT_ID` (default `glasses_recall`). No credential is ever read
from a file in this repo.

### Every command

| command | what it does |
|---|---|
| `glasses check` | dependency versions, weights digests, what is in the store |
| `glasses enroll --name N --consent "…" --camera` | capture 6 views from the webcam |
| `glasses enroll --name N --consent "…" photos/` | same, from image files |
| `glasses identify photos/` | run the pipeline over images and print verdicts |
| `glasses watch` | live webcam, boxes and names |
| `glasses people` | who is enrolled, and their consent record |
| `glasses forget "Sam"` | delete a person and all their data, in one operation |
| `glasses calibrate-pose frames/` | measure `nose_frontal_fraction` for your camera |
| `glasses bench frames/` | per-stage latency against the budget |
| `glasses voice-check` | fetch the voice models, report the wearer |
| `glasses enroll-voice --name N --consent "…" --mic` | record a voice now (`--wearer` for yourself) |
| `glasses enroll-voice --name N --consent "…" audio/` | same, from audio files |
| `glasses identify-voice --mic` | listen now and say who it heard |
| `glasses identify-voice audio/` | same, from files (`--diarize` for several voices) |
| `glasses forget-voice "Sam"` | drop a bad voice enrollment, keep the face one |
| `glasses voice-compare a.wav b.wav` | pairwise similarity between recordings |
| `glasses transcribe audio/` | who said what, attributed (`--mic` to listen now) |
| `glasses voice-eval recordings/` | calibrate `accept_threshold` on your own voices |
| `glasses ask "who did I meet at Cal Hacks"` | ask the Agent Builder agent about people you have met |
| `glasses search-check` | what Elastic is configured, reachable, and holding |
| `glasses search-setup` | create the two indices (`--recreate`, `--semantic`, `--dry-run`) |
| `glasses agent-setup` | apply the Agent Builder tools and agent (`--dry-run`) |
| `glasses search-seed` | load the synthetic demo encounters (`--clear` to remove) |
| `glasses search-index audio/` | transcribe real recordings and index what was said |

`--store DIR` moves the gallery, `--help` works on any subcommand.

The store is a directory (`glasses-store/` by default): an `.npz` of vectors
beside a JSON index. Delete it to start over.

### As a library

```python
from glasses.face import ArcFaceEmbedder, FacePipeline, InMemoryGallery, ScrfdDetector

pipeline = FacePipeline(ScrfdDetector(), ArcFaceEmbedder(), InMemoryGallery())
result = pipeline.process_frame(frame)  # one frame
index, result = pipeline.best_frame(burst)  # pick the best of a burst
```

Tests need none of that — numpy alone:

```sh
uv pip install -e . --group dev && .venv/bin/python -m pytest
```

## What is built

Server-side face recognition only — the six stages of the PRD's hot loop, plus
the schema and the measurement harness they depend on. Nothing else from the
PRD exists yet: no firmware, no ingest, no trigger, no audio pipeline, no
identity resolution, no surfacing.

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
    schema.sql     the full data model, not only the face tables
  voice/
    config.py      thresholds; min_duration_s is the one that matters
    audio.py       WAV/ffmpeg loading, resampling, level and SNR
    segment.py     VAD, and pyannote diarization, behind one Protocol
    quality.py     the gate: duration, level, SNR, clipping
    embed.py       WeSpeaker CAM++, 512-d
    models.py      model downloads; no HF token, no torch
    pipeline.py    segment -> merge -> gate -> embed -> wearer -> match
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
  search_cli.py  glasses ask, and the setup/seed/index commands
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
- `watch` has been exercised on still images but not on a live camera here.
- The voice threshold is supported by three recordings of two people, in
  Chinese, through an English-trained model. Run `voice-eval` before relying
  on a name.
- `DiarizingSegmenter` runs and is wired up, but has not been checked against
  a recording with known speaker turns — only the single-speaker path has.
- ReDimNet is the voice equivalent of the AdaFace swap: least cross-domain
  degradation of the current crop, which is what far-field audio needs.
- `PgVectorGallery` is written but untested — it needs a Postgres with
  pgvector to test against.
- **The whole Elastic path is unverified against a live cluster.** No
  request in `glasses.search` has ever left the process: the tests run
  against a fake transport, and the request shapes come from Elastic's
  docs rather than from a response. The specific things to expect to
  debug are the Kibana-vs-Elasticsearch base URL, whether `PUT
  /api/agent_builder/tools/{id}` accepts each definition as written, and
  whether `TO_LOWER(value.keyword) LIKE ?value` is accepted in an ES|QL
  tool.
- Facts are only ever seeded. Nothing extracts a fact from a real
  transcript yet, so `search-index` indexes segments and no facts.
- Elastic holds a copy, and the sync is manual. There is no trigger that
  indexes an encounter when it closes; `search-index` and `search-seed`
  are run by hand.
- No anti-spoofing and no re-identification across clothing changes. Both are
  deliberately excluded from v1.

## Privacy

`schema.sql` carries `person.consented_at`, `media_asset.retention_expires_at`
and a soft-delete column on every table holding biometric data, because the
PRD puts the consent and deletion paths in M0 rather than retrofitting them.
The deletion job that enforces retention is not written yet. Until it is, do
not capture anyone.
