import os
import select
import socket
import sys
import threading
import time
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime

import cv2

from glasses.binding.asd import AsdConfig
from glasses.coach import Coach
from glasses.device import GLASSES_CAMERA, GlassesLink, find_camera_index, open_camera
from glasses.encounter.clock import Clock, seconds_to_us, us_to_seconds
from glasses.encounter.record import build_encounter, face_observations, speech_observations
from glasses.env import load as load_env
from glasses.face import ArcFaceEmbedder, FacePipeline, ScrfdDetector
from glasses.face.models import default_providers
from glasses.facts import GeminiError, GeminiExtractor
from glasses.meet import AudioRecorder, Capture, FaceTracker, MouthReader, assign_speakers
from glasses.oled import UNKNOWN, PersonDisplay
from glasses.profiles import profile_path, save_encounter
from glasses.recall import identify_track
from glasses.storage import open_store, store_label
from glasses.voice import VadSegmenter, VoicePipeline, WeSpeakerEmbedder
from glasses.voice.gemini import GeminiTranscriber
from glasses.voice.transcribe import Utterance

load_env()

CAMERA_INDEX = find_camera_index()
if CAMERA_INDEX is None:
    sys.exit(f"No {GLASSES_CAMERA} camera found. Plug in the Logitech; the Mac camera is not used.")
PREVIEW = os.environ.get("GLASSES_PREVIEW") == "1"
#: Microphone to record from: an input index, or part of its name. Defaults
#: to the C270's own mic; the macOS default input follows whatever headset
#: is connected rather than the camera on the glasses.
MIC = os.environ.get("GLASSES_MIC") or GLASSES_CAMERA
if MIC and MIC.isdigit():
    MIC = int(MIC)
#: Which device this is, written on every encounter. One hub per pair of
#: glasses, so the host name is the honest answer until there are more.
DEVICE_ID = os.environ.get("GLASSES_DEVICE_ID") or socket.gethostname()

print("Loading Face and Voice machine learning models...")
store = open_store()
print(f"Store: {store_label()}")
#: Only the Postgres store can hold an encounter. Offline runs still
#: recognise faces; they just have nowhere to put the history.
records_encounters = hasattr(store, "record_encounter")
if not records_encounters:
    print("No DATABASE_URL — recognising only, encounters will not be recorded.")
providers = default_providers()

face_pipeline = FacePipeline(
    detector=ScrfdDetector(providers=providers),
    embedder=ArcFaceEmbedder(providers=providers),
    gallery=store.gallery("face"),
)

voice_pipeline = VoicePipeline(
    segmenter=VadSegmenter(),
    embedder=WeSpeakerEmbedder(),
    gallery=store.gallery("voice"),
    wearer_embeddings=store.wearer_embeddings(),
)

mouths = MouthReader(providers=providers)

try:
    link = GlassesLink()
except RuntimeError as error:
    sys.exit(str(error))
print(f"Connected to the glasses on {link.port} ({link.oled or 'no boot report'}).")
#: Talking points need facts from TigerData; offline, the plain profile card.
coach = Coach(store, link) if hasattr(store, "facts_of") else None
display = PersonDisplay(link, store, coach=coach)
display.clear()
#: How often the talking points are redone from what has been said so far.
LIVE_EVERY_S = 10
#: How much of the conversation the live talking points look at.
LIVE_WINDOW_S = 30
SAMPLE_RATE = 16_000


def next_command(recording):
    """A touch event from the glasses, or Enter in this terminal as a stand-in
    double tap for when the touch sensor is not registering."""
    cmd = link.poll()
    if cmd:
        return cmd
    if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
        return "CAM_OFF" if recording else "CAM_ON"
    return None


def camera_failed():
    """Say why the camera stopped. Terminal only: the OLED is for people."""
    print(
        f"[camera] no frame from the {GLASSES_CAMERA}. If macOS printed 'not authorized', "
        "allow this terminal app in System Settings > Privacy & Security > Camera."
    )


def resolve_identities(tracker):
    """Who each tracked face is, resolved once for the whole session.

    One embedding per track rather than per utterance: the same face often
    speaks several times and the answer cannot change between them.
    """
    identities = {}
    for track in tracker.tracks.values():
        if track.best_frame is None:
            continue
        frame_result = face_pipeline.process_frame(track.best_frame)
        if not frame_result.faces:
            continue
        best = max(frame_result.faces, key=lambda f: f.detection.bbox.width)
        if best.quality.passed and best.match and best.match.person_id:
            identities[track.id] = (best.match.person_id, float(best.match.similarity))
    return identities


def record_session(tracker, captures, identities, clock, started_us, ended_us, frame_width):
    """Write the session to the store, if the store can hold one.

    A failure here is reported and swallowed: losing an encounter is bad,
    and taking the hub down mid-conversation with it is worse.
    """
    if not records_encounters:
        return None
    faces = face_observations(list(tracker.tracks.values()), identities, frame_width=frame_width)
    speech = speech_observations(captures)
    if not faces and not speech:
        print("[DB] nothing observed, no encounter recorded.")
        return None

    encounter = build_encounter(
        device_id=DEVICE_ID,
        started_us=started_us,
        ended_us=ended_us,
        faces=faces,
        speech=speech,
    )
    try:
        store.record_encounter(encounter, clock)
    except Exception as error:  # noqa: BLE001 - reported, never fatal
        print(f"[DB] FAILED to record encounter: {error}")
        return None
    print(
        f"[DB] encounter {encounter.id} [{encounter.status}] recorded: "
        f"{len(faces)} face, {len(speech)} speech observation(s)"
    )
    return encounter.id


def subject_of(tracker):
    """Who the conversation was with: the face seen in the most frames."""
    tracks = [t for t in tracker.tracks.values() if t.best_frame is not None]
    if not tracks:
        return None
    return identify_track(face_pipeline, store, max(tracks, key=lambda t: t.frame_count))


def learn(captures, tracker, audio_start_us, encounter_id):
    """Transcribe, pull out facts, and file them under the person in front.

    Someone new gets a profile (their face, and their name if they said it);
    someone known gets the new facts added to theirs. Either way the facts
    go to TigerData, and the OLED is updated: the saved facts for a new
    person, fresh talking points for a known one.
    """
    subject = subject_of(tracker)
    if subject is None or (not subject.known and subject.embedding is None):
        why = subject.note if subject else "none seen"
        print(f"[learn] no clear face this session ({why}); nothing saved.")
        return

    wearer = store.wearer()
    # Talking to the camera yourself: the wearer *is* the person in front,
    # so their speech is about the subject rather than set aside as the wearer's.
    solo = subject.known and wearer is not None and subject.person_id == wearer.id
    wearer_name = "the wearer" if solo or wearer is None else wearer.display_name
    label = subject.name if subject.known else "the other person"

    transcriber = GeminiTranscriber()
    utterances = []
    for capture in captures:
        text = (transcriber.transcribe(capture.audio) or "").strip()
        if not text:
            continue
        by_wearer = capture.is_wearer and not solo
        utterances.append(
            Utterance(
                text=text,
                start_s=us_to_seconds(capture.start_us - audio_start_us),
                end_s=us_to_seconds(capture.end_us - audio_start_us),
                person_id=None if by_wearer else subject.person_id,
                person_name=wearer_name if by_wearer else label,
                is_wearer=by_wearer,
            )
        )
    transcript = " ".join(u.text for u in utterances)
    print(f"[learn] heard: {transcript or '(nothing intelligible)'}")

    if encounter_id is None:
        encounter_id = store.open_encounter(DEVICE_ID, datetime.now(UTC))
    occurred_at = datetime.now(UTC).isoformat()
    facts = []
    if utterances:
        facts = GeminiExtractor().extract(
            utterances, encounter_id=encounter_id, wearer_name=wearer_name, occurred_at=occurred_at
        )
    facts = [
        f for f in facts
        if f.person_name == label or (subject.known and f.person_id == subject.person_id)
    ]

    if subject.known:
        person_id, name = subject.person_id, subject.name
    else:
        names = [f for f in facts if f.kind == "name" and f.value.strip()]
        name = max(names, key=lambda f: f.confidence).value.strip() if names else None
        person_id = store.enroll(
            name,
            [subject.embedding],
            source_encounter_id=encounter_id,
            source="vision_hub",
            consent_note="met in a recorded conversation",
        )
        face_pipeline.gallery = store.gallery("face")
        name = store.name_of(person_id)
        print(f"[learn] new person saved: {name}")

    facts = [replace(f, person_id=person_id, person_name=name) for f in facts]
    added = store.add_facts(encounter_id, person_id, facts)
    print(f"[DB] {added} new fact(s) saved for {name}:")
    for fact in facts:
        print(f"    {fact.kind:<12} {fact.value}")

    # Kept in step so `glasses ask` still answers from the same memories.
    save_encounter(
        profile_path(store),
        [replace(u, person_id=person_id, person_name=name) for u in utterances if not u.is_wearer],
        facts,
        encounter_id=encounter_id,
        occurred_at=occurred_at,
        location=os.environ.get("GLASSES_LOCATION"),
    )

    if coach is not None:
        if subject.known:
            coach.current = person_id
            coach.refresh(person_id, transcript)
        else:
            coach.show_facts(person_id, facts)


class LiveTalkingPoints:
    """While recording a known person, redo their talking points every
    ``LIVE_EVERY_S`` from the last ``LIVE_WINDOW_S`` of audio."""

    def __init__(self, recorder, person_id):
        self.recorder = recorder
        self.person_id = person_id
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        if coach is not None and self.person_id:
            coach.current = self.person_id
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()

    def _run(self):
        transcriber = GeminiTranscriber()
        while not self._stop.wait(LIVE_EVERY_S):
            samples = self.recorder.samples[-LIVE_WINDOW_S * SAMPLE_RATE:]
            try:
                transcript = transcriber.transcribe(samples)
            except GeminiError as error:
                print(f"[coach] live transcription failed: {error}")
                continue
            if transcript.strip() and not self._stop.is_set():
                coach.refresh(self.person_id, transcript)


def resolve_session(
    audio, audio_start_us, tracker, clock, started_us, ended_us, frame_width=None
):
    print("\n[Processing] Analyzing audio and binding to faces...")
    captures = []
    # A silent session is still an encounter: the faces seen are worth
    # recording even when nobody said anything the segmenter could use.
    result = voice_pipeline.process(audio) if audio.size > 0 else None
    for item in result.segments if result else []:
        if item.embedding is None:
            continue
        captures.append(
            Capture(
                start_us=audio_start_us + seconds_to_us(item.segment.start_s),
                end_us=audio_start_us + seconds_to_us(item.segment.end_s),
                audio=item.segment.samples,
                asd=None,
                track=None,
                is_wearer=item.is_wearer,
                speaker=item.segment.speaker,
                voice_person_id=item.match.person_id if item.match else None,
                voice_name=store.name_of(item.match.person_id)
                if item.match and item.match.person_id
                else None,
                voice_similarity=item.match.similarity if item.match else 0.0,
            )
        )

    if not captures:
        print("No usable speech detected (requires 3+ continuous seconds).")

    partners = [c for c in captures if not c.is_wearer]
    if partners:
        assign_speakers(
            partners, tracker, audio, audio_start_us,
            AsdConfig(min_correlation=0.30, min_margin=0.15),
        )

    identities = resolve_identities(tracker)

    for capture in captures:
        at = us_to_seconds(capture.start_us - audio_start_us)
        if capture.is_wearer:
            print(f"[{at:5.1f}s] WEARER (Audio: {capture.duration_s():.1f}s)")
            continue

        who = capture.voice_name or "unknown voice"
        line = f"[{at:5.1f}s] {who}"
        if capture.voice_similarity:
            line += f" (voice similarity: {capture.voice_similarity:.2f})"

        if capture.track is None:
            print(f"{line}  -> UNASSIGNED: {capture.asd.reason}")
            continue

        track = capture.track
        face_id = "new person"
        if track.id in identities:
            person_id, similarity = identities[track.id]
            face_id = f"{store.name_of(person_id)} ({similarity:.2f})"

        print(f"{line}  -> face #{track.id} [{face_id}]")

    # The encounter and its observation streams, in TigerData.
    encounter_id = record_session(
        tracker, captures, identities, clock, started_us, ended_us, frame_width
    )

    # Finish the encounter after CAM_OFF: transcribe, extract facts and file
    # them under the person in front. Needs speech to work with, so a silent
    # session is recorded above and stops here.
    if not captures or not records_encounters:
        return
    try:
        learn(captures, tracker, audio_start_us, encounter_id)
    except GeminiError as error:
        print(f"[learn] FAILED: {error}")


def record_session_mode():
    """Tap tap: record camera + mic until the next double tap, then save.

    The OLED is left alone the whole time (it keeps showing whoever was last
    in frame); progress goes to the terminal.
    """
    print("\n[mode] RECORDING: camera & mic on. Double-tap (or press Enter) to stop.")
    cap = open_camera(CAMERA_INDEX)

    clock = Clock.start()
    tracker = FaceTracker()
    frames_seen = 0
    detections = []
    frame_width = None
    # Whoever was recognised just before recording started.
    known = display.showing if display.showing not in (None, UNKNOWN) else None

    with AudioRecorder(clock, device=MIC) as recorder, LiveTalkingPoints(recorder, known):
        started = clock.now_us()
        prev_frame_time = 0

        while True:
            inner_cmd = next_command(recording=True)
            if inner_cmd == "CAM_OFF":
                print("[mode] recording stopped, saving meeting...")
                break
            elif inner_cmd:
                print(f"ESP32: {inner_cmd}")

            ret, frame = cap.read()
            if not ret:
                camera_failed()
                break

            frame_width = frame.shape[1]
            new_frame_time = time.time()
            time_diff = new_frame_time - prev_frame_time
            fps = int(1 / time_diff) if time_diff > 0 else 0
            prev_frame_time = new_frame_time

            at_us = clock.now_us()
            frames_seen += 1

            if frames_seen % 5 == 1 or not detections:
                detections = face_pipeline.detector.detect(frame)

            tracker.update(detections, frame, at_us)

            for track in tracker.active(at_us):
                if track.current_detection is not None:
                    sample = mouths.sample(frame, track.current_detection, at_us)
                    if sample is not None:
                        track.mouth.append(sample)

                box = track.bbox
                cv2.rectangle(
                    frame, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0, 200, 200), 2
                )
                cv2.putText(
                    frame,
                    f"Track #{track.id}",
                    (int(box.x1), max(16, int(box.y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 200),
                    2,
                )

            cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if PREVIEW:
                cv2.imshow("Multimodal Feed - double-tap to end", frame)
            if PREVIEW and cv2.waitKey(1) & 0xFF == ord("q"):
                break

    audio = recorder.samples
    audio_start_us = recorder.start_us or started
    ended = clock.now_us()
    cap.release()
    if PREVIEW:
        with suppress(cv2.error):
            cv2.destroyWindow("Multimodal Feed - double-tap to end")

    # Runs even with no audio: the faces seen are still an encounter,
    # and a silent one is the expired case the schema has a state for.
    resolve_session(audio, audio_start_us, tracker, clock, started, ended, frame_width=frame_width)
    # The profile of whoever was just met may have changed; re-send it.
    display.reset()


def identify_mode():
    """Always on: look for faces on every frame and show who is in front.

    Returns "CAM_ON" when a double tap asks to record, or None to quit.
    """
    print(
        "\n[mode] IDENTIFY: always looking for faces. "
        "Double-tap (or press Enter) to record, Ctrl-C to quit."
    )
    cap = open_camera(CAMERA_INDEX)
    prev_frame_time = 0
    failures = 0

    try:
        while True:
            inner_cmd = next_command(recording=False)
            if inner_cmd == "CAM_ON":
                return "CAM_ON"
            elif inner_cmd:
                print(f"ESP32: {inner_cmd}")  # single tap / CAM_OFF: nothing to do

            ret, frame = cap.read()
            if not ret:
                failures += 1
                if failures == 1:
                    camera_failed()
                # Keep trying: a replugged camera comes back on reopen.
                cap.release()
                time.sleep(1.0)
                cap = open_camera(CAMERA_INDEX)
                continue
            if failures:
                print("[camera] frames are back.")
                failures = 0

            new_frame_time = time.time()
            time_diff = new_frame_time - prev_frame_time
            fps = int(1 / time_diff) if time_diff > 0 else 0
            prev_frame_time = new_frame_time

            result = face_pipeline.process_frame(frame)
            display.update(result.faces)

            if PREVIEW:
                for face in result.faces:
                    box = face.detection.bbox
                    if face.match is not None and face.match.is_identified:
                        text = f"{store.name_of(face.match.person_id)} ({face.match.similarity:.2f})"
                        color = (0, 255, 0)
                    elif face.quality.passed:
                        text, color = "Unknown person", (0, 0, 255)
                    else:
                        text, color = "Face (low quality)", (0, 165, 255)
                    cv2.rectangle(
                        frame, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, 2
                    )
                    cv2.putText(
                        frame,
                        text,
                        (int(box.x1), max(16, int(box.y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )
                cv2.putText(
                    frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
                )
                cv2.imshow("Live Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return None
    finally:
        cap.release()
        if PREVIEW:
            with suppress(cv2.error):
                cv2.destroyWindow("Live Recognition")


try:
    while identify_mode() == "CAM_ON":
        record_session_mode()
except KeyboardInterrupt:
    pass

display.clear()
link.close()
