"""Show your face to the camera.

Known face: prints everything TigerData has on that person.
Unknown face: listens to what they say, pulls out facts, and saves the person
(face + facts) to TigerData so they are recognised next time.

    .venv/bin/python who.py
"""

from __future__ import annotations

import select
import socket
import sys
import time
import uuid
from datetime import UTC, datetime

from glasses.device import GLASSES_CAMERA, GlassesLink, find_camera_index, open_camera
from glasses.encounter.clock import Clock
from glasses.env import load as load_env
from glasses.face import ArcFaceEmbedder, FacePipeline, ScrfdDetector
from glasses.face.models import default_providers
from glasses.facts import DEFAULT_MODEL, GeminiExtractor
from glasses.meet import AudioRecorder
from glasses.storage import open_store
from glasses.voice.gemini import GeminiTranscriber
from glasses.voice.transcribe import Utterance

load_env()

#: Consecutive good frames needed before trusting what the camera sees.
STABLE_FRAMES = 8
#: Longest a conversation is listened to; press Enter to stop sooner.
MAX_LISTEN_S = 40


def look(cap, pipeline):
    """Block until a face has been seen steadily; return (match, embedding)."""
    streak, last = 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            sys.exit("Camera gave no frame. Allow the terminal under System Settings > Privacy > Camera.")
        result = pipeline.process_frame(frame)
        good = [f for f in result.faces if f.quality.passed and f.embedding is not None]
        if good:
            last = max(good, key=lambda f: f.detection.bbox.width)
            streak += 1
        else:
            streak = 0
        if streak >= STABLE_FRAMES:
            return last


def oled(link, name, when="-", where="-", fact="-"):
    """One profile line for the glasses; no-op without the glasses."""
    if link is None:
        return
    fields = [" ".join(str(v).split()) for v in (name, when, where, fact)]
    link.write(("PROFILE:" + "\t".join(fields) + "\n").encode("ascii", "replace"))


def print_person(store, person_id, link):
    conn = store.connection
    name = store.name_of(person_id)
    rows = conn.execute(
        "SELECT kind, value, extracted_at FROM fact WHERE person_id = %s "
        "AND superseded_by IS NULL ORDER BY kind, extracted_at",
        (person_id,),
    ).fetchall()
    shown = [(k, v) for k, v, _ in rows if k != "name"]
    oled(
        link, name,
        min(r[2] for r in rows).strftime("%b %d") if rows else "-",
        "-", f"{shown[0][0]}: {shown[0][1]}" if shown else "No facts yet",
    )
    print(f"\nI know you: {name}")
    if not rows:
        print("  (no facts saved about you yet)")
    for kind, value, _ in rows:
        print(f"  {kind}: {value}")


def listen():
    print(f"\nI don't know you yet. Say hi and tell me about yourself. Press Enter when done (max {MAX_LISTEN_S}s).")
    clock = Clock.start()
    with AudioRecorder(clock, device=GLASSES_CAMERA) as recorder:
        end = time.monotonic() + MAX_LISTEN_S
        while time.monotonic() < end:
            if select.select([sys.stdin], [], [], 0.2)[0]:
                if sys.stdin.readline():
                    break
                time.sleep(0.2)  # stdin closed (no terminal): just listen for the full time
    return recorder.samples


def learn(store, link, face, samples):
    print("Working out who you are...")
    text = GeminiTranscriber().transcribe(samples)
    if not text.strip():
        print("I didn't catch anything. Try again.")
        return
    started = datetime.now(UTC)
    encounter_id = str(uuid.uuid4())
    facts = GeminiExtractor(model=DEFAULT_MODEL).extract(
        [Utterance(text=text, start_s=0.0, end_s=len(samples) / 16000, speaker=0)],
        encounter_id=encounter_id,
        wearer_name="the camera operator",
        occurred_at=started.isoformat(),
    )
    names = [f for f in facts if f.kind == "name" and f.value.strip()]
    name = max(names, key=lambda f: f.confidence).value.strip() if names else None

    store.open_encounter(socket.gethostname(), started, encounter_id=encounter_id)
    person_id = store.enroll(
        name, [face.embedding], quality_scores=[1.0],
        source_encounter_id=encounter_id, source="who.py",
        consent_note="introduced themselves to the camera",
    )
    for f in facts:
        store.connection.execute(
            "INSERT INTO fact (person_id, encounter_id, kind, value, evidence_span, confidence) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (person_id, encounter_id, f.kind, f.value, f.evidence_span, f.confidence),
        )
    other = [f for f in facts if f.kind != "name"]
    oled(link, name or "New person", started.strftime("%b %d"), "-",
         f"{other[0].kind}: {other[0].value}" if other else "-")
    print(f"\nSaved to TigerData: {name or 'unnamed person'}")
    for f in facts:
        print(f"  {f.kind}: {f.value}")


def main():
    index = find_camera_index()
    if index is None:
        print(f"No {GLASSES_CAMERA} camera found; using camera 0.")
        index = 0
    print("Loading models...")
    store = open_store()
    if not hasattr(store, "connection"):
        sys.exit("DATABASE_URL is not set, so there is no TigerData to use.")
    providers = default_providers()
    pipeline = FacePipeline(
        detector=ScrfdDetector(providers=providers),
        embedder=ArcFaceEmbedder(providers=providers),
        gallery=store.gallery("face"),
    )
    try:
        link = GlassesLink()
    except Exception as error:  # noqa: BLE001
        print(f"No glasses display ({error}); printing to the terminal only.")
        link = None
    cap = open_camera(index)
    print("Put your face in front of the camera.")
    face = look(cap, pipeline)
    cap.release()

    if face.match is not None and face.match.is_identified:
        print_person(store, face.match.person_id, link)
    else:
        if link is not None:
            link.write(b"STATUS:Unknown person\n")
        learn(store, link, face, listen())
    if link is not None:
        time.sleep(1)
        link.close()


if __name__ == "__main__":
    main()
