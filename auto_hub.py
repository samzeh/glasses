"""Automatic encounter capture. Nothing to press.

The camera runs continuously and the phone's microphone reports voice
activity. When a centred face has dwelled and speech has been heard, the
encounter is confirmed and both halves record; when the conversation ends it
closes, the encounter goes to TigerData, and the video is kept. An encounter
that never became a conversation expires and its video is deleted.

The trigger policy is ``glasses.encounter.session``; this file is the camera
loop that feeds it and the side effects that follow from it. Run it alongside
the Voice Notes server, which proxies the phone's voice activity in and the
trigger state back out:

    python auto_hub.py
    cd voice-notes && npm run dev

Then open the Voice Notes page on the phone and tap Auto. One tap is needed
because a browser will not open a microphone without a gesture; after it,
nothing is pressed again.

The audio itself never passes through here. The phone records it, the Voice
Notes server sends it to Gemini, and the note is written against the same
encounter id this hub created — which is the whole reason the trigger lives
on one side instead of being implemented twice.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2

from glasses import env
from glasses.device import LcdNotifier, glasses_camera, is_centred, open_camera
from glasses.encounter.auto import AutoTrigger, FaceSample
from glasses.encounter.bridge import PORT, TriggerBridge
from glasses.encounter.clock import Clock
from glasses.encounter.session import TriggerConfig
from glasses.face import ArcFaceEmbedder, FacePipeline, ScrfdDetector
from glasses.face.models import default_providers
from glasses.meet import FaceTracker
from glasses.storage import open_store, store_label

env.load()

DEVICE_ID = os.environ.get("GLASSES_DEVICE_ID") or "glasses-auto"
#: Same names vision_hub.py and the README use, so one rig has one set of
#: variables rather than two that drift.
CAMERA_INDEX = glasses_camera(
    int(os.environ["GLASSES_CAMERA_INDEX"]) if os.environ.get("GLASSES_CAMERA_INDEX") else None
)
#: Where session video goes. ``data/`` is gitignored for exactly this reason.
SESSION_DIR = Path(os.environ.get("GLASSES_SESSIONS", "data/sessions"))
#: Detection is the expensive call; tracking carries the boxes between.
DETECT_EVERY = 5
#: Recognition is more expensive still, and an identity cannot change
#: mid-track, so it is resolved once per track and cached.
RECOGNISE_EVERY = 15
#: Nominal rate written into the video container.
VIDEO_FPS = 15.0
#: Off by default: a pair of glasses has no monitor, and a GUI-less OpenCV
#: build raises rather than ignoring imshow.
PREVIEW = os.environ.get("GLASSES_PREVIEW") == "1"


def build_pipeline(store):
    providers = default_providers()
    return FacePipeline(
        detector=ScrfdDetector(providers=providers),
        embedder=ArcFaceEmbedder(providers=providers),
        gallery=store.gallery("face"),
    )


class VideoSink:
    """Writes one encounter's frames, and throws them away if it expires.

    The discard path is the point. A provisional encounter is recorded
    speculatively, because by the time it is confirmed the interesting
    seconds have already happened — and most are never confirmed, so most of
    what is written here is deleted. Doing that honestly matters more than
    saving the writes.
    """

    def __init__(self, directory: Path, fps: float = VIDEO_FPS) -> None:
        self.directory = directory
        self.fps = fps
        self._writer = None
        self._path: Path | None = None
        self.encounter_id: str | None = None

    @property
    def open(self) -> bool:
        return self._writer is not None

    def start(self, encounter_id: str, frame) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        path = self.directory / f"{encounter_id}.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
        )
        if not writer.isOpened():
            print(f"[video] could not open {path}; continuing without video")
            return
        self._writer, self._path, self.encounter_id = writer, path, encounter_id

    def write(self, frame) -> None:
        if self._writer is not None:
            self._writer.write(frame)

    def finish(self, *, keep: bool) -> Path | None:
        """Close the file, returning its path when it was kept."""
        if self._writer is None:
            return None
        self._writer.release()
        path, self._writer, self.encounter_id = self._path, None, None
        self._path = None
        if path is None:
            return None
        if keep:
            return path
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            print(f"[video] could not delete {path}: {error}")
        return None


def sample_faces(tracker, pipeline, identities, frame, at_us, frames_seen):
    """The frame's faces, as the trigger wants them.

    Identity is cached per track: the embedder is the expensive call and the
    answer cannot change while the same face is being followed.
    """
    active = tracker.active(at_us)
    if not active:
        return []

    widest = max(track.best_width for track in active)
    samples = []
    for track in active:
        if track.id not in identities and frames_seen % RECOGNISE_EVERY == 0:
            identities[track.id] = _recognise(pipeline, track)
        person_id, confidence = identities.get(track.id, (None, 0.0))
        box = track.bbox
        samples.append(
            FaceSample(
                person_id=person_id,
                confidence=confidence,
                centred=is_centred(box.x1, box.x2, frame.shape[1]),
                is_primary=track.best_width >= widest,
            )
        )
    return samples


def _recognise(pipeline, track):
    if track.best_frame is None:
        return (None, 0.0)
    result = pipeline.process_frame(track.best_frame)
    if not result.faces:
        return (None, 0.0)
    best = max(result.faces, key=lambda f: f.detection.bbox.width)
    if best.quality.passed and best.match and best.match.person_id:
        return (best.match.person_id, float(best.match.similarity))
    return (None, 0.0)


def status_line(state) -> str:
    if state.confirmed:
        return "RECORDING"
    if state.active:
        return "Listening..."
    return "Ready"


def main() -> int:
    print("Loading face recognition models...")
    store = open_store()
    print(f"Store: {store_label()}")
    if not hasattr(store, "record_encounter"):
        print("No DATABASE_URL: encounters cannot be recorded. Set it in .env and restart.")
        return 1

    pipeline = build_pipeline(store)
    clock = Clock.start()
    trigger = AutoTrigger(clock, device_id=DEVICE_ID, config=TriggerConfig())
    tracker = FaceTracker()
    sink = VideoSink(SESSION_DIR)
    lcd = LcdNotifier()
    if lcd.connected:
        print("Glasses display connected; status will show on the LCD.")

    camera = open_camera(CAMERA_INDEX)
    if not camera.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}. Set GLASSES_CAMERA_INDEX to another index.")
        return 1

    identities: dict[int, tuple[str | None, float]] = {}
    detections: list = []
    frames_seen = 0

    with TriggerBridge(trigger) as bridge:
        host, port = bridge.address
        print(f"Trigger bridge on http://{host}:{port} (Voice Notes proxies to it)")
        if port != PORT:
            print(f"NOTE: not the default port {PORT}; set TRIGGER_URL for the web server.")
        print("Watching. Talk to someone in front of the camera; nothing to press.\n")

        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Camera returned no frame; stopping.")
                    break
                at_us = clock.now_us()
                frames_seen += 1

                if frames_seen % DETECT_EVERY == 1 or not detections:
                    detections = pipeline.detector.detect(frame)
                tracker.update(detections, frame, at_us)

                faces = sample_faces(tracker, pipeline, identities, frame, at_us, frames_seen)
                state = trigger.observe_frame(faces, at_us)

                # Record from the moment an encounter opens, not from
                # confirmation: by then the first seconds have gone.
                if state.active and not sink.open:
                    sink.start(state.encounter_id, frame)
                if sink.open:
                    sink.write(frame)

                finished = trigger.take_finished()
                if finished is not None:
                    _settle(store, clock, sink, finished)
                    identities.clear()

                lcd.show(status_line(state))
                if PREVIEW:
                    _draw(frame, tracker, at_us, state)
                    cv2.imshow("Auto capture - q to quit", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            leftover = trigger.close()
            if leftover is not None:
                _settle(store, clock, sink, leftover)
            elif sink.open:
                sink.finish(keep=False)
            camera.release()
            lcd.close()
            if PREVIEW:
                cv2.destroyAllWindows()
    return 0


def _settle(store, clock, sink, encounter) -> None:
    """Write a finished encounter, and keep or discard its video."""
    kept = None
    if sink.open and sink.encounter_id == encounter.id:
        kept = sink.finish(keep=str(encounter.status) == "closed")

    if str(encounter.status) == "expired":
        print(f"[trigger] {encounter.id[:8]} expired; never a conversation, video discarded")
        return

    try:
        store.record_encounter(encounter, clock)
    except Exception as error:  # noqa: BLE001 - reported, never fatal
        print(f"[DB] FAILED to record encounter {encounter.id[:8]}: {error}")
        return
    print(
        f"[DB] encounter {encounter.id[:8]} [{encounter.status}] recorded: "
        f"{len(encounter.faces)} face, {len(encounter.speech)} speech observation(s)"
        + (f", video {kept}" if kept else "")
    )


def _draw(frame, tracker, at_us, state) -> None:
    for track in tracker.active(at_us):
        box = track.bbox
        colour = (0, 220, 0) if state.confirmed else (0, 200, 200)
        cv2.rectangle(frame, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), colour, 2)
    label = status_line(state)
    if state.active and not state.confirmed:
        label += f"  dwell={'ok' if state.dwell_satisfied else '..'}"
        label += f" speech={'ok' if state.heard_speech else '..'}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


if __name__ == "__main__":
    sys.exit(main())
