"""The capture loop: photograph whoever is speaking.

Speech is the trigger. When someone who is not the wearer talks, the
system works out which visible face is producing that sound, and captures
that face and that audio together as one observation of one person. A
stranger becomes known by the act of talking to you.

Binding therefore happens at capture time, not by reconciling two
independent streams afterwards. The hard part is picking the right face,
because the failure is unrecoverable: somebody's words filed against
somebody else's face, with nothing downstream aware it should doubt the
pairing. So when the speaker cannot be identified confidently the audio is
kept and left unassigned, which is what a nullable ``person_id`` is for.

Timing budget, measured on this laptop: face detection costs ~22 ms and
106-point landmarks ~1 ms per face. Detection every fifth frame with
landmarks on every frame averages ~6 ms, which holds 30 fps — and the
frame rate is not negotiable, since lip-sync needs to see a 4-7 Hz
rhythm.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .binding.asd import AsdConfig, AsdResult, pick_speaker
from .binding.mouth import MouthSample, mouth_openness
from .device import AudioFrame, EspSerialStream, TouchEvent
from .encounter.clock import Clock, seconds_to_us, us_to_seconds
from .face.types import BBox, Detection
from .voice.config import SAMPLE_RATE


def iou(a: BBox, b: BBox) -> float:
    """Overlap between two boxes, for carrying a track between detections."""
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class Track:
    """One face followed across frames.

    Tracking matters here in a way it did not for still recognition: the
    mouth signal is a *time series*, so the frames have to be known to
    belong to the same face. Identity is not available for strangers —
    which is most people — so the association is geometric.
    """

    id: int
    bbox: BBox
    last_seen_us: int
    mouth: list[MouthSample] = field(default_factory=list)
    #: Most recent detection, used for landmarks on every frame. Between
    #: detections its box is a few frames stale, which is what tracking is
    #: for and is harmless at 33 ms intervals.
    current_detection: Detection | None = None
    #: Best frame seen of this face, kept for the crop and the embedding.
    #: Ranked by face width, which at a fixed camera is the best cheap
    #: proxy for how much detail the embedder will get.
    best_frame: np.ndarray | None = None
    best_width: float = 0.0
    best_detection: Detection | None = None

    def observe(self, detection: Detection, frame: np.ndarray, at_us: int) -> None:
        self.bbox = detection.bbox
        self.current_detection = detection
        self.last_seen_us = at_us
        if detection.bbox.width > self.best_width:
            self.best_width = detection.bbox.width
            self.best_frame = frame.copy()
            self.best_detection = detection

    def samples_within(self, start_us: int, end_us: int) -> list[MouthSample]:
        return [s for s in self.mouth if start_us <= s.at_us <= end_us]


class FaceTracker:
    """Greedy IoU tracking. Enough for a conversation, where faces are
    large, few, and move slowly between frames 33 ms apart."""

    def __init__(self, *, iou_threshold: float = 0.3, drop_after_s: float = 1.0) -> None:
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self._iou_threshold = iou_threshold
        self._drop_after_us = seconds_to_us(drop_after_s)

    def update(self, detections: list[Detection], frame: np.ndarray, at_us: int) -> list[Track]:
        unclaimed = set(self.tracks)
        touched: list[Track] = []

        for detection in detections:
            best_id, best_score = None, self._iou_threshold
            for track_id in unclaimed:
                score = iou(self.tracks[track_id].bbox, detection.bbox)
                if score > best_score:
                    best_id, best_score = track_id, score

            if best_id is None:
                track = Track(self._next_id, detection.bbox, at_us)
                self.tracks[self._next_id] = track
                self._next_id += 1
            else:
                track = self.tracks[best_id]
                unclaimed.discard(best_id)

            track.observe(detection, frame, at_us)
            touched.append(track)

        for track_id in list(self.tracks):
            if at_us - self.tracks[track_id].last_seen_us > self._drop_after_us:
                # Keep the mouth history: an utterance may still need it.
                self.tracks[track_id].bbox = BBox(0, 0, 0, 0)
        return touched

    def active(self, at_us: int) -> list[Track]:
        return [t for t in self.tracks.values() if at_us - t.last_seen_us <= self._drop_after_us]


class AudioRecorder:
    """Microphone capture on the shared clock.

    The start timestamp is taken when the stream opens, and every sample
    offset is computed from it. That single number is what lines a
    sentence up against a face.
    """

    def __init__(
        self,
        clock: Clock,
        device: int | None = None,
        *,
        source: str = "laptop",
        esp_port: str | None = None,
        on_touch: Callable[[TouchEvent], None] | None = None,
    ) -> None:
        self.clock = clock
        self.device = device
        self.source = source
        self.esp_port = esp_port
        self.on_touch = on_touch
        self.start_us: int | None = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream = None
        self._serial: EspSerialStream | None = None
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: Exception | None = None
        self._device_start_us: int | None = None
        self._sample_count = 0

    def __enter__(self) -> AudioRecorder:
        if self.source == "esp":
            if self.device is not None:
                raise ValueError("--device selects a laptop input; use --esp-port for ESP audio")
            if not self.esp_port:
                raise ValueError("ESP audio needs --esp-port")
            self._serial = EspSerialStream(self.esp_port)
            self._serial.__enter__()
            self._reader = threading.Thread(target=self._read_esp, daemon=True)
            self._reader.start()
            return self
        if self.source != "laptop":
            raise ValueError(f"unknown audio source: {self.source}")
        if self.esp_port:
            raise ValueError("--esp-port requires --audio-source esp")
        from .voice.capture import _require_sounddevice

        sounddevice = _require_sounddevice()

        def callback(indata, frames, time_info, status):  # type: ignore[no-untyped-def]
            del frames, time_info, status
            with self._lock:
                self._chunks.append(indata[:, 0].copy())

        self._stream = sounddevice.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=callback,
        )
        self.start_us = self.clock.now_us()
        self._stream.start()
        return self

    def _read_esp(self) -> None:
        assert self._serial is not None
        try:
            while not self._stop.is_set():
                event = self._serial.read()
                if isinstance(event, TouchEvent):
                    if self.on_touch is not None:
                        self.on_touch(event)
                elif isinstance(event, AudioFrame):
                    with self._lock:
                        if self._device_start_us is None:
                            self._device_start_us = event.t_us
                            # The frame arrived after its samples were captured.
                            self.start_us = self.clock.now_us() - round(
                                event.samples.size / SAMPLE_RATE * 1_000_000
                            )
                        offset = round(
                            (event.t_us - self._device_start_us) * SAMPLE_RATE / 1_000_000
                        )
                        if offset < 0 or offset - self._sample_count > SAMPLE_RATE * 2:
                            raise ValueError("ESP audio timestamps jumped or the device restarted")
                        if offset > self._sample_count:
                            gap = offset - self._sample_count
                            self._chunks.append(np.zeros(gap, dtype=np.float32))
                            self._sample_count += gap
                        samples = event.samples[max(0, self._sample_count - offset) :]
                        if samples.size:
                            self._chunks.append(samples)
                            self._sample_count += samples.size
        except Exception as error:  # noqa: BLE001 - passed to the capture caller
            self._error = error

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._reader is not None:
            self._reader.join(timeout=1.0)
        if self._serial is not None:
            self._serial.__exit__(*exc)
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    @property
    def samples(self) -> np.ndarray:
        if self._error is not None:
            raise self._error
        with self._lock:
            return np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)

    def level_dbfs(self) -> float:
        from .voice.audio import rms_dbfs

        with self._lock:
            return rms_dbfs(self._chunks[-1]) if self._chunks else -120.0


@dataclass
class Capture:
    """One utterance, and the face the system decided produced it."""

    start_us: int
    end_us: int
    audio: np.ndarray
    asd: AsdResult
    track: Track | None
    #: Set once the voice gallery has been consulted.
    voice_person_id: str | None = None
    voice_name: str | None = None
    voice_similarity: float = 0.0
    is_wearer: bool = False
    face_person_id: str | None = None
    face_name: str | None = None
    face_similarity: float = 0.0
    image_path: Path | None = None
    text: str | None = None

    @property
    def assigned(self) -> bool:
        return self.track is not None

    def duration_s(self) -> float:
        return us_to_seconds(self.end_us - self.start_us)


class MouthReader:
    """106-point landmarks reduced to one number per face per frame."""

    def __init__(self, *, providers: list[str] | None = None) -> None:
        self._providers = providers
        self._model = None

    def _ensure_loaded(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            from insightface import model_zoo
            from insightface.utils import storage

            from .face.models import default_providers

            pack = storage.ensure_available("models", "buffalo_l")
            self._model = model_zoo.get_model(
                f"{pack}/2d106det.onnx", providers=self._providers or default_providers()
            )
            self._model.prepare(ctx_id=0)
        return self._model

    def sample(self, frame: np.ndarray, detection: Detection, at_us: int) -> MouthSample | None:
        from insightface.app.common import Face

        from .face.quality import head_pose

        model = self._ensure_loaded()
        box = detection.bbox
        face = Face(
            bbox=np.array([box.x1, box.y1, box.x2, box.y2], dtype=np.float32),
            kps=detection.landmarks.as_array().astype(np.float32),
            det_score=detection.score,
        )
        try:
            points = np.asarray(model.get(frame, face))
        except Exception:  # noqa: BLE001 - a bad box costs one frame, not the run
            return None

        roll = head_pose(detection.landmarks).roll
        openness, width = mouth_openness(points, roll_deg=roll)
        return MouthSample(at_us=at_us, openness=openness, width_px=width)


def assign_speakers(
    captures: list[Capture],
    tracker: FaceTracker,
    audio: np.ndarray,
    audio_start_us: int,
    config: AsdConfig | None = None,
) -> None:
    """Decide, for each utterance, which tracked face was speaking.

    Mutates each capture in place with the verdict. Called after capture
    rather than during it because the mouth series for an utterance is
    only complete once the utterance has ended.
    """
    for capture in captures:
        candidates = {
            str(track.id): samples
            for track in tracker.tracks.values()
            if (samples := track.samples_within(capture.start_us, capture.end_us))
        }
        result = pick_speaker(candidates, capture.audio, capture.start_us, config)
        capture.asd = result
        if result.speaker_key is not None:
            capture.track = tracker.tracks[int(result.speaker_key)]


def save_crop(frame: np.ndarray, bbox: BBox, path: Path, *, margin: float = 0.4) -> Path:
    """Write the chosen face, with context around it.

    The margin is for the human checking the result: a tight crop of the
    wrong person is much harder to recognise as wrong.
    """
    import cv2

    height, width = frame.shape[:2]
    pad = bbox.width * margin
    x1, y1 = max(0, int(bbox.x1 - pad)), max(0, int(bbox.y1 - pad))
    x2, y2 = min(width, int(bbox.x2 + pad)), min(height, int(bbox.y2 + pad))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame[y1:y2, x1:x2])
    return path
