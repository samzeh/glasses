"""The capture loop's logic, without a camera.

Tracking and speaker assignment are tested on synthetic frames and
synthetic mouths. What cannot be tested here is whether a real mouth on a
real webcam produces a usable correlation — that needs footage, and is
what ``glasses meet`` exists to find out.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from glasses.binding.mouth import MouthSample
from glasses.encounter.clock import seconds_to_us
from glasses.face.types import BBox, Detection, Landmarks5
from glasses.meet import Capture, FaceTracker, Track, assign_speakers, iou, save_crop
from glasses.voice.config import SAMPLE_RATE

FRAME_US = 1_000_000 // 30


def box(x: float, y: float, size: float = 100.0) -> BBox:
    return BBox(x, y, x + size, y + size)


def detection(bbox: BBox, score: float = 0.9) -> Detection:
    cx, cy = bbox.center
    w = bbox.width
    return Detection(
        bbox=bbox,
        landmarks=Landmarks5(
            left_eye=(cx - w * 0.2, cy - w * 0.1),
            right_eye=(cx + w * 0.2, cy - w * 0.1),
            nose=(cx, cy),
            mouth_left=(cx - w * 0.15, cy + w * 0.2),
            mouth_right=(cx + w * 0.15, cy + w * 0.2),
        ),
        score=score,
    )


def frame(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (480, 640, 3), dtype=np.uint8)


# --- IoU and tracking -----------------------------------------------------


def test_iou_is_one_for_the_same_box_and_zero_for_disjoint():
    assert iou(box(0, 0), box(0, 0)) == pytest.approx(1.0)
    assert iou(box(0, 0), box(500, 0)) == 0.0


def test_a_face_keeps_its_track_id_as_it_drifts():
    """The mouth signal is a time series, so frames must be known to
    belong to the same face. For strangers there is no identity to use, so
    the association is geometric."""
    tracker = FaceTracker()
    tracker.update([detection(box(100, 100))], frame(), 0)
    tracker.update([detection(box(108, 104))], frame(), FRAME_US)

    assert len(tracker.tracks) == 1
    assert tracker.tracks[1].mouth == []


def test_two_faces_get_two_tracks():
    tracker = FaceTracker()

    tracker.update([detection(box(50, 100)), detection(box(400, 100))], frame(), 0)

    assert len(tracker.tracks) == 2


def test_a_face_that_jumps_across_the_frame_becomes_a_new_track():
    """Better a split track than two people merged into one mouth series."""
    tracker = FaceTracker()
    tracker.update([detection(box(50, 100))], frame(), 0)
    tracker.update([detection(box(500, 100))], frame(), FRAME_US)

    assert len(tracker.tracks) == 2


def test_a_track_goes_inactive_when_the_face_leaves():
    tracker = FaceTracker(drop_after_s=0.5)
    tracker.update([detection(box(100, 100))], frame(), 0)

    assert len(tracker.active(0)) == 1
    assert tracker.active(seconds_to_us(2)) == []


def test_mouth_history_outlives_the_track_going_inactive():
    """An utterance that ended as the person turned away still needs the
    frames from while they were speaking."""
    tracker = FaceTracker(drop_after_s=0.5)
    tracker.update([detection(box(100, 100))], frame(), 0)
    tracker.tracks[1].mouth.append(MouthSample(0, 0.3, 40.0))

    tracker.update([], frame(), seconds_to_us(5))

    assert tracker.tracks[1].mouth != []


def test_the_best_frame_is_the_one_with_the_largest_face():
    """Sizes drift the way a face does over 167 ms, not by doubling."""
    tracker = FaceTracker()
    tracker.update([detection(box(100, 100, size=60))], frame(1), 0)
    tracker.update([detection(box(100, 100, size=90))], frame(2), FRAME_US)
    tracker.update([detection(box(100, 100, size=70))], frame(3), 2 * FRAME_US)

    assert len(tracker.tracks) == 1
    assert tracker.tracks[1].best_width == 90.0


def test_a_face_that_changes_size_wildly_splits_rather_than_merging():
    """A box jumping from 60px to 160px between frames is not the same
    face moving — it is more likely a different face, and splitting the
    track is safer than blending two mouths into one series."""
    tracker = FaceTracker()
    tracker.update([detection(box(100, 100, size=60))], frame(1), 0)
    tracker.update([detection(box(100, 100, size=160))], frame(2), FRAME_US)

    assert len(tracker.tracks) == 2


# --- assigning the speaker ------------------------------------------------


def talking_samples(start_us: int, count: int, *, rate_hz: float = 5.0) -> list[MouthSample]:
    return [
        MouthSample(
            at_us=start_us + i * FRAME_US,
            openness=0.3 + 0.2 * math.sin(2 * math.pi * rate_hz * (i * FRAME_US / 1e6)),
            width_px=40.0,
        )
        for i in range(count)
    ]


def still_samples(start_us: int, count: int) -> list[MouthSample]:
    rng = np.random.default_rng(0)
    return [
        MouthSample(start_us + i * FRAME_US, 0.15 + 0.0004 * rng.normal(), 40.0)
        for i in range(count)
    ]


def speech(duration_s: float, *, rate_hz: float = 5.0) -> np.ndarray:
    n = int(SAMPLE_RATE * duration_s)
    t = np.arange(n) / SAMPLE_RATE
    envelope = 0.5 + 0.45 * np.sin(2 * math.pi * rate_hz * t)
    return (envelope * np.random.default_rng(1).normal(size=n) * 0.2).astype(np.float32)


def capture(start_us: int, duration_s: float) -> Capture:
    return Capture(
        start_us=start_us,
        end_us=start_us + seconds_to_us(duration_s),
        audio=speech(duration_s),
        asd=None,  # type: ignore[arg-type]
        track=None,
    )


def tracker_with(**mouths: list[MouthSample]) -> FaceTracker:
    tracker = FaceTracker()
    for index, (_, samples) in enumerate(mouths.items(), start=1):
        track = Track(index, box(index * 150, 100), samples[-1].at_us)
        track.mouth = samples
        tracker.tracks[index] = track
    return tracker


def test_the_talking_face_is_the_one_captured():
    """The whole point: with two people in frame, photograph the speaker."""
    tracker = tracker_with(talker=talking_samples(0, 60), listener=still_samples(0, 60))
    item = capture(0, 2.0)

    assign_speakers([item], tracker, speech(2.0), 0)

    assert item.track is not None
    assert item.track.id == 1


def test_the_listener_is_not_captured():
    tracker = tracker_with(listener=still_samples(0, 60), talker=talking_samples(0, 60))
    item = capture(0, 2.0)

    assign_speakers([item], tracker, speech(2.0), 0)

    assert item.track.id == 2


def test_an_ambiguous_utterance_is_left_unassigned():
    """Unassigned audio can be resolved by a later encounter. Audio filed
    against the wrong face cannot, because nothing knows to doubt it."""
    tracker = tracker_with(a=talking_samples(0, 60), b=talking_samples(0, 60))
    item = capture(0, 2.0)

    assign_speakers([item], tracker, speech(2.0), 0)

    assert item.track is None
    assert not item.assigned
    assert "alike" in item.asd.reason


def test_only_frames_inside_the_utterance_are_considered():
    """A face that was talking before the utterance started should not win
    on the strength of frames outside the window."""
    early = talking_samples(0, 60)
    during = talking_samples(seconds_to_us(5), 60)
    tracker = tracker_with(earlier=early, current=during)

    item = capture(seconds_to_us(5), 2.0)
    assign_speakers([item], tracker, speech(2.0), seconds_to_us(5))

    assert item.track is not None
    assert item.track.id == 2


def test_no_faces_at_all_leaves_it_unassigned():
    item = capture(0, 2.0)

    assign_speakers([item], FaceTracker(), speech(2.0), 0)

    assert item.track is None
    assert "no face" in item.asd.reason


def test_every_utterance_gets_its_own_verdict():
    tracker = tracker_with(a=talking_samples(0, 200), b=still_samples(0, 200))
    first, second = capture(0, 2.0), capture(seconds_to_us(3), 2.0)

    assign_speakers([first, second], tracker, speech(6.0), 0)

    assert first.asd is not None
    assert second.asd is not None


# --- the saved crop -------------------------------------------------------


def test_the_crop_is_written_with_margin_for_the_human_checking_it(tmp_path):
    """A tight crop of the wrong person is much harder to recognise as
    wrong, and recognising it is the entire point of saving it."""
    path = save_crop(frame(), box(200, 150, size=100), tmp_path / "001.jpg")

    import cv2

    written = cv2.imread(str(path))
    assert path.exists()
    assert written.shape[0] > 100


def test_a_crop_at_the_frame_edge_is_clamped(tmp_path):
    path = save_crop(frame(), box(0, 0, size=80), tmp_path / "edge.jpg")

    assert path.exists()
