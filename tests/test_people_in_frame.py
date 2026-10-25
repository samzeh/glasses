"""Counting the people in front of the camera.

The wearer is behind it and never in frame, so a face is always somebody
else — which makes this a census the audio cannot produce.
"""

from __future__ import annotations

from glasses.binding.mouth import MouthSample
from glasses.face.types import BBox
from glasses.meet import FaceTracker, Track, people_in_frame


def track(track_id: int, start_us: int, end_us: int, samples: int = 20) -> Track:
    """A track whose mouth history spans a given interval."""
    item = Track(track_id, BBox(0.0, 0.0, 10.0, 10.0), end_us)
    if samples:
        step = (end_us - start_us) / max(1, samples - 1)
        item.mouth = [
            MouthSample(at_us=int(start_us + step * i), openness=0.1, width_px=40.0)
            for i in range(samples)
        ]
    return item


def tracker_with(*tracks: Track) -> FaceTracker:
    tracker = FaceTracker()
    tracker.tracks = {t.id: t for t in tracks}
    return tracker


def test_an_empty_frame_has_nobody():
    assert people_in_frame(tracker_with()) == 0


def test_one_face_is_one_person():
    assert people_in_frame(tracker_with(track(1, 0, 5_000_000))) == 1


def test_two_faces_at_once_are_two_people():
    assert people_in_frame(tracker_with(track(1, 0, 5_000_000), track(2, 0, 5_000_000))) == 2


def test_one_person_re_found_is_still_one_person():
    """Greedy IoU tracking starts a new id whenever a face is lost and
    re-found, so someone who turns away twice becomes three ids — but
    never three at once."""
    churn = tracker_with(
        track(1, 0, 2_000_000),
        track(5, 3_000_000, 5_000_000),
        track(9, 6_000_000, 8_000_000),
    )
    assert people_in_frame(churn) == 1


def test_overlap_is_what_counts_not_the_number_of_ids():
    mixed = tracker_with(
        track(1, 0, 4_000_000),
        track(2, 2_000_000, 6_000_000),  # overlaps 1
        track(3, 9_000_000, 12_000_000),  # alone later
    )
    assert people_in_frame(mixed) == 2


def test_tracks_that_merely_touch_do_not_overlap():
    """A face lost at exactly the moment another is found is one person
    handed from one id to the next, not two people."""
    touching = tracker_with(track(1, 0, 5_000_000), track(2, 5_000_000, 9_000_000))
    assert people_in_frame(touching) == 1


def test_flickers_are_ignored():
    """A detection or two creates ids that never belonged to a face
    anyone was talking to."""
    noisy = tracker_with(
        track(1, 0, 5_000_000),
        track(2, 1_000_000, 1_100_000, samples=2),
        track(3, 1_000_000, 1_100_000, samples=0),
    )
    assert people_in_frame(noisy) == 1


def test_the_flicker_threshold_is_adjustable():
    brief = tracker_with(track(1, 0, 5_000_000), track(2, 0, 5_000_000, samples=3))
    assert people_in_frame(brief) == 1
    assert people_in_frame(brief, min_samples=2) == 2


def test_three_people_at_once():
    crowd = tracker_with(
        track(1, 0, 9_000_000),
        track(2, 1_000_000, 8_000_000),
        track(3, 2_000_000, 7_000_000),
    )
    assert people_in_frame(crowd) == 3
