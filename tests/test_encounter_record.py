"""Turning a finished capture session into an ``Encounter``.

The step that was missing between the hub and Tiger. Pure translation —
captures and tracks in, observations out — so the shape that reaches the
hypertables can be stated here instead of inferred from a live camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from glasses.encounter import EncounterStatus, seconds_to_us
from glasses.encounter.record import (
    build_encounter,
    face_observations,
    speech_observations,
)

PERSON = "913effeb-c21e-4aa4-8583-a5cc75f42d65"
OTHER = "0c9f1d22-5d24-4c36-9f1a-2e0f6d7b1a44"


@dataclass
class FakeBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1


@dataclass
class FakeTrack:
    id: int
    first_seen_us: int
    last_seen_us: int
    frame_count: int = 1
    best_width: float = 100.0
    bbox: FakeBox = field(default_factory=lambda: FakeBox(0, 0, 100, 100))
    best_frame: object | None = object()


@dataclass
class FakeCapture:
    start_us: int
    end_us: int
    is_wearer: bool = False
    voice_person_id: str | None = None
    voice_similarity: float = 0.0
    speaker: int | None = None
    track: FakeTrack | None = None
    text: str | None = None


def capture(start_s, end_s, **kwargs):
    return FakeCapture(seconds_to_us(start_s), seconds_to_us(end_s), **kwargs)


def track(track_id, first_s, last_s, **kwargs):
    return FakeTrack(track_id, seconds_to_us(first_s), seconds_to_us(last_s), **kwargs)


# --- speech ---------------------------------------------------------------


def test_speech_observation_carries_the_span_and_the_identity():
    [observation] = speech_observations(
        [capture(1.0, 4.0, voice_person_id=PERSON, voice_similarity=0.81)]
    )
    assert observation.span.start_us == seconds_to_us(1.0)
    assert observation.span.end_us == seconds_to_us(4.0)
    assert observation.person_id == PERSON
    assert observation.confidence == pytest.approx(0.81)
    assert not observation.is_wearer


def test_wearer_speech_is_kept_but_marked():
    """The wearer's own speech is not a partner, and the trigger needs it."""
    [observation] = speech_observations([capture(0.0, 2.0, is_wearer=True)])
    assert observation.is_wearer
    assert not observation.is_partner


def test_unrecognised_voice_has_no_person_but_keeps_its_speaker_label():
    [observation] = speech_observations([capture(0.0, 2.0, speaker=3)])
    assert observation.person_id is None
    assert observation.speaker == 3


def test_the_speaker_label_falls_back_to_the_bound_face_track():
    """Lip-sync assigned a face; that track id is the only label there is."""
    [observation] = speech_observations([capture(0.0, 2.0, track=track(7, 0.0, 2.0))])
    assert observation.speaker == 7


# --- faces ----------------------------------------------------------------


def test_face_observation_spans_first_to_last_sighting():
    [observation] = face_observations([track(1, 2.0, 9.0, frame_count=14)], {})
    assert observation.span.start_us == seconds_to_us(2.0)
    assert observation.span.end_us == seconds_to_us(9.0)
    assert observation.frame_count == 14
    assert observation.person_id is None


def test_identified_faces_take_their_person_and_confidence():
    [observation] = face_observations([track(1, 0.0, 3.0)], {1: (PERSON, 0.74)})
    assert observation.person_id == PERSON
    assert observation.confidence == pytest.approx(0.74)


def test_the_widest_face_is_the_primary_one():
    observations = face_observations(
        [track(1, 0.0, 3.0, best_width=40.0), track(2, 0.0, 3.0, best_width=180.0)], {}
    )
    primary = [o.is_primary for o in observations]
    assert primary == [False, True]


def test_a_lone_face_is_primary():
    [observation] = face_observations([track(1, 0.0, 3.0)], {})
    assert observation.is_primary


def test_centred_is_measured_against_the_frame_when_one_is_given():
    middle = track(1, 0.0, 1.0, bbox=FakeBox(270, 100, 370, 200))
    edge = track(2, 0.0, 1.0, bbox=FakeBox(0, 100, 60, 200))
    observations = face_observations([middle, edge], {}, frame_width=640)
    assert [o.centred for o in observations] == [True, False]


def test_tracks_never_actually_seen_are_dropped():
    """A track with no frames is tracker bookkeeping, not an observation."""
    assert face_observations([track(1, 0.0, 0.0, frame_count=0)], {}) == []


# --- the encounter --------------------------------------------------------


def test_partner_speech_closes_the_encounter():
    encounter = build_encounter(
        device_id="glasses-01",
        started_us=0,
        ended_us=seconds_to_us(30),
        faces=face_observations([track(1, 1.0, 20.0)], {1: (PERSON, 0.8)}),
        speech=speech_observations([capture(2.0, 6.0, voice_person_id=PERSON)]),
    )
    assert encounter.status is EncounterStatus.CLOSED
    assert encounter.device_id == "glasses-01"
    assert encounter.ended_us == seconds_to_us(30)
    assert len(encounter.faces) == 1
    assert len(encounter.speech) == 1


def test_an_encounter_with_only_the_wearer_talking_expires():
    """Nobody else spoke, so there was no conversation to keep."""
    encounter = build_encounter(
        device_id="glasses-01",
        started_us=0,
        ended_us=seconds_to_us(10),
        faces=[],
        speech=speech_observations([capture(0.0, 4.0, is_wearer=True)]),
    )
    assert encounter.status is EncounterStatus.EXPIRED


def test_each_encounter_gets_a_uuid_the_database_will_accept():
    import uuid

    one = build_encounter(device_id="d", started_us=0, ended_us=1, faces=[], speech=[])
    two = build_encounter(device_id="d", started_us=0, ended_us=1, faces=[], speech=[])
    assert one.id != two.id
    uuid.UUID(one.id)  # raises if it is not a real uuid


def test_an_explicit_id_is_kept():
    encounter = build_encounter(
        device_id="d", started_us=0, ended_us=1, faces=[], speech=[], encounter_id=OTHER
    )
    assert encounter.id == OTHER
