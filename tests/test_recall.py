"""The camera path's join: face identity onto transcribed captures."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from glasses import recall as R
from glasses.identity.types import Embedding


def embedding(seed: int = 0) -> Embedding:
    rng = np.random.default_rng(seed)
    return Embedding(
        vector=rng.normal(size=512).astype(np.float32),
        model_name="arcface/buffalo_l",
        model_version="test",
    )


@dataclass
class FakeTrack:
    id: int
    best_frame: object = None
    bbox: object = None


@dataclass
class FakeCapture:
    start_us: int
    end_us: int
    audio: np.ndarray = field(default_factory=lambda: np.zeros(16000, dtype=np.float32))
    track: FakeTrack | None = None
    is_wearer: bool = False


class FakeTranscriber:
    """Returns a line per capture, by start time."""

    model_name = "fake/transcriber"

    def __init__(self, texts: dict[int, str]) -> None:
        self.texts = texts
        self.calls = 0

    def transcribe(self, samples):  # type: ignore[no-untyped-def]
        self.calls += 1
        return self.texts.get(self.calls, "")


def capture(start_s, end_s, track=None, wearer=False):
    return FakeCapture(
        start_us=int(start_s * 1_000_000),
        end_us=int(end_s * 1_000_000),
        track=track,
        is_wearer=wearer,
    )


# --- labelling ------------------------------------------------------------


def test_a_recognised_face_names_the_speaker():
    identity = R.FaceIdentity(person_id="p1", name="Vrinda", similarity=0.72)
    assert R.speaker_label(capture(0, 1, FakeTrack(3)), identity) == "Vrinda"


def test_the_wearer_is_always_you():
    assert R.speaker_label(capture(0, 1, FakeTrack(3), wearer=True), None) == "you"


def test_an_unrecognised_face_keeps_its_own_track_label():
    """Two unidentified people are two people. Collapsing them into one
    'unknown' is how one person's facts land on the other."""
    one = R.speaker_label(capture(0, 1, FakeTrack(1)), R.FaceIdentity(note="not recognised"))
    two = R.speaker_label(capture(0, 1, FakeTrack(2)), R.FaceIdentity(note="not recognised"))
    assert one == "Speaker 1"
    assert two == "Speaker 2"
    assert one != two


def test_a_capture_with_no_track_is_labelled_unassigned():
    assert R.speaker_label(capture(0, 1, None), None) == "Unassigned speaker"


# --- utterances -----------------------------------------------------------


def test_identity_rides_along_with_the_words():
    track = FakeTrack(7)
    identities = {7: R.FaceIdentity(person_id="p1", name="Vrinda", similarity=0.7)}
    transcriber = FakeTranscriber({1: "I work at Acme."})

    [utterance] = R.utterances_from_captures(
        [capture(2.0, 5.0, track)], identities, transcriber, audio_start_us=0
    )
    assert utterance.text == "I work at Acme."
    assert utterance.person_id == "p1"
    assert utterance.person_name == "Vrinda"
    assert (utterance.start_s, utterance.end_s) == (2.0, 5.0)
    assert utterance.speaker == 7


def test_timestamps_are_relative_to_the_recording_start():
    track = FakeTrack(1)
    transcriber = FakeTranscriber({1: "hello"})
    [utterance] = R.utterances_from_captures(
        [capture(12.0, 14.0, track)], {}, transcriber, audio_start_us=10_000_000
    )
    assert (utterance.start_s, utterance.end_s) == (2.0, 4.0)


def test_an_empty_transcript_is_dropped():
    """A segment with no words would otherwise reach the extractor as a
    speaker who said nothing."""
    transcriber = FakeTranscriber({1: "   ", 2: "real words"})
    out = R.utterances_from_captures(
        [capture(0, 1, FakeTrack(1)), capture(1, 2, FakeTrack(1))],
        {},
        transcriber,
        audio_start_us=0,
    )
    assert [u.text for u in out] == ["real words"]


def test_the_wearer_is_marked_as_such():
    transcriber = FakeTranscriber({1: "hi there"})
    [utterance] = R.utterances_from_captures(
        [capture(0, 2, FakeTrack(0), wearer=True)], {}, transcriber, audio_start_us=0
    )
    assert utterance.is_wearer


# --- templates for auto-enrollment ---------------------------------------


def test_an_unknown_face_offers_a_template():
    track = FakeTrack(4)
    identities = {4: R.FaceIdentity(embedding=embedding(1), note="not recognised")}
    table = R.face_templates_by_label([capture(0, 2, track)], identities)
    assert list(table) == ["Speaker 4"]
    assert len(table["Speaker 4"]) == 1


def test_a_known_face_offers_nothing():
    """Someone already enrolled is matched, not re-created."""
    track = FakeTrack(4)
    identities = {4: R.FaceIdentity(person_id="p1", name="Dan", embedding=embedding(1))}
    assert R.face_templates_by_label([capture(0, 2, track)], identities) == {}


def test_the_wearer_offers_nothing():
    track = FakeTrack(0)
    identities = {0: R.FaceIdentity(embedding=embedding(1), note="not recognised")}
    assert R.face_templates_by_label([capture(0, 2, track, wearer=True)], identities) == {}


def test_several_captures_on_one_track_contribute_one_template():
    """The embedding is the track's best frame, so every capture on that
    track would contribute the same vector, and duplicates score as one."""
    track = FakeTrack(4)
    identities = {4: R.FaceIdentity(embedding=embedding(1), note="not recognised")}
    table = R.face_templates_by_label(
        [capture(0, 2, track), capture(3, 5, track), capture(6, 8, track)], identities
    )
    assert len(table["Speaker 4"]) == 1


def test_a_face_with_no_embedding_offers_nothing():
    """A frame the quality gate rejected has no embedding, and embedding it
    anyway is how a bad template enters the gallery."""
    identities = {4: R.FaceIdentity(note="rejected (blur)")}
    assert R.face_templates_by_label([capture(0, 2, FakeTrack(4))], identities) == {}


# --- identify_track guards ------------------------------------------------


def test_a_track_with_no_frame_is_not_identified():
    identity = R.identify_track(object(), object(), FakeTrack(1, best_frame=None))
    assert not identity.known
    assert identity.note == "no frame"


def test_a_missing_track_is_not_identified():
    assert R.identify_track(object(), object(), None).note == "no frame"
