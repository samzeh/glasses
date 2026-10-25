"""Enrolling someone from the conversation they introduced themselves in."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from glasses import autoenroll as A
from glasses.identity.types import Embedding
from glasses.search.documents import PersonFact
from glasses.store import LocalStore
from glasses.voice.transcribe import Utterance


def embedding(seed: int = 0) -> Embedding:
    rng = np.random.default_rng(seed)
    return Embedding(
        vector=rng.normal(size=512).astype(np.float32),
        model_name="wespeaker/en_voxceleb_CAMPP",
        model_version="test",
    )


@dataclass
class FakeSegment:
    speaker: int | None


@dataclass
class FakeResult:
    """Stands in for VoiceResult: only two fields are read."""

    segment: FakeSegment
    embedding: Embedding | None


def result(speaker: int | None, *, with_embedding: bool = True, seed: int = 0) -> FakeResult:
    return FakeResult(FakeSegment(speaker), embedding(seed) if with_embedding else None)


def fact(kind: str, value: str, *, speaker: str | None, confidence: float = 0.95, person=None):
    return PersonFact(
        encounter_id="e1",
        kind=kind,
        value=value,
        person_id=person,
        person_name=speaker,
        confidence=confidence,
    )


def store(tmp_path) -> LocalStore:
    return LocalStore.load(tmp_path / "store")


# --- what counts as a name -----------------------------------------------


@pytest.mark.parametrize("value", ["Vrinda", "Test Person", "Maria de la Cruz"])
def test_a_name_is_plausible(value):
    assert A.plausible_name(value)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "...",
        "the person sitting across from me at the table",  # a clause, not a name
        "x" * 61,
    ],
)
def test_junk_is_not_a_name(value):
    """A gallery entry is forever, so the cheap checks run first."""
    assert not A.plausible_name(value)


def test_only_confident_names_are_offered():
    facts = [
        fact("name", "Vrinda", speaker="Speaker 1", confidence=0.95),
        fact("name", "Dan", speaker="Speaker 2", confidence=0.4),
    ]
    assert A.names_by_speaker(facts) == {"Speaker 1": ("Vrinda", 0.95)}


def test_the_best_supported_name_wins():
    facts = [
        fact("name", "V", speaker="Speaker 1", confidence=0.85),
        fact("name", "Vrinda", speaker="Speaker 1", confidence=0.97),
    ]
    assert A.names_by_speaker(facts)["Speaker 1"] == ("Vrinda", 0.97)


def test_a_name_for_an_already_known_person_is_ignored():
    """Someone the gallery placed is matched, not re-created."""
    facts = [fact("name", "Sam", speaker="Sam", person="existing-id")]
    assert A.names_by_speaker(facts) == {}


def test_other_fact_kinds_are_not_names():
    assert A.names_by_speaker([fact("employer", "Helix Bio", speaker="Speaker 1")]) == {}


# --- which voices are available ------------------------------------------


def test_only_unplaced_non_wearer_speakers_offer_a_voice():
    results = [result(0, seed=1), result(1, seed=2), result(2, seed=3)]
    labels = [
        ("w1", "Sam", True),  # the wearer
        (None, None, False),  # a stranger
        ("p9", "Dan", False),  # already enrolled
    ]
    assert list(A.embeddings_by_speaker(results, labels)) == ["Speaker 1"]


def test_segments_with_no_embedding_are_skipped():
    results = [result(1, with_embedding=False)]
    assert A.embeddings_by_speaker(results, [(None, None, False)]) == {}


# --- enrolling ------------------------------------------------------------


def test_a_stranger_who_introduces_themselves_is_enrolled(tmp_path):
    s = store(tmp_path)
    facts = [
        fact("name", "Vrinda", speaker="Speaker 1"),
        fact("employer", "Acme", speaker="Speaker 1"),
    ]
    enrolled, updated = A.enroll_from_conversation(
        s, facts, [result(1, seed=1), result(1, seed=2)], [(None, None, False)] * 2
    )

    assert len(enrolled) == 1
    assert enrolled[0].display_name == "Vrinda"
    assert enrolled[0].templates == 2
    assert s.find_by_name("Vrinda") is not None
    # And her facts now point at her, not at a label.
    assert {f.person_name for f in updated} == {"Vrinda"}
    assert all(f.person_id == enrolled[0].person_id for f in updated)


def test_the_voiceprints_are_stored_so_the_next_encounter_matches(tmp_path):
    """Enrolling without a voiceprint makes a person nobody can recognise."""
    s = store(tmp_path)
    A.enroll_from_conversation(
        s, [fact("name", "Vrinda", speaker="Speaker 1")], [result(1)], [(None, None, False)]
    )
    assert len(s.gallery(kind="voice")) == 1


def test_a_name_with_no_voiceprint_enrolls_nobody(tmp_path):
    s = store(tmp_path)
    enrolled, updated = A.enroll_from_conversation(
        s,
        [fact("name", "Vrinda", speaker="Speaker 1")],
        [result(1, with_embedding=False)],
        [(None, None, False)],
    )
    assert enrolled == []
    assert len(s) == 0
    assert updated[0].person_id is None


def test_a_name_collision_declines_rather_than_guesses(tmp_path):
    """Two people called Dan is a merge, and merging is M4's job."""
    s = store(tmp_path)
    s.enroll("Vrinda", [embedding(9)], consent_note="earlier", kind="voice")

    enrolled, _ = A.enroll_from_conversation(
        s, [fact("name", "Vrinda", speaker="Speaker 1")], [result(1)], [(None, None, False)]
    )
    assert enrolled == []
    assert len(s) == 1


def test_nothing_happens_when_nobody_introduced_themselves(tmp_path):
    s = store(tmp_path)
    facts = [fact("employer", "Acme", speaker="Speaker 1")]
    enrolled, updated = A.enroll_from_conversation(s, facts, [result(1)], [(None, None, False)])
    assert enrolled == []
    assert updated == facts


def test_facts_are_rebuilt_not_mutated(tmp_path):
    """doc_id derives from person_id, so a fact that changes owner has to
    become a different document or the index keeps the orphan too."""
    s = store(tmp_path)
    original = fact("name", "Vrinda", speaker="Speaker 1")
    before = original.doc_id

    _, updated = A.enroll_from_conversation(
        s, [original], [result(1)], [(None, None, False)]
    )
    assert original.person_id is None
    assert original.doc_id == before
    assert updated[0].doc_id != before


# --- the transcript agrees -----------------------------------------------


def test_the_transcript_gets_the_new_name(tmp_path):
    """Otherwise the facts say Vrinda while the segments they cite still
    say Speaker 1, and one encounter disagrees with itself."""
    enrolled = [
        A.Enrolled(
            person_id="p1",
            display_name="Vrinda",
            speaker_label="Speaker 1",
            templates=2,
            confidence=0.95,
        )
    ]
    utterances = [
        Utterance(text="Hi, I'm Vrinda.", start_s=0.0, end_s=2.0, speaker=1),
        Utterance(text="Good to meet you.", start_s=2.0, end_s=3.0, speaker=0, is_wearer=True),
    ]
    out = A.relabel_utterances(utterances, enrolled)
    assert out[0].person_name == "Vrinda"
    assert out[0].person_id == "p1"
    assert out[1].person_name is None  # the wearer is untouched


def test_relabelling_leaves_already_identified_speakers_alone():
    enrolled = [
        A.Enrolled(
            person_id="p1",
            display_name="Vrinda",
            speaker_label="Speaker 1",
            templates=1,
            confidence=0.9,
        )
    ]
    known = [
        Utterance(
            text="hi", start_s=0.0, end_s=1.0, speaker=1, person_id="other", person_name="Dan"
        )
    ]
    assert A.relabel_utterances(known, enrolled)[0].person_name == "Dan"


# --- face is the identity, the name is only what someone said ------------


def test_a_new_face_is_enrolled_even_when_the_name_is_taken(tmp_path):
    """The camera path's rule. Everyone reaching enrollment is a face the
    gallery already failed to match, so they are a new person whatever
    they are called — declining on the name would mean a new face never
    gets enrolled at all."""
    s = store(tmp_path)
    s.enroll("Vrinda", [embedding(9)], consent_note="an earlier encounter", kind="voice")

    enrolled, updated = A.enroll_from_conversation(
        s,
        [fact("name", "Vrinda", speaker="Speaker 1")],
        [],
        [],
        embeddings={"Speaker 1": [embedding(2)]},
        kind="face",
        allow_duplicate_names=True,
    )

    assert len(enrolled) == 1
    assert enrolled[0].kind == "face"
    assert len(s) == 2  # the voice-only one, plus this new face
    assert updated[0].person_id == enrolled[0].person_id


def test_the_voice_path_still_declines_on_a_taken_name(tmp_path):
    """There the name is the only evidence, so a collision is a merge."""
    s = store(tmp_path)
    s.enroll("Vrinda", [embedding(9)], consent_note="earlier", kind="voice")

    enrolled, _ = A.enroll_from_conversation(
        s, [fact("name", "Vrinda", speaker="Speaker 1")], [result(1)], [(None, None, False)]
    )

    assert enrolled == []
    assert len(s) == 1


def test_a_recognised_face_never_reaches_enrollment(tmp_path):
    """Their stored name wins over anything said in the room: the facts
    already carry a person_id, so names_by_speaker skips them."""
    s = store(tmp_path)
    facts = [fact("name", "Vrinda", speaker="V. Sharma", person="known-id")]

    enrolled, updated = A.enroll_from_conversation(
        s, facts, [], [], embeddings={"V. Sharma": [embedding(2)]}, kind="face"
    )

    assert enrolled == []
    assert updated[0].person_name == "V. Sharma"
    assert updated[0].person_id == "known-id"
