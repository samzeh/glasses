"""Enrolling someone from the conversation you just had with them.

Without this, meeting a stranger produces facts attached to "Speaker 1".
They are correct facts — including, usually, a ``name`` fact saying what
the stranger is called — but they hang off a label that means nothing
after the recording ends. Meet the same person tomorrow and they are
"Speaker 1" again, in a new encounter, with no history.

The two halves needed to fix that are already in hand at the same moment:

* Gemini returns ``kind="name"`` for a speaker label, read out of the
  conversation itself ("Hi, I'm Vrinda").
* The voice pipeline holds that speaker's embedding, because it just
  tried and failed to match it against the gallery.

Joining them is this module. The glasses hear someone introduce
themselves and remember the voice under that name, so the next encounter
recognises them.

Three things keep that from enrolling nonsense, and they are the whole
design:

* **Only a speaker the gallery could not place.** Someone already
  enrolled is matched, not re-created, and the wearer is never a
  conversational partner.
* **Only a confidently heard name.** The floor here is higher than the
  one for ordinary facts: a wrong employer is a wrong row, a wrong name
  is a person who does not exist and who will collect other people's
  facts for as long as the gallery lives.
* **Only a name that is not already taken.** Two people called Dan in
  one gallery is a merge problem, and merging is M4's job, not this
  module's. A collision declines rather than guesses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .search.documents import PersonFact
from .voice.transcribe import Utterance

#: A name has to clear this to become a person. Higher than
#: ``facts.MIN_CONFIDENCE`` on purpose — see the module docstring.
MIN_NAME_CONFIDENCE = 0.80

#: Longer than this is a sentence the model mislabelled as a name.
MAX_NAME_WORDS = 4


@dataclass
class Enrolled:
    """One person created from a conversation."""

    person_id: str
    display_name: str
    speaker_label: str
    templates: int
    confidence: float
    #: Which gallery the templates landed in: "face" or "voice".
    kind: str = "voice"

    def __str__(self) -> str:
        word = "faceprint" if self.kind == "face" else "voiceprint"
        return (
            f"{self.display_name} (was {self.speaker_label}) — "
            f"{self.templates} {word}(s), name heard at {self.confidence:.2f}"
        )


def plausible_name(value: str) -> bool:
    """Whether a ``name`` fact's value looks like something to call someone.

    The model is asked for a name and mostly returns one, but a garbled
    line can produce a whole clause. A gallery entry is forever, so the
    cheap checks run before one is created.
    """
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 60:
        return False
    if len(cleaned.split()) > MAX_NAME_WORDS:
        return False
    # Needs at least one letter; "..." and "?" are not names.
    return any(character.isalpha() for character in cleaned)


def names_by_speaker(
    facts: Sequence[PersonFact],
    *,
    min_confidence: float = MIN_NAME_CONFIDENCE,
) -> dict[str, tuple[str, float]]:
    """The best-supported name offered for each unattributed speaker.

    Only facts that are still unattached matter: a fact that already has a
    ``person_id`` belongs to someone the gallery placed, and their name is
    not up for revision here.
    """
    best: dict[str, tuple[str, float]] = {}
    for fact in facts:
        if fact.kind != "name" or fact.person_id is not None:
            continue
        label = (fact.person_name or "").strip()
        if not label or fact.confidence < min_confidence:
            continue
        if not plausible_name(fact.value):
            continue
        current = best.get(label)
        if current is None or fact.confidence > current[1]:
            best[label] = (fact.value.strip(), float(fact.confidence))
    return best


def embeddings_by_speaker(
    results: Sequence[Any],
    labels: Sequence[tuple[str | None, str | None, bool]],
) -> dict[str, list[Any]]:
    """Voice embeddings for each speaker the gallery could not place.

    ``results`` and ``labels`` are the parallel lists the transcription
    path already builds: one ``VoiceResult`` and one
    ``(person_id, name, is_wearer)`` per usable segment.
    """
    table: dict[str, list[Any]] = {}
    for result, (person_id, _name, is_wearer) in zip(results, labels, strict=True):
        if person_id is not None or is_wearer:
            continue
        if result.embedding is None or result.segment.speaker is None:
            continue
        table.setdefault(f"Speaker {result.segment.speaker}", []).append(result.embedding)
    return table


def enroll_from_conversation(
    store: Any,
    facts: Sequence[PersonFact],
    results: Sequence[Any],
    labels: Sequence[tuple[str | None, str | None, bool]],
    *,
    utterances: Sequence[Utterance] = (),
    min_confidence: float = MIN_NAME_CONFIDENCE,
    source: str = "",
    embeddings: dict[str, list[Any]] | None = None,
    kind: str = "voice",
) -> tuple[list[Enrolled], list[PersonFact]]:
    """Create people from introductions, and re-attribute their facts.

    ``embeddings`` overrides where the templates come from, keyed by the
    same speaker label the facts carry. The camera path passes face
    embeddings here, which matters: voice alone cannot reliably separate
    two people, so enrolling a stranger on their voiceprint creates
    someone the next encounter will not recognise. ``kind`` says which
    gallery they land in.

    Returns what was enrolled and the facts with ``person_id`` filled in
    for them. The facts are rebuilt rather than mutated because
    ``PersonFact.doc_id`` is derived from ``person_id`` — a fact that
    changes owner has to become a different document, or the index keeps
    the orphaned original alongside the corrected one.
    """
    offered = names_by_speaker(facts, min_confidence=min_confidence)
    if not offered:
        return [], list(facts)

    voices = embeddings if embeddings is not None else embeddings_by_speaker(results, labels)
    enrolled: list[Enrolled] = []
    assigned: dict[str, tuple[str, str]] = {}

    for label, (name, confidence) in offered.items():
        templates = voices.get(label)
        if not templates:
            # Heard a name but kept no usable template for them. Enrolling
            # now would make a person the next encounter cannot recognise.
            continue
        if store.find_by_name(name) is not None:
            # A second Dan is a merge, and merging is not this module's job.
            continue

        person_id = store.enroll(
            name,
            list(templates),
            consent_note=f"introduced themselves during {source or 'a recorded conversation'}",
            source=source,
            kind=kind,
        )
        assigned[label] = (person_id, name)
        enrolled.append(
            Enrolled(
                person_id=person_id,
                display_name=name,
                speaker_label=label,
                templates=len(templates),
                confidence=confidence,
                kind=kind,
            )
        )

    if not assigned:
        return [], list(facts)

    store.save()
    return enrolled, _reattribute(facts, assigned)


def _reattribute(
    facts: Sequence[PersonFact], assigned: dict[str, tuple[str, str]]
) -> list[PersonFact]:
    """Point each newly-enrolled speaker's facts at their new person."""
    out: list[PersonFact] = []
    for fact in facts:
        match = assigned.get((fact.person_name or "").strip()) if fact.person_id is None else None
        if match is None:
            out.append(fact)
            continue
        person_id, name = match
        out.append(
            PersonFact(
                encounter_id=fact.encounter_id,
                kind=fact.kind,
                value=fact.value,
                person_id=person_id,
                person_name=name,
                evidence_span=fact.evidence_span,
                evidence_start_ms=fact.evidence_start_ms,
                evidence_end_ms=fact.evidence_end_ms,
                confidence=fact.confidence,
                occurred_at=fact.occurred_at,
                source=fact.source,
            )
        )
    return out


def relabel_utterances(
    utterances: Sequence[Utterance], enrolled: Sequence[Enrolled]
) -> list[Utterance]:
    """Give the transcript the new names, so the indexed segments agree.

    Without this the facts say "Vrinda" while the segments they cite still
    say "Speaker 1", and the two halves of one encounter disagree about
    who was in it.
    """
    if not enrolled:
        return list(utterances)

    by_label = {item.speaker_label: item for item in enrolled}
    out: list[Utterance] = []
    for utterance in utterances:
        label = f"Speaker {utterance.speaker}" if utterance.speaker is not None else None
        item = by_label.get(label or "")
        if utterance.person_id is not None or item is None:
            out.append(utterance)
            continue
        out.append(
            Utterance(
                text=utterance.text,
                start_s=utterance.start_s,
                end_s=utterance.end_s,
                speaker=utterance.speaker,
                person_id=item.person_id,
                person_name=item.display_name,
                is_wearer=utterance.is_wearer,
            )
        )
    return out


__all__ = [
    "MAX_NAME_WORDS",
    "MIN_NAME_CONFIDENCE",
    "Enrolled",
    "embeddings_by_speaker",
    "enroll_from_conversation",
    "names_by_speaker",
    "plausible_name",
    "relabel_utterances",
]
