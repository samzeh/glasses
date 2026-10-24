"""Deciding that a voice and a face belong to the same person.

The evidence is co-occurrence. If exactly one non-wearer face was visible
while someone spoke, that face is probably the speaker. Nothing off the
shelf does this, and it is the piece that turns two separate recognisers
into one system that gets more confident about a person every time you
meet them.

Three rules carry the design.

**Refuse to guess.** Two faces visible during one utterance produces *no
evidence at all*, not a best guess weighted by who was more centred.
Guessing here is how a profile ends up asserting that your colleague has
three children who actually belong to whoever they were standing next to,
and a wrong fact is worse than a missing one because nobody knows to doubt
it. Later encounters with one of those people alone usually resolve it
anyway.

**Weight by how good the evidence was**, not merely that it existed. Six
seconds of clear speech with a confidently matched face is worth far more
than a 200 ms overlap with a marginal one.

**A binding is a hypothesis, not a fact.** Agreement strengthens it,
contradiction weakens it, and it takes agreement across two separate
encounters to be believed — the same two-encounter rule the PRD applies to
auto-enrollment, and for the same reason: one encounter's observations are
highly correlated and will happily agree with themselves.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum

from ..encounter.clock import seconds_to_us, us_to_seconds
from ..encounter.types import Encounter, SpeechObservation


class Refusal(StrEnum):
    """Why a given utterance yielded no evidence. Kept rather than
    discarded: when nothing binds, the distribution of reasons is what
    tells you whether the camera, the gallery or the room is at fault."""

    NO_FACE = "no_face_visible"
    AMBIGUOUS = "several_faces_visible"
    UNIDENTIFIED_FACE = "face_not_recognised"
    UNIDENTIFIED_VOICE = "voice_not_recognised"
    WEARER = "wearer_speaking"
    TOO_BRIEF = "overlap_too_brief"


@dataclass(frozen=True)
class Evidence:
    """One utterance's worth of support for one (voice, face) pairing."""

    voice_id: str
    face_id: str
    weight: float
    encounter_id: str
    overlap_s: float

    def __str__(self) -> str:
        return (
            f"{self.voice_id} = {self.face_id}  weight {self.weight:.2f} from {self.overlap_s:.1f}s"
        )


@dataclass(frozen=True)
class BindingConfig:
    #: An overlap shorter than this is coincidence, not co-occurrence.
    min_overlap_s: float = 1.0
    #: Overlap at which an utterance contributes its full weight. Longer
    #: helps no further: the question is whether they were both present,
    #: and a few seconds settles it.
    saturating_overlap_s: float = 4.0
    #: Total weight needed before a binding is believed.
    min_support: float = 2.0
    #: And across how many separate encounters. One encounter's
    #: observations are correlated and will agree with themselves.
    min_encounters: int = 2
    #: The best pairing must beat the runner-up by this ratio. Without it,
    #: two people who are always in the room together bind arbitrarily.
    min_margin: float = 2.0


def evidence_from_encounter(
    encounter: Encounter,
    config: BindingConfig | None = None,
) -> tuple[list[Evidence], dict[Refusal, int]]:
    """Read one encounter for co-occurrence evidence.

    Returns the evidence found and a tally of why the rest was refused.
    """
    cfg = config or BindingConfig()
    found: list[Evidence] = []
    refused: dict[Refusal, int] = defaultdict(int)

    for utterance in encounter.speech:
        refusal = _refuse(utterance)
        if refusal is not None:
            refused[refusal] += 1
            continue

        visible = encounter.faces_during(
            utterance.span, min_overlap_us=seconds_to_us(cfg.min_overlap_s)
        )
        identified = [f for f in visible if f.person_id is not None]

        if not visible:
            refused[Refusal.NO_FACE] += 1
            continue
        if len(visible) > 1:
            # The whole point. Several people were in frame, so which of
            # them was speaking is unknowable from this utterance.
            refused[Refusal.AMBIGUOUS] += 1
            continue
        if not identified:
            refused[Refusal.UNIDENTIFIED_FACE] += 1
            continue

        face = identified[0]
        overlap_us = face.span.overlap_us(utterance.span)
        overlap_s = us_to_seconds(overlap_us)
        duration = min(1.0, overlap_s / cfg.saturating_overlap_s)

        found.append(
            Evidence(
                voice_id=str(utterance.person_id),
                face_id=str(face.person_id),
                weight=duration * face.confidence * utterance.confidence,
                encounter_id=encounter.id,
                overlap_s=overlap_s,
            )
        )

    return found, dict(refused)


def _refuse(utterance: SpeechObservation) -> Refusal | None:
    if utterance.is_wearer:
        # The wearer is already known, and binding their voice to a face in
        # front of them would be exactly backwards.
        return Refusal.WEARER
    if utterance.person_id is None:
        return Refusal.UNIDENTIFIED_VOICE
    return None


@dataclass
class Hypothesis:
    """An accumulated belief that one voice and one face are one person."""

    voice_id: str
    face_id: str
    support: float = 0.0
    encounters: set[str] = field(default_factory=set)
    overlap_s: float = 0.0

    def add(self, evidence: Evidence) -> None:
        self.support += evidence.weight
        self.encounters.add(evidence.encounter_id)
        self.overlap_s += evidence.overlap_s

    def __str__(self) -> str:
        return (
            f"{self.voice_id} = {self.face_id}  support {self.support:.2f} "
            f"over {len(self.encounters)} encounter(s), {self.overlap_s:.0f}s"
        )
