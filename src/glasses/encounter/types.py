"""Observations on a shared timeline.

Both sensors produce the same shape: *something was true between these two
microseconds*. A face is visible over a span; a voice speaks over a span.
Once both are expressed that way, binding is interval arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .clock import us_to_seconds


@dataclass(frozen=True, order=True)
class Interval:
    """A closed-open span of device-clock microseconds."""

    start_us: int
    end_us: int

    def __post_init__(self) -> None:
        if self.end_us < self.start_us:
            raise ValueError(f"interval ends before it starts: {self.start_us}..{self.end_us}")

    @property
    def duration_us(self) -> int:
        return self.end_us - self.start_us

    @property
    def duration_s(self) -> float:
        return us_to_seconds(self.duration_us)

    def overlaps(self, other: Interval) -> bool:
        return self.start_us < other.end_us and other.start_us < self.end_us

    def overlap_us(self, other: Interval) -> int:
        """How much time the two share. Zero when they do not touch."""
        return max(0, min(self.end_us, other.end_us) - max(self.start_us, other.start_us))

    def contains(self, device_us: int) -> bool:
        return self.start_us <= device_us < self.end_us

    def union(self, other: Interval) -> Interval:
        return Interval(min(self.start_us, other.start_us), max(self.end_us, other.end_us))


@dataclass(frozen=True)
class FaceObservation:
    """A face seen over a span, built from consecutive frames.

    Frames are discrete samples — one a second during a burst — so a face
    present at t=10, 11 and 12 is one observation spanning 10 to 12, not
    three. ``frame_count`` is kept because the trigger counts detections
    rather than trusting a continuous track: the capture rate is too low
    for tracking to be reliable.
    """

    span: Interval
    person_id: str | None
    confidence: float
    frame_count: int
    is_primary: bool
    #: True when the face sat in the centre region, which is what dwell means.
    centred: bool = True

    @property
    def identified(self) -> bool:
        return self.person_id is not None


@dataclass(frozen=True)
class SpeechObservation:
    """A stretch of speech over a span.

    ``person_id`` is set when the voice gallery recognised them. ``speaker``
    is the anonymous within-recording label, which survives even when the
    identity does not.
    """

    span: Interval
    person_id: str | None
    speaker: int | None
    confidence: float
    is_wearer: bool
    text: str | None = None

    @property
    def is_partner(self) -> bool:
        """Speech from someone other than the wearer.

        The signal the trigger actually cares about: a person standing in
        front of you is not a conversation, and this is what separates the
        two.
        """
        return not self.is_wearer


class EncounterStatus(StrEnum):
    #: One qualifying signal seen. Media is being captured but nothing is
    #: profiled, and it is discarded if the encounter expires.
    PROVISIONAL = "provisional"
    #: Dwell and partner speech both seen. The only kind that gets profiled.
    CONFIRMED = "confirmed"
    #: Ended normally.
    CLOSED = "closed"
    #: Never saw partner speech. Media is deleted; that discard path matters
    #: for storage and for privacy.
    EXPIRED = "expired"


@dataclass
class Encounter:
    """One continuous interaction, on one clock.

    The unit everything else hangs off: the frames captured, the audio
    recorded, and the identities eventually resolved all belong to an
    encounter, and that is what makes a fact traceable back to a moment.
    """

    id: str
    device_id: str
    started_us: int
    status: EncounterStatus = EncounterStatus.PROVISIONAL
    ended_us: int | None = None
    location: str | None = None
    faces: list[FaceObservation] = field(default_factory=list)
    speech: list[SpeechObservation] = field(default_factory=list)

    @property
    def span(self) -> Interval:
        return Interval(self.started_us, self.ended_us if self.ended_us else self.started_us)

    @property
    def is_open(self) -> bool:
        return self.status in (EncounterStatus.PROVISIONAL, EncounterStatus.CONFIRMED)

    @property
    def partner_speech(self) -> list[SpeechObservation]:
        return [s for s in self.speech if s.is_partner]

    @property
    def partner_speech_us(self) -> int:
        return sum(s.span.duration_us for s in self.partner_speech)

    def faces_during(self, span: Interval, *, min_overlap_us: int = 0) -> list[FaceObservation]:
        """Which faces were visible while something else was happening.

        The primitive the whole binding rests on.
        """
        return [f for f in self.faces if f.span.overlap_us(span) > min_overlap_us]
