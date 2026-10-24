"""The encounter lifecycle, and the trigger that drives it.

Dwell alone over-fires badly — a person on a train sits in front of you for
forty minutes. Speech alone fires on a podcast. Two signals in combination
is what makes the trigger usable:

* **Dwell** — a face in the centre region for >= 5 seconds across >= 3
  separate detections. Counting detections rather than requiring a
  continuous track survives a low capture rate.
* **Partner speech** — voice activity from someone who is not the wearer.
  This is what separates a conversation from a person who merely happens
  to be standing there.

Either alone opens a *provisional* encounter. Both together promote it to
*confirmed*, which is the only kind that gets profiled. Provisional
encounters that never hear partner speech expire, and their media is
deleted — that discard path matters for storage and for privacy.

The session is fed observations; it does no detection itself. That keeps
the state machine pure and testable, and it means the same logic serves a
live camera and a recording being replayed.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from .clock import seconds_to_us
from .types import (
    Encounter,
    EncounterStatus,
    FaceObservation,
    Interval,
    SpeechObservation,
)


@dataclass(frozen=True)
class TriggerConfig:
    """CALIBRATE against real wear. The false-trigger target is under one
    per hour of ordinary walking around, and only measurement will tell you
    whether these numbers meet it."""

    #: Dwell: how long a centred face must be present.
    min_dwell_s: float = 5.0
    #: And across how many separate detections. Requiring several rather
    #: than a continuous track survives the capture rate.
    min_detections: int = 3
    #: Close when partner speech has been absent this long...
    silence_timeout_s: float = 60.0
    #: ...*and* no face has been seen this long. Separate timeouts, because
    #: people pause and because people look away, and those are different.
    face_timeout_s: float = 30.0
    #: A provisional encounter with no partner speech by now never becomes
    #: a conversation.
    expiry_s: float = 30.0


class EncounterSession:
    """Opens, promotes, and closes encounters from a stream of observations."""

    def __init__(
        self,
        device_id: str = "laptop",
        *,
        config: TriggerConfig | None = None,
    ) -> None:
        self.device_id = device_id
        self.config = config or TriggerConfig()
        self.current: Encounter | None = None
        self.finished: list[Encounter] = []

        self._dwell_us = 0
        self._detections = 0
        self._last_face_us: int | None = None
        self._last_partner_speech_us: int | None = None

    # --- input -------------------------------------------------------------

    def observe_face(self, observation: FaceObservation) -> Encounter:
        """Record a face seen over a span, opening an encounter if needed."""
        encounter = self._ensure_open(observation.span.start_us)
        encounter.faces.append(observation)

        self._last_face_us = max(self._last_face_us or 0, observation.span.end_us)
        if observation.centred:
            self._dwell_us += observation.span.duration_us
            self._detections += observation.frame_count

        self._maybe_confirm(observation.span.end_us)
        return encounter

    def observe_speech(self, observation: SpeechObservation) -> Encounter:
        """Record speech over a span, opening an encounter if needed."""
        encounter = self._ensure_open(observation.span.start_us)
        encounter.speech.append(observation)

        if observation.is_partner:
            self._last_partner_speech_us = max(
                self._last_partner_speech_us or 0, observation.span.end_us
            )
        self._maybe_confirm(observation.span.end_us)
        return encounter

    def tick(self, now_us: int) -> Encounter | None:
        """Advance time without new observations.

        Returns the encounter if this tick ended one. Must be called even
        when nothing is happening, because closing and expiring are both
        defined by the *absence* of signal.
        """
        if self.current is None:
            return None

        if self._should_expire(now_us):
            return self._finish(EncounterStatus.EXPIRED, now_us)
        if self._should_close(now_us):
            return self._finish(EncounterStatus.CLOSED, now_us)
        return None

    def close(self, now_us: int) -> Encounter | None:
        """End the current encounter now, whatever its state.

        A confirmed encounter closes; one that never heard partner speech
        expires, because it was never a conversation.
        """
        if self.current is None:
            return None
        status = (
            EncounterStatus.CLOSED
            if self.current.status is EncounterStatus.CONFIRMED
            else EncounterStatus.EXPIRED
        )
        return self._finish(status, now_us)

    # --- state ---------------------------------------------------------------

    @property
    def dwell_satisfied(self) -> bool:
        return (
            self._dwell_us >= seconds_to_us(self.config.min_dwell_s)
            and self._detections >= self.config.min_detections
        )

    @property
    def heard_partner(self) -> bool:
        return self._last_partner_speech_us is not None

    def _ensure_open(self, at_us: int) -> Encounter:
        if self.current is None:
            self.current = Encounter(
                id=uuid.uuid4().hex,
                device_id=self.device_id,
                started_us=at_us,
            )
            self._dwell_us = 0
            self._detections = 0
            self._last_face_us = None
            self._last_partner_speech_us = None
        return self.current

    def _maybe_confirm(self, at_us: int) -> None:
        """Both signals together promote the encounter. Only then."""
        if (
            self.current is not None
            and self.current.status is EncounterStatus.PROVISIONAL
            and self.dwell_satisfied
            and self.heard_partner
        ):
            self.current.status = EncounterStatus.CONFIRMED

    def _should_expire(self, now_us: int) -> bool:
        if self.current is None or self.current.status is not EncounterStatus.PROVISIONAL:
            return False
        age = now_us - self.current.started_us
        return age >= seconds_to_us(self.config.expiry_s) and not self.heard_partner

    def _should_close(self, now_us: int) -> bool:
        """Both timeouts must be satisfied, not either.

        People pause mid-conversation and people look away mid-conversation;
        closing on one signal alone chops an encounter in half and the two
        halves then look like two different meetings.
        """
        if self.current is None or self.current.status is not EncounterStatus.CONFIRMED:
            return False

        silent_for = now_us - (self._last_partner_speech_us or self.current.started_us)
        faceless_for = now_us - (self._last_face_us or self.current.started_us)
        return silent_for >= seconds_to_us(
            self.config.silence_timeout_s
        ) and faceless_for >= seconds_to_us(self.config.face_timeout_s)

    def _finish(self, status: EncounterStatus, now_us: int) -> Encounter:
        assert self.current is not None
        encounter = self.current
        encounter.status = status
        encounter.ended_us = max(now_us, encounter.started_us)
        self.finished.append(encounter)
        self.current = None
        return encounter


def merge_face_frames(
    detections: list[tuple[int, str | None, float, bool]],
    *,
    max_gap_s: float = 2.0,
) -> list[FaceObservation]:
    """Turn per-frame detections into visibility spans.

    ``detections`` is ``(captured_at_us, person_id, confidence, centred)``,
    one per frame per face. Consecutive frames of the same identity within
    ``max_gap_s`` become one observation — a face present at t=10, 11 and 12
    is one span, not three.

    Unidentified faces (``person_id`` None) are never merged with each
    other: two strangers in frame would otherwise collapse into one span
    and the ambiguity that protects the binding would be lost.
    """
    if not detections:
        return []

    gap_us = seconds_to_us(max_gap_s)
    ordered = sorted(detections, key=lambda d: d[0])

    runs: list[list[tuple[int, str | None, float, bool]]] = [[ordered[0]]]
    for detection in ordered[1:]:
        previous = runs[-1][-1]
        same_person = detection[1] is not None and detection[1] == previous[1]
        close_enough = detection[0] - previous[0] <= gap_us
        if same_person and close_enough:
            runs[-1].append(detection)
        else:
            runs.append([detection])

    observations: list[FaceObservation] = []
    for run in runs:
        times = [d[0] for d in run]
        # A single frame is an instant, not a zero-length span: give it the
        # sampling interval so dwell can accumulate from it at all.
        end = max(times) if len(run) > 1 else max(times) + seconds_to_us(1.0)
        observations.append(
            FaceObservation(
                span=Interval(min(times), end),
                person_id=run[0][1],
                confidence=sum(d[2] for d in run) / len(run),
                frame_count=len(run),
                is_primary=True,
                centred=all(d[3] for d in run),
            )
        )
    return observations
