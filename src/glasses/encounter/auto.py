"""Drive an ``EncounterSession`` from live signals, across two machines.

The trigger policy lives in ``session.py`` and is not repeated here. What
this adds is the plumbing that policy needs to run unattended:

* **Faces** arrive as camera frames on this host. A frame is an instant, and
  the session wants spans, so each frame's faces are given the gap since the
  previous frame — clamped, because a stalled camera must not be able to
  credit two minutes of dwell with one frame.
* **Speech** arrives as *events* from the phone browser that owns the
  microphone. The browser sends "speech started" and "speech ended" and
  nothing else: its clock is not this clock, and a timestamp crossing that
  gap would silently misalign every binding. Events are stamped on arrival.

Long speech is committed in segments rather than held until it ends. A
sixty-second answer would otherwise leave ``_last_partner_speech_us`` unset
and the encounter would expire mid-sentence.

Browser voice activity cannot say *who* spoke. These observations are
recorded unattributed — ``person_id`` None and ``confidence`` zero — and as
partner speech, which is what makes them count towards the trigger: the
wearer's own voice starting a recording is the stated intent, and attribution
is a later pass over the audio, not something a level meter can do.
"""

from __future__ import annotations

from dataclasses import dataclass

from .clock import Clock, seconds_to_us
from .session import EncounterSession, TriggerConfig
from .types import Encounter, EncounterStatus, FaceObservation, Interval, SpeechObservation

#: Commit an in-progress utterance at least this often.
SPEECH_SEGMENT_S = 2.0
#: Longest gap between frames that may count as dwell.
MAX_FRAME_GAP_S = 1.0
#: How long a finished encounter stays visible to a polling client.
FINISHED_LINGER_S = 15.0


@dataclass(frozen=True)
class FaceSample:
    """One face in one frame, as the camera loop saw it."""

    person_id: str | None = None
    confidence: float = 0.0
    centred: bool = True
    is_primary: bool = True


@dataclass(frozen=True)
class TriggerState:
    """What the hub and the phone both need to know, as JSON would carry it."""

    active: bool
    confirmed: bool
    speaking: bool
    encounter_id: str | None
    faces_in_view: int
    dwell_satisfied: bool
    heard_speech: bool
    finished_id: str | None = None
    finished_status: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "active": self.active,
            "confirmed": self.confirmed,
            "speaking": self.speaking,
            "encounter_id": self.encounter_id,
            "faces_in_view": self.faces_in_view,
            "dwell_satisfied": self.dwell_satisfied,
            "heard_speech": self.heard_speech,
            "finished_id": self.finished_id,
            "finished_status": self.finished_status,
        }


class AutoTrigger:
    """Opens and closes encounters without anybody pressing anything."""

    def __init__(
        self,
        clock: Clock,
        *,
        device_id: str = "glasses",
        config: TriggerConfig | None = None,
        speech_segment_s: float = SPEECH_SEGMENT_S,
        max_frame_gap_s: float = MAX_FRAME_GAP_S,
        finished_linger_s: float = FINISHED_LINGER_S,
    ) -> None:
        self.clock = clock
        self.session = EncounterSession(device_id, config=config)
        self._segment_us = seconds_to_us(speech_segment_s)
        self._max_gap_us = seconds_to_us(max_frame_gap_s)
        self._linger_us = seconds_to_us(finished_linger_s)

        self._speaking_since: int | None = None
        self._last_frame_us: int | None = None
        self._faces_in_view = 0
        self._pending: Encounter | None = None
        self._finished_id: str | None = None
        self._finished_status: str | None = None
        self._finished_at_us: int | None = None

    # --- input -------------------------------------------------------------

    def observe_frame(self, faces: list[FaceSample], at_us: int | None = None) -> TriggerState:
        """One camera frame. Call it for every frame, faces or not.

        An empty frame still advances the span bookkeeping, so the next face
        is credited from this moment rather than from the last time one was
        visible.
        """
        now = self.clock.now_us() if at_us is None else at_us
        span = self._frame_span(now)
        self._last_frame_us = now
        self._faces_in_view = len(faces)

        for sample in faces:
            self.session.observe_face(
                FaceObservation(
                    span=span,
                    person_id=sample.person_id,
                    confidence=float(sample.confidence),
                    frame_count=1,
                    is_primary=sample.is_primary,
                    centred=sample.centred,
                )
            )
        return self.tick(now)

    def speech_started(self, at_us: int | None = None) -> TriggerState:
        """The browser heard voice activity begin."""
        now = self.clock.now_us() if at_us is None else at_us
        if self._speaking_since is None:
            self._speaking_since = now
        return self.tick(now)

    def speech_ended(self, at_us: int | None = None) -> TriggerState:
        """The browser heard it stop. Commits whatever is left of the span."""
        now = self.clock.now_us() if at_us is None else at_us
        if self._speaking_since is not None:
            self._commit_speech(self._speaking_since, now)
            self._speaking_since = None
        return self.tick(now)

    def tick(self, at_us: int | None = None) -> TriggerState:
        """Advance time. Must be called regularly: closing and expiring are
        both defined by the absence of signal."""
        now = self.clock.now_us() if at_us is None else at_us

        if self._speaking_since is not None and now - self._speaking_since >= self._segment_us:
            self._commit_speech(self._speaking_since, now)
            self._speaking_since = now

        finished = self.session.tick(now)
        if finished is not None:
            self._remember(finished, now)
        return self._state(now)

    def close(self, at_us: int | None = None) -> Encounter | None:
        """End the open encounter now — shutting down, or switching modes."""
        now = self.clock.now_us() if at_us is None else at_us
        if self._speaking_since is not None:
            self._commit_speech(self._speaking_since, now)
            self._speaking_since = None
        finished = self.session.close(now)
        if finished is not None:
            self._remember(finished, now)
        return finished

    # --- output ------------------------------------------------------------

    @property
    def state(self) -> TriggerState:
        return self._state(self.clock.now_us())

    def take_finished(self) -> Encounter | None:
        """Hand over a finished encounter, once.

        The hub writes it to the store, and writing it twice would record one
        conversation as two.
        """
        encounter, self._pending = self._pending, None
        return encounter

    # --- internals ---------------------------------------------------------

    def _frame_span(self, now: int) -> Interval:
        previous = self._last_frame_us
        if previous is None or now <= previous:
            return Interval(now, now)
        return Interval(max(previous, now - self._max_gap_us), now)

    def _commit_speech(self, start_us: int, end_us: int) -> None:
        self.session.observe_speech(
            SpeechObservation(
                span=Interval(start_us, max(start_us, end_us)),
                person_id=None,
                speaker=None,
                confidence=0.0,
                is_wearer=False,
                text=None,
            )
        )

    def _remember(self, encounter: Encounter, now: int) -> None:
        self._pending = encounter
        self._finished_id = encounter.id
        self._finished_status = str(encounter.status)
        self._finished_at_us = now
        self._last_frame_us = None

    def _state(self, now: int) -> TriggerState:
        current = self.session.current
        expired = (
            self._finished_at_us is None or now - self._finished_at_us > self._linger_us
        )
        return TriggerState(
            active=current is not None,
            # Read the session's own verdict rather than recomputing it: the
            # promotion rule belongs to the state machine, and two copies of
            # it would eventually disagree.
            confirmed=current is not None and current.status is EncounterStatus.CONFIRMED,
            speaking=self._speaking_since is not None,
            encounter_id=current.id if current else None,
            faces_in_view=self._faces_in_view,
            # Both reset on the next open; reporting them while idle would
            # show the previous conversation's progress.
            dwell_satisfied=current is not None and self.session.dwell_satisfied,
            heard_speech=current is not None and self.session.heard_partner,
            finished_id=None if expired else self._finished_id,
            finished_status=None if expired else self._finished_status,
        )


__all__ = ["AutoTrigger", "FaceSample", "TriggerState"]
