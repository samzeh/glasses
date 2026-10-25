"""Encounters: one interaction, one clock, both sensors.

The unit everything hangs off. A face and a voice only become the same
person because they were observed at the same moment, so the shared clock
and the encounter that scopes it are prerequisites for binding, not
bookkeeping.
"""

from .clock import US, Clock, FixedClock, seconds_to_us, us_to_seconds
from .record import build_encounter, face_observations, speech_observations
from .session import EncounterSession, TriggerConfig, merge_face_frames
from .types import (
    Encounter,
    EncounterStatus,
    FaceObservation,
    Interval,
    SpeechObservation,
)

__all__ = [
    "US",
    "Clock",
    "Encounter",
    "EncounterSession",
    "EncounterStatus",
    "FaceObservation",
    "FixedClock",
    "Interval",
    "SpeechObservation",
    "TriggerConfig",
    "build_encounter",
    "face_observations",
    "merge_face_frames",
    "speech_observations",
    "seconds_to_us",
    "us_to_seconds",
]
