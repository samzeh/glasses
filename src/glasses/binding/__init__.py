"""Binding a voice to a face: deciding they are the same person.

The capture loop turns on one question — when several people are in frame
and one is talking, which face is the speaker? Photographing the wrong one
files somebody's words under somebody else's face, and nothing downstream
knows to doubt it.

``asd`` answers it from agreement between mouth motion and sound.
``evidence`` accumulates agreement across encounters, which is how a
pairing made at capture time gets confirmed or reopened later.
"""

from .asd import AsdConfig, AsdResult, SpeakerScore, audio_envelope, pick_speaker, score_candidates
from .evidence import (
    BindingConfig,
    Evidence,
    Hypothesis,
    Refusal,
    evidence_from_encounter,
)
from .mouth import LIP_INDICES, MouthSample, motion_energy, mouth_openness, openness_series

__all__ = [
    "LIP_INDICES",
    "AsdConfig",
    "AsdResult",
    "BindingConfig",
    "Evidence",
    "Hypothesis",
    "MouthSample",
    "Refusal",
    "SpeakerScore",
    "audio_envelope",
    "evidence_from_encounter",
    "motion_energy",
    "mouth_openness",
    "openness_series",
    "pick_speaker",
    "score_candidates",
]
