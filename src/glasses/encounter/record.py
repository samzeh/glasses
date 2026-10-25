"""Turn a finished capture session into an ``Encounter``.

The missing step between the hub and the store. ``glasses.encounter`` knows
how to *describe* an interaction and ``glasses.db.store`` knows how to write
one, but nothing built the description from what a session actually produced,
so the observation hypertables stayed empty while enrollment filled up.

Pure translation, deliberately: captures and tracks in, observations out, no
model and no database. The hub supplies identities it has already resolved
rather than having them looked up here, because resolving a face costs an
embedding and the hub has usually just paid for it.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

from .types import Encounter, EncounterStatus, FaceObservation, Interval, SpeechObservation

#: How much of the frame counts as the middle. ``centred`` is what dwell
#: means, and a face at the edge of a 640px frame is someone walking past.
CENTRE_FRACTION = 0.5


class _Track(Protocol):
    """What this module needs from ``glasses.meet.Track``."""

    id: int
    first_seen_us: int
    last_seen_us: int
    frame_count: int
    best_width: float


def speech_observations(captures: Iterable[Any]) -> list[SpeechObservation]:
    """One observation per utterance.

    The wearer's own speech is kept. It is not a partner and never triggers
    anything, but dropping it here would make the recorded timeline disagree
    with what the trigger saw.
    """
    observations = []
    for item in captures:
        track = getattr(item, "track", None)
        observations.append(
            SpeechObservation(
                span=Interval(int(item.start_us), int(item.end_us)),
                person_id=getattr(item, "voice_person_id", None),
                speaker=_speaker_label(item, track),
                confidence=float(getattr(item, "voice_similarity", 0.0) or 0.0),
                is_wearer=bool(getattr(item, "is_wearer", False)),
                text=getattr(item, "text", None),
            )
        )
    return observations


def _speaker_label(item: Any, track: Any) -> int | None:
    """The diarizer's label, or the face it was bound to.

    Either one separates two speakers from each other within a recording,
    which is the only job the column has.
    """
    speaker = getattr(item, "speaker", None)
    if speaker is not None:
        return int(speaker)
    return int(track.id) if track is not None else None


def face_observations(
    tracks: Sequence[_Track],
    identities: Mapping[int, tuple[str | None, float]],
    *,
    frame_width: int | None = None,
    centre_fraction: float = CENTRE_FRACTION,
) -> list[FaceObservation]:
    """One observation per tracked face, spanning first to last sighting.

    ``identities`` maps a track id to the ``(person_id, confidence)`` the
    caller resolved. A track missing from it is an unrecognised face, which
    is most of them and is still worth recording.
    """
    seen = [t for t in tracks if getattr(t, "frame_count", 0) > 0]
    if not seen:
        return []

    widest = max(t.best_width for t in seen)
    observations = []
    for item in seen:
        person_id, confidence = identities.get(item.id, (None, 0.0))
        observations.append(
            FaceObservation(
                span=Interval(int(item.first_seen_us), int(item.last_seen_us)),
                person_id=person_id,
                confidence=float(confidence),
                frame_count=int(item.frame_count),
                is_primary=item.best_width >= widest,
                centred=_is_centred(item, frame_width, centre_fraction),
            )
        )
    return observations


def _is_centred(track: Any, frame_width: int | None, centre_fraction: float) -> bool:
    """True when the face sat in the middle band of the frame.

    Without a frame width there is nothing to measure against, so the answer
    is yes: a single-camera session where every face is the subject should not
    be recorded as nobody dwelling.
    """
    box = getattr(track, "bbox", None)
    if frame_width is None or box is None:
        return True
    centre = (box.x1 + box.x2) / 2
    margin = frame_width * (1 - centre_fraction) / 2
    return margin <= centre <= frame_width - margin


def build_encounter(
    *,
    device_id: str,
    started_us: int,
    ended_us: int | None,
    faces: Sequence[FaceObservation],
    speech: Sequence[SpeechObservation],
    location: str | None = None,
    encounter_id: str | None = None,
) -> Encounter:
    """Assemble the encounter the store will write.

    The status is decided the same way ``EncounterSession`` decides it: an
    encounter that never heard anyone but the wearer saw no conversation and
    is recorded as expired, so the discard path stays visible in the data
    rather than looking like a session that was never captured.
    """
    partner_spoke = any(s.is_partner for s in speech)
    return Encounter(
        id=encounter_id or str(uuid.uuid4()),
        device_id=device_id,
        started_us=int(started_us),
        status=EncounterStatus.CLOSED if partner_spoke else EncounterStatus.EXPIRED,
        ended_us=None if ended_us is None else int(ended_us),
        location=location,
        faces=list(faces),
        speech=list(speech),
    )


__all__ = ["build_encounter", "face_observations", "speech_observations"]
