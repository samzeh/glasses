"""Captures to recall: the camera path's half of the cold loop.

``glasses meet`` already does the hard part. It records camera and mic on
one clock, tracks faces, and uses lip-sync to decide which visible face
produced each utterance — so every capture arrives already bound to a
face, rather than two streams being reconciled afterwards.

What it stopped short of was meaning. This module takes those captures
through the rest: transcribe each one, extract facts, enroll anyone who
introduced themselves, and index both halves where ``glasses ask`` can
reach them.

**Identity comes from the face, not the voice**, and that is the point of
this path existing beside the audio-only one. Measured on this project's
own recordings, two people's voiceprints scored 0.800 and 0.806 against
the same enrolled wearer — the genuine and impostor ranges overlap, so no
threshold separates them. Face recognition runs at a 0.40 cosine with
twenty times the latency budget spare. So the face answers *who*, and the
voice is left doing what it is good at: saying *when* somebody spoke.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .voice.transcribe import Utterance


@dataclass
class FaceIdentity:
    """What the face gallery made of one tracked face."""

    person_id: str | None = None
    name: str | None = None
    similarity: float = 0.0
    embedding: Any = None
    note: str = ""

    @property
    def known(self) -> bool:
        return self.person_id is not None


def identify_track(face_pipeline: Any, store: Any, track: Any) -> FaceIdentity:
    """Resolve one tracked face against the gallery.

    The embedding comes back whether or not the person is known, because
    an unknown face is exactly the one auto-enrollment needs a template
    for. A face that fails the quality gate returns neither: embedding a
    frame the gate rejected is how a bad template enters a gallery that
    then recognises the wrong person with it.
    """
    if track is None or track.best_frame is None:
        return FaceIdentity(note="no frame")

    result = face_pipeline.process_frame(track.best_frame)
    if not result.faces:
        return FaceIdentity(note="no face in the kept frame")

    from .meet import iou

    # Other faces in the frame belong to other tracks.
    best = max(result.faces, key=lambda f: iou(f.detection.bbox, track.bbox))
    if not best.quality.passed:
        return FaceIdentity(
            note=(
                f"rejected ({', '.join(best.quality.failures)}) "
                f"iod={best.quality.interocular_px:.0f}px "
                f"yaw={best.quality.pose.yaw:+.0f} pitch={best.quality.pose.pitch:+.0f}"
            )
        )

    match = best.match
    if match is None or match.person_id is None:
        return FaceIdentity(embedding=best.embedding, note="not recognised")
    return FaceIdentity(
        person_id=match.person_id,
        name=store.name_of(match.person_id),
        similarity=float(match.similarity),
        embedding=best.embedding,
        note="",
    )


def label_captures(
    captures: Sequence[Any], face_pipeline: Any, store: Any
) -> dict[int, FaceIdentity]:
    """Identify every distinct face track exactly once.

    Several utterances usually share a track, and the embedder is the
    expensive stage, so this is keyed by track rather than by capture.
    """
    identities: dict[int, FaceIdentity] = {}
    for capture in captures:
        track = getattr(capture, "track", None)
        if track is None or track.id in identities:
            continue
        identities[track.id] = identify_track(face_pipeline, store, track)
    return identities


def speaker_label(capture: Any, identity: FaceIdentity | None) -> str:
    """The name the transcript and the extractor will use for this speaker.

    An unassigned capture gets a label of its own rather than being
    merged into a single "unknown": two unidentified people in one
    conversation are two people, and collapsing them is how one person's
    facts land on the other.
    """
    if capture.is_wearer:
        return "you"
    if identity is not None and identity.name:
        return identity.name
    # Lip-sync is the preferred binding and the one that can fail hardest:
    # a quiet mouth, a face turned away, or a correlation that never
    # clears its floor leaves the utterance with no face at all. Falling
    # through to the voice gallery and then to the diarizer's own label
    # keeps two speakers apart even then, which is the difference between
    # facts landing on the right person and landing on "unknown".
    if getattr(capture, "voice_name", None):
        return str(capture.voice_name)
    track = getattr(capture, "track", None)
    if track is not None:
        return f"Speaker {track.id}"
    speaker = getattr(capture, "speaker", None)
    if speaker is not None:
        return f"Speaker {speaker}"
    return "Unassigned speaker"


def utterances_from_captures(
    captures: Sequence[Any],
    identities: dict[int, FaceIdentity],
    transcriber: Any,
    *,
    audio_start_us: int,
) -> list[Utterance]:
    """Transcribe each capture and attach the face's identity to its words.

    A capture whose transcript comes back empty is dropped: a segment with
    no words is not an utterance, and it would otherwise reach the
    extractor as a speaker who said nothing.
    """
    from .meet import us_to_seconds

    out: list[Utterance] = []
    for capture in captures:
        text = (transcriber.transcribe(capture.audio) or "").strip()
        if not text:
            continue
        track = getattr(capture, "track", None)
        identity = identities.get(track.id) if track is not None else None
        # Face first, then the voice gallery. Both can come back empty, and
        # the label falls through the same order so the transcript and the
        # facts agree about who this was.
        person_id = (identity.person_id if identity else None) or capture.voice_person_id
        name = (identity.name if identity and identity.name else None) or capture.voice_name
        out.append(
            Utterance(
                text=text,
                start_s=us_to_seconds(capture.start_us - audio_start_us),
                end_s=us_to_seconds(capture.end_us - audio_start_us),
                speaker=track.id if track is not None else capture.speaker,
                person_id=person_id,
                person_name=name if person_id else None,
                is_wearer=capture.is_wearer,
            )
        )
    return out


def face_templates_by_label(
    captures: Sequence[Any], identities: dict[int, FaceIdentity]
) -> dict[str, list[Any]]:
    """Face embeddings for each speaker the gallery could not place.

    Keyed by the same label the transcript used, so auto-enrollment can
    join a heard name to a face. Known people are skipped — they are
    matched, not re-created — and so is the wearer.
    """
    table: dict[str, list[Any]] = {}
    for capture in captures:
        track = getattr(capture, "track", None)
        if capture.is_wearer or track is None or capture.voice_person_id:
            continue
        identity = identities.get(track.id)
        if identity is None or identity.known or identity.embedding is None:
            continue
        label = speaker_label(capture, identity)
        bucket = table.setdefault(label, [])
        # One template per track: the embedding is the track's best frame,
        # so every capture on that track would contribute the same vector,
        # and a gallery of duplicates scores exactly as one would.
        if not bucket:
            bucket.append(identity.embedding)
    return table


__all__ = [
    "FaceIdentity",
    "face_templates_by_label",
    "identify_track",
    "label_captures",
    "speaker_label",
    "utterances_from_captures",
]
