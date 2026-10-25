"""Carrying identity onto speech too short to identify on its own.

The quality gate drops any segment under ``min_duration_s`` (3 s), and
that floor is right for what it was written for: speaker embeddings
degrade sharply below about 3.6 s, so a voiceprint built from a short
clip identifies the wrong person. It is wrong for everything else.

Measured on a real 59-second encounter: 26 segments, 7 of them long
enough to embed. The discarded 19 held the introduction — "Hi, I'm
Vrinda" three times over, and "I work at Vinta" — because people say
their name quickly and then stop. The gate was systematically destroying
the most valuable sentence in the conversation and keeping the rambling.

So the floor stays where it is for *identity*, and transcription sees
everything. A short segment cannot say who is speaking, but the diarizer
already grouped it with the long segments of the same voice, and those
can be identified. Identity therefore propagates along the speaker
label: resolve the clusters that can be resolved, then hand the answer
to their short siblings.

Nothing is invented. A short segment whose cluster was never identified
stays unidentified, and keeps its own label so it is still distinguished
from the other speaker.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

#: Shorter than this is not speech worth transcribing — a click, a
#: breath, the first phoneme of a word cut off by the segmenter. Well
#: below the identity floor on purpose: this one only has to beat noise.
MIN_TRANSCRIBE_S = 0.4


@dataclass(frozen=True)
class SpeakerIdentity:
    """What a diarized speaker cluster turned out to be."""

    person_id: str | None = None
    name: str | None = None
    is_wearer: bool = False


def identities_by_speaker(results: Sequence[Any], store: Any) -> dict[int, SpeakerIdentity]:
    """Resolve each speaker cluster from the segments long enough to embed.

    A cluster is the wearer if any of its identified segments was, and
    takes the first matched person otherwise. Both are decided only from
    segments that cleared the identity floor — the short ones are the
    consumers of this answer, never its evidence.
    """
    table: dict[int, SpeakerIdentity] = {}
    for item in results:
        speaker = item.segment.speaker
        if speaker is None or item.embedding is None:
            continue

        known = table.get(speaker)
        if item.is_wearer:
            table[speaker] = SpeakerIdentity(is_wearer=True)
            continue
        if known is not None and (known.is_wearer or known.person_id):
            continue

        match = getattr(item, "match", None)
        person_id = match.person_id if match else None
        table[speaker] = SpeakerIdentity(
            person_id=person_id,
            name=store.name_of(person_id) if person_id else None,
        )
    return table


def transcribable(results: Sequence[Any], *, min_duration_s: float = MIN_TRANSCRIBE_S) -> list[Any]:
    """Every segment worth sending to a recogniser, long or short.

    The identity floor does not apply here. An introduction is short, and
    dropping it because it is short is how an encounter loses the one
    sentence that names the person in it.
    """
    out = []
    for item in results:
        segment = item.segment
        if segment.end_s - segment.start_s < min_duration_s:
            continue
        out.append(item)
    return out


def labels_for(
    results: Sequence[Any],
    identities: dict[int, SpeakerIdentity],
    remap: dict[int, int] | None = None,
) -> list[tuple[str | None, str | None, bool]]:
    """A ``(person_id, name, is_wearer)`` label per segment.

    Segments that were identified directly keep their own answer; the
    rest inherit their cluster's. A cluster nobody could identify yields
    ``(None, None, False)``, which is honest: unknown, but still kept
    apart from the other speaker by its label.
    """
    out: list[tuple[str | None, str | None, bool]] = []
    for item in results:
        if item.embedding is not None and item.is_wearer:
            out.append((None, None, True))
            continue

        match = getattr(item, "match", None) if item.embedding is not None else None
        if match is not None and match.person_id:
            out.append((match.person_id, None, False))
            continue

        speaker = item.segment.speaker
        inherited = identities.get(speaker)
        if inherited is None:
            out.append((None, None, False))
            continue
        name = inherited.name
        if name is None and not inherited.is_wearer and speaker is not None:
            # A folded cluster has to *say* it is the same speaker, or the
            # transcript still shows six of them and the extractor is
            # asked to attribute facts across people who do not exist.
            canonical = (remap or {}).get(speaker, speaker)
            name = f"Speaker {canonical}"
        out.append((inherited.person_id, name, inherited.is_wearer))
    return out


def partner_remap(
    results: Sequence[Any],
    identities: dict[int, SpeakerIdentity],
    *,
    partners: int | None = None,
) -> dict[int, int]:
    """Fold the diarizer's surplus clusters into the people actually present.

    Clustering over-splits badly on real recordings: a measured
    two-person, 59-second encounter came back as six speakers at the
    default threshold and four at every higher one, so the same person's
    name landed on one "speaker" and their employer on another. Tuning
    did not reach two, and a voice embedding that cannot separate two
    people reliably is not going to count them.

    It does not have to. The person recording knows who is in the room.
    ``partners=1`` asserts one other person, and every non-wearer cluster
    is then the same person — which is true, and is something the
    clusterer was never in a position to know. The wearer's own clusters
    are left alone, because the wearer check is reliable.

    Returns a mapping from the diarizer's speaker label to the label to
    use instead. Without ``partners`` it changes nothing.
    """
    if not partners or partners < 1:
        return {}

    non_wearer = [
        speaker
        for speaker in sorted({item.segment.speaker for item in results} - {None})
        if not identities.get(speaker, SpeakerIdentity()).is_wearer
    ]
    if len(non_wearer) <= partners:
        return {}

    # Prefer a cluster that was actually identified as the survivor, so a
    # recognised person keeps their own label rather than being folded
    # into an anonymous one.
    def rank(speaker: int) -> tuple[int, int]:
        identity = identities.get(speaker, SpeakerIdentity())
        return (0 if identity.person_id else 1, speaker)

    keep = sorted(non_wearer, key=rank)[:partners]
    return {speaker: keep[0] for speaker in non_wearer if speaker not in keep}


def apply_remap(
    identities: dict[int, SpeakerIdentity], remap: dict[int, int]
) -> dict[int, SpeakerIdentity]:
    """Point every folded cluster at the identity of the one it joined."""
    if not remap:
        return identities
    merged = dict(identities)
    for source, target in remap.items():
        merged[source] = merged.get(target, SpeakerIdentity())
    return merged


__all__ = [
    "MIN_TRANSCRIBE_S",
    "apply_remap",
    "partner_remap",
    "SpeakerIdentity",
    "identities_by_speaker",
    "labels_for",
    "transcribable",
]
