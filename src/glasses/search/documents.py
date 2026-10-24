"""The two document shapes, and how real data becomes them.

The indexer never sees a ``Utterance``, a ``LocalStore`` row or a Postgres
cursor — it sees these. That is the seam that lets the source change
without reshaping the index: today the input is the transcription pipeline
plus the local store, tomorrow it is a ``SELECT`` over ``fact`` joined to
``person``, and both arrive here as the same dataclass.

Document ids are deterministic, which makes the sync idempotent:

* a segment is keyed by ``encounter_id`` plus its span, so re-transcribing
  with a better Whisper model updates the text in place rather than
  leaving two versions of the same sentence in the index;
* a fact is keyed by a digest of encounter, person, kind and value,
  because a fact has no natural key and re-extraction must not duplicate
  it.

``person_id`` is nullable on both, matching ``db/schema.sql``. Unattributed
speech and unassigned facts are real states, not errors, and they have to
be searchable — "what did someone say about biotech?" is answerable from a
segment whose speaker was never identified.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .config import SOURCE_CAPTURE
from .mappings import FACT_TYPE, SEGMENT_TYPE, SEMANTIC_FIELD


def _digest(*parts: object) -> str:
    """A short, stable id from the fields that identify a row.

    Truncated SHA-256 rather than a uuid: the point is that the same input
    produces the same id in a different process, on a different day, from a
    different source database.
    """
    joined = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]


def _clean(doc: dict[str, Any]) -> dict[str, Any]:
    """Drop nulls. A missing keyword and a null keyword are the same thing
    to a query, and omitting it keeps ``dynamic: strict`` happy about
    optional fields that were never populated."""
    return {key: value for key, value in doc.items() if value is not None}


@dataclass(frozen=True)
class TranscriptSegment:
    """One attributed utterance, ready to index."""

    encounter_id: str
    start_ms: int
    end_ms: int
    text: str
    person_id: str | None = None
    person_name: str | None = None
    is_wearer: bool = False
    speaker: str | None = None
    occurred_at: str | None = None
    location: str | None = None
    source: str = SOURCE_CAPTURE
    model: str | None = None

    @property
    def doc_id(self) -> str:
        return f"{self.encounter_id}:{self.start_ms}-{self.end_ms}"

    def to_doc(self, *, semantic: bool = False) -> dict[str, Any]:
        doc = _clean(
            {
                "doc_type": SEGMENT_TYPE,
                "text": self.text,
                "person_id": self.person_id,
                "person_name": self.person_name,
                "encounter_id": self.encounter_id,
                "start_ms": self.start_ms,
                "end_ms": self.end_ms,
                "duration_ms": max(0, self.end_ms - self.start_ms),
                "is_wearer": self.is_wearer,
                "speaker": self.speaker,
                "occurred_at": self.occurred_at,
                "location": self.location,
                "source": self.source,
                "model": self.model,
            }
        )
        if semantic:
            # Written explicitly rather than with copy_to: the same string
            # in two fields is obvious, and it does not depend on copy_to
            # being supported into a semantic_text field.
            doc[SEMANTIC_FIELD] = self.text
        return doc

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> TranscriptSegment:
        """Build from a database row.

        Accepts the column names a ``SELECT`` over the Postgres schema
        would produce, so the eventual Postgres source needs no mapping
        layer of its own.
        """
        return cls(
            encounter_id=str(row["encounter_id"]),
            start_ms=int(row["start_ms"]),
            end_ms=int(row["end_ms"]),
            text=str(row["text"]),
            person_id=_optional_str(row.get("person_id")),
            person_name=_optional_str(row.get("person_name") or row.get("display_name")),
            is_wearer=bool(row.get("is_wearer", False)),
            speaker=_optional_str(row.get("speaker")),
            occurred_at=_optional_str(row.get("occurred_at") or row.get("started_at")),
            location=_optional_str(row.get("location")),
            source=str(row.get("source", SOURCE_CAPTURE)),
            model=_optional_str(row.get("model")),
        )


@dataclass(frozen=True)
class PersonFact:
    """One extracted fact about a person, with its evidence.

    Field for field the ``fact`` table: ``kind``, ``value``,
    ``evidence_span``, ``confidence``, and a ``person_id`` that may be
    null.
    """

    encounter_id: str
    kind: str
    value: str
    person_id: str | None = None
    person_name: str | None = None
    evidence_span: str | None = None
    evidence_start_ms: int | None = None
    evidence_end_ms: int | None = None
    confidence: float = 0.0
    occurred_at: str | None = None
    source: str = SOURCE_CAPTURE

    @property
    def doc_id(self) -> str:
        return _digest(self.encounter_id, self.person_id, self.kind, self.value)

    def to_doc(self, *, semantic: bool = False) -> dict[str, Any]:
        doc = _clean(
            {
                "doc_type": FACT_TYPE,
                "kind": self.kind,
                "value": self.value,
                "person_id": self.person_id,
                "person_name": self.person_name,
                "encounter_id": self.encounter_id,
                "evidence_span": self.evidence_span,
                "evidence_start_ms": self.evidence_start_ms,
                "evidence_end_ms": self.evidence_end_ms,
                "confidence": float(self.confidence),
                "occurred_at": self.occurred_at,
                "source": self.source,
            }
        )
        if semantic:
            doc[SEMANTIC_FIELD] = f"{self.kind}: {self.value}"
        return doc

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> PersonFact:
        return cls(
            encounter_id=str(row["encounter_id"]),
            kind=str(row["kind"]),
            value=str(row["value"]),
            person_id=_optional_str(row.get("person_id")),
            person_name=_optional_str(row.get("person_name") or row.get("display_name")),
            evidence_span=_optional_str(row.get("evidence_span")),
            evidence_start_ms=_optional_int(row.get("evidence_start_ms")),
            evidence_end_ms=_optional_int(row.get("evidence_end_ms")),
            confidence=float(row.get("confidence") or 0.0),
            occurred_at=_optional_str(row.get("occurred_at") or row.get("extracted_at")),
            source=str(row.get("source", SOURCE_CAPTURE)),
        )


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)  # type: ignore[arg-type]


def segments_from_utterances(
    utterances: Iterable[Any],
    *,
    encounter_id: str,
    occurred_at: str | None = None,
    location: str | None = None,
    source: str = SOURCE_CAPTURE,
    model: str | None = None,
    name_of: Any = None,
) -> list[TranscriptSegment]:
    """Transcription output, as indexable segments.

    Takes ``glasses.voice.transcribe.Utterance`` values — duck-typed rather
    than imported, so this module stays free of the voice stack and the
    test suite needs none of it. ``name_of`` is an optional
    ``store.name_of``-shaped callable for filling in a display name the
    utterance did not carry.
    """
    out: list[TranscriptSegment] = []
    for utterance in utterances:
        name = utterance.person_name
        if name is None and name_of is not None and utterance.person_id is not None:
            name = name_of(utterance.person_id)
        out.append(
            TranscriptSegment(
                encounter_id=encounter_id,
                start_ms=int(round(utterance.start_s * 1000.0)),
                end_ms=int(round(utterance.end_s * 1000.0)),
                text=utterance.text,
                person_id=utterance.person_id,
                person_name=name,
                is_wearer=bool(utterance.is_wearer),
                speaker=None if utterance.speaker is None else str(utterance.speaker),
                occurred_at=occurred_at,
                location=location,
                source=source,
                model=model,
            )
        )
    return out


def bulk_body(
    index: str,
    documents: Sequence[TranscriptSegment | PersonFact],
    *,
    semantic: bool = False,
) -> str:
    """The ndjson for one ``_bulk`` request.

    ``index`` rather than ``create``: re-running the sync must overwrite,
    since the derived copy is allowed to be rebuilt and a conflict would
    turn a retry into a failure.
    """
    lines: list[str] = []
    for document in documents:
        action = {"index": {"_index": index, "_id": document.doc_id}}
        lines.append(_json(action))
        lines.append(_json(document.to_doc(semantic=semantic)))
    return "".join(f"{line}\n" for line in lines)


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, separators=(",", ":"))
