"""Index mappings for the two derived document types.

Elastic is a **derived index, not the system of record**. Postgres holds
the truth — people, encounters, facts — and this is a searchable copy that
can be dropped and rebuilt at any time. Three things follow from that, and
they are the whole design:

* **Deterministic document ids.** Re-indexing the same segment overwrites
  it instead of appending a second copy, so the sync is idempotent and can
  be re-run after a crash without a dedupe pass.
* **Explicit mappings, ``dynamic: strict``.** A field name typo fails the
  bulk request instead of silently inventing a field, and a derived index
  has no reason to accept a shape its writer did not intend. The cost is
  that adding a field means editing this file, which is the right cost.
* **``source`` on every document.** ``capture`` is real, ``seed`` is
  synthetic demo data. One keyword makes "show me only real encounters"
  and "delete every fake person" single queries rather than careful ones.

``person_name`` is a ``keyword`` with a ``text`` subfield rather than the
other way round. The exact term is what a filter needs ("facts about
Test Person"), and ES|QL compares keywords; the analysed subfield is there for
"did I meet a Test?" where the question is fuzzy.
"""

from __future__ import annotations

from typing import Any

#: Written into every document. Two indices rather than one is the choice
#: here, so this is strictly redundant — but Agent Builder tools name an
#: index, and a document that cannot say what it is becomes unreadable the
#: moment anything aggregates across the two.
SEGMENT_TYPE = "transcript_segment"
FACT_TYPE = "person_fact"

#: Added when semantic search is switched on. Elastic Serverless ships a
#: default inference endpoint, so ``semantic_text`` needs no model
#: deployment of its own — but it is off by default because an index that
#: cannot be written without an inference endpoint is a demo that can fail
#: on stage. See ``glasses search-setup --semantic``.
SEMANTIC_FIELD = "text_semantic"


def _person_name() -> dict[str, Any]:
    return {"type": "keyword", "fields": {"text": {"type": "text"}}}


def segment_mappings(*, semantic: bool = False) -> dict[str, Any]:
    """One attributed utterance: who said what, when, in which encounter."""
    properties: dict[str, Any] = {
        "doc_type": {"type": "keyword"},
        "text": {"type": "text"},
        "person_id": {"type": "keyword"},
        "person_name": _person_name(),
        "encounter_id": {"type": "keyword"},
        "start_ms": {"type": "long"},
        "end_ms": {"type": "long"},
        "duration_ms": {"type": "long"},
        # The wearer's own speech. Kept rather than dropped: "what did I
        # promise him?" is as much a question as "what did he say?". Every
        # tool filters on it explicitly, because the wearer's words must
        # never read as a fact about the person opposite.
        "is_wearer": {"type": "boolean"},
        "speaker": {"type": "keyword"},
        "occurred_at": {"type": "date"},
        "location": {"type": "keyword"},
        "source": {"type": "keyword"},
        "model": {"type": "keyword"},
    }
    if semantic:
        properties[SEMANTIC_FIELD] = {"type": "semantic_text"}
    return {"dynamic": "strict", "properties": properties}


def fact_mappings(*, semantic: bool = False) -> dict[str, Any]:
    """One extracted fact, with the transcript span it came from.

    Mirrors the ``fact`` table in ``db/schema.sql`` field for field,
    including the nullable ``person_id``: when two non-wearer faces were
    visible at once the fact belongs to the encounter and waits, and that
    unassigned state has to survive into search rather than being resolved
    by a guess on the way in.
    """
    properties: dict[str, Any] = {
        "doc_type": {"type": "keyword"},
        "kind": {"type": "keyword"},
        "topic": {"type": "keyword"},
        "value": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
        "person_id": {"type": "keyword"},
        "person_name": _person_name(),
        "encounter_id": {"type": "keyword"},
        # A fact with no evidence is a fact you cannot debug. This is the
        # quote a human reads to decide whether the profile is wrong.
        "evidence_span": {"type": "text"},
        "evidence_start_ms": {"type": "long"},
        "evidence_end_ms": {"type": "long"},
        "confidence": {"type": "float"},
        "occurred_at": {"type": "date"},
        "source": {"type": "keyword"},
    }
    if semantic:
        properties[SEMANTIC_FIELD] = {"type": "semantic_text"}
    return {"dynamic": "strict", "properties": properties}


def all_mappings(*, semantic: bool = False) -> dict[str, dict[str, Any]]:
    """Keyed by document type, for the setup command to walk."""
    return {
        SEGMENT_TYPE: segment_mappings(semantic=semantic),
        FACT_TYPE: fact_mappings(semantic=semantic),
    }
