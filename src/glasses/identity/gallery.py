"""The gallery: enrolled templates, searched by cosine similarity.

Two implementations behind one Protocol.

``InMemoryGallery`` is brute-force numpy. At a thousand people and a few
embeddings each this is a few thousand vectors and the search is instant — the
PRD is right that a dedicated vector database would be an operational
dependency for nothing. It is the M0 default.

``PgVectorGallery`` is the same search in the Postgres that already holds the
profiles, with an HNSW index, for when the gallery outgrows a process.

Both filter on ``(model_name, model_version)``. A search that silently spanned
two model versions would return meaningless similarities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .types import Embedding, TemplateHit
from .vectors import l2_normalize


@runtime_checkable
class Gallery(Protocol):
    def search(self, probe: Embedding, *, limit: int) -> list[TemplateHit]:
        """The ``limit`` most similar templates sharing the probe's model
        name and version, most similar first."""
        ...


@dataclass
class GalleryEntry:
    template_id: str
    person_id: str
    embedding: Embedding


class InMemoryGallery:
    """Brute-force cosine search. Rebuilds its matrix on write, which is fine
    at enrollment rates measured in people per week."""

    def __init__(self, entries: list[GalleryEntry] | None = None) -> None:
        self._entries: list[GalleryEntry] = []
        self._matrix: np.ndarray | None = None
        for entry in entries or []:
            self.add(entry)

    def add(self, entry: GalleryEntry) -> None:
        self._entries.append(entry)
        self._matrix = None

    def add_template(self, template_id: str, person_id: str, embedding: Embedding) -> None:
        self.add(GalleryEntry(template_id, person_id, embedding))

    def remove_person(self, person_id: str) -> int:
        """Drop every template for one person. Part of the deletion path, which
        has to remove embeddings for a person in a single operation."""
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.person_id != person_id]
        self._matrix = None
        return before - len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def person_ids(self) -> set[str]:
        return {e.person_id for e in self._entries}

    def _compatible(self, probe: Embedding) -> list[GalleryEntry]:
        key = probe.key()
        return [e for e in self._entries if e.embedding.key() == key]

    def search(self, probe: Embedding, *, limit: int = 64) -> list[TemplateHit]:
        entries = self._compatible(probe)
        if not entries:
            return []

        matrix = np.stack([l2_normalize(e.embedding.vector) for e in entries])
        if matrix.shape[1] != probe.dim:
            raise ValueError(
                f"probe dim {probe.dim} does not match gallery dim {matrix.shape[1]} "
                f"for model {probe.model_name}@{probe.model_version}"
            )

        similarities = matrix @ l2_normalize(probe.vector)
        take = min(limit, len(entries))
        top = np.argpartition(-similarities, take - 1)[:take]
        top = top[np.argsort(-similarities[top])]
        return [
            TemplateHit(
                template_id=entries[i].template_id,
                person_id=entries[i].person_id,
                similarity=float(similarities[i]),
            )
            for i in top
        ]


#: Tables this gallery may be pointed at. An allowlist rather than an escaped
#: identifier: the table name is interpolated into SQL, and the set of tables
#: holding embeddings is closed and known at import time.
TEMPLATE_TABLES = frozenset({"face_template", "voice_template"})

_SEARCH_SQL = """
SELECT id::text, person_id::text, 1 - (embedding <=> %(probe)s::vector) AS similarity
FROM {table}
WHERE model_name = %(model_name)s
  AND model_version = %(model_version)s
  AND deleted_at IS NULL
ORDER BY embedding <=> %(probe)s::vector
LIMIT %(limit)s
"""


class PgVectorGallery:
    """pgvector-backed search against a template table.

    ``<=>`` is cosine distance, so similarity is ``1 - distance``. The ORDER BY
    is what lets the HNSW index serve the query; filtering in Python after the
    fact would not.

    The same class serves faces and voices because the two tables differ only
    in vector width, which pgvector carries in the column type.
    """

    def __init__(self, connection, table: str = "face_template") -> None:  # type: ignore[no-untyped-def]
        if table not in TEMPLATE_TABLES:
            raise ValueError(
                f"unknown template table {table!r}; expected {sorted(TEMPLATE_TABLES)}"
            )
        self._connection = connection
        self._sql = _SEARCH_SQL.format(table=table)
        self.table = table

    def search(self, probe: Embedding, *, limit: int = 64) -> list[TemplateHit]:
        vector = l2_normalize(probe.vector)
        literal = "[" + ",".join(repr(float(v)) for v in vector) + "]"
        with self._connection.cursor() as cursor:
            cursor.execute(
                self._sql,
                {
                    "probe": literal,
                    "model_name": probe.model_name,
                    "model_version": probe.model_version,
                    "limit": limit,
                },
            )
            rows = cursor.fetchall()
        return [
            TemplateHit(template_id=row[0], person_id=row[1], similarity=float(row[2]))
            for row in rows
        ]
