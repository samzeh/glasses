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

#: Only ``voice_template`` has the column, because only a voice can be the
#: wearer's own.
_WEARER_TABLES = frozenset({"voice_template"})

_SEARCH_SQL = """
SELECT id::text, person_id::text, 1 - (embedding <=> %(probe)s::vector) AS similarity
FROM {table}
WHERE model_name = %(model_name)s
  AND model_version = %(model_version)s
  AND deleted_at IS NULL
  {extra}
ORDER BY embedding <=> %(probe)s::vector
LIMIT %(limit)s
"""

_DIM_SQL = """
SELECT atttypmod
FROM pg_attribute
WHERE attrelid = %s::regclass AND attname = 'embedding' AND NOT attisdropped
"""

#: pgvector's own default. Raising ef_search costs search time, so the floor
#: stays where pgvector put it and only the *requested* limit lifts it.
DEFAULT_EF_SEARCH = 40


def to_pgvector(vector: np.ndarray) -> str:
    """A pgvector literal. Normalized first, because every similarity in this
    project is a cosine between unit vectors."""
    return "[" + ",".join(repr(float(v)) for v in l2_normalize(vector)) + "]"


class PgVectorGallery:
    """pgvector-backed search against a template table.

    ``<=>`` is cosine distance, so similarity is ``1 - distance``. The ORDER BY
    is what lets the HNSW index serve the query; filtering in Python after the
    fact would not.

    The same class serves faces and voices because the two tables differ only
    in vector width, which pgvector carries in the column type.

    Two index settings are applied per search, and they are not tuning — they
    are the difference between this returning the right rows and returning a
    plausible subset of them:

    * ``hnsw.ef_search`` defaults to 40, so an unmodified ``LIMIT 64`` comes
      back with 40 rows whenever the planner uses the vector index. The
      matcher's default ``search_limit`` *is* 64, so the shortfall is the
      normal case rather than an edge one, and it is invisible: fewer
      candidates reach the top-3 aggregation and the person is scored on
      whatever survived.
    * ``hnsw.iterative_scan`` (pgvector 0.8+) is what makes the
      ``(model_name, model_version)`` filter safe. An HNSW scan finds the
      nearest rows *before* the filter is applied, so a gallery part-way
      through a re-embedding can return nothing at all for the new model
      while thousands of its templates sit in the table. Iterative scanning
      keeps going until the limit is satisfied.

    Both are set with ``set_config(..., is_local => true)`` inside a
    transaction, so nothing leaks into the caller's session, and both are
    applied best-effort: an older pgvector that does not know a setting is
    left alone rather than refused.
    """

    def __init__(  # type: ignore[no-untyped-def]
        self,
        connection,
        table: str = "face_template",
        *,
        exclude_wearer: bool = True,
    ) -> None:
        if table not in TEMPLATE_TABLES:
            raise ValueError(
                f"unknown template table {table!r}; expected {sorted(TEMPLATE_TABLES)}"
            )
        self._connection = connection
        # The wearer is never a person you are trying to recognise, and
        # leaving them in means every one of their own utterances matches
        # them as the conversational partner. LocalStore.gallery() drops
        # them for the same reason.
        self.exclude_wearer = exclude_wearer and table in _WEARER_TABLES
        self._sql = _SEARCH_SQL.format(
            table=table,
            extra="AND NOT is_wearer" if self.exclude_wearer else "",
        )
        self.table = table
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        """The width of the table's vector column, read once.

        Checked before the query so that a probe of the wrong width raises the
        same ``ValueError`` ``InMemoryGallery`` raises, rather than a driver
        exception from inside pgvector. The two galleries are meant to be
        interchangeable, and that includes how they fail.
        """
        if self._dim is None:
            with self._connection.cursor() as cursor:
                cursor.execute(_DIM_SQL, (self.table,))
                row = cursor.fetchone()
            if row is None:
                raise ValueError(f"{self.table} has no embedding column — is the schema applied?")
            self._dim = int(row[0])
        return self._dim

    def search(self, probe: Embedding, *, limit: int = 64) -> list[TemplateHit]:
        vector = to_pgvector(probe.vector)
        if probe.dim != self.dim:
            raise ValueError(
                f"probe dim {probe.dim} does not match gallery dim {self.dim} "
                f"for model {probe.model_name}@{probe.model_version}"
            )

        with self._connection.transaction(), self._connection.cursor() as cursor:
            self._tune(cursor, limit)
            cursor.execute(
                self._sql,
                {
                    "probe": vector,
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

    def explain(self, probe: Embedding, *, limit: int = 64) -> list[str]:
        """The query plan for a search, as lines.

        Worth having rather than reaching for psql, because which index serves
        this query changes what it returns, not just how fast it is: an HNSW
        scan is approximate and bounded by ``ef_search``, and a btree scan on
        ``(model_name, model_version)`` followed by a sort is exact. Two
        different answers from one piece of SQL, decided by the planner.
        """
        with self._connection.transaction(), self._connection.cursor() as cursor:
            self._tune(cursor, limit)
            cursor.execute(
                "EXPLAIN " + self._sql,
                {
                    "probe": to_pgvector(probe.vector),
                    "model_name": probe.model_name,
                    "model_version": probe.model_version,
                    "limit": limit,
                },
            )
            return [row[0] for row in cursor.fetchall()]

    def _tune(self, cursor, limit: int) -> None:  # type: ignore[no-untyped-def]
        """Make the index capable of returning as many rows as were asked for.

        Each setting goes in its own savepoint: an unknown name is an error
        that would otherwise abort the surrounding transaction and take the
        search with it.
        """
        settings = (
            ("hnsw.ef_search", str(max(limit, DEFAULT_EF_SEARCH))),
            ("hnsw.iterative_scan", "relaxed_order"),
        )
        for name, value in settings:
            try:
                with self._connection.transaction():
                    cursor.execute("SELECT set_config(%s, %s, true)", (name, value))
            except Exception:  # noqa: BLE001 - an older pgvector; the search still works
                continue
