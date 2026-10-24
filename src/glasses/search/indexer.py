"""Postgres → Elastic, one way only.

The sync is deliberately dumb: build documents, bulk them in, overwrite
what was there. There is no write path back, no field only Elastic knows,
and nothing in the product reads an answer out of Elastic and stores it as
truth. If the index is lost, ``glasses search-index`` rebuilds it; if the
index and the database disagree, the database is right.

That constraint is what makes it safe to let an LLM agent query this at
all. An agent with a search tool over a derived copy can be wrong about
what it finds; an agent with a search tool over the system of record can be
wrong about what is true.

Bulk errors are reported per document rather than raised. One malformed
segment in a batch of four hundred should not lose the other three hundred
and ninety-nine, and the failing id plus Elastic's own reason is what makes
it fixable.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Any

from .client import ElasticClient, ElasticError
from .config import SOURCE_CAPTURE, SOURCE_SEED, ElasticConfig
from .documents import PersonFact, TranscriptSegment, bulk_body
from .mappings import fact_mappings, segment_mappings

DEFAULT_BATCH = 400


@dataclass
class IndexReport:
    """What one ``index_*`` call did. Printed verbatim by the CLI."""

    index: str
    indexed: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    batches: int = 0

    @property
    def ok(self) -> bool:
        return not self.failures

    def __str__(self) -> str:
        line = f"{self.index}: {self.indexed} document(s) in {self.batches} batch(es)"
        if self.failures:
            line += f", {len(self.failures)} failed"
        return line


@dataclass(frozen=True)
class IndexState:
    """Whether an index is there, and what is in it."""

    index: str
    exists: bool
    total: int = 0
    capture: int = 0
    seed: int = 0

    def __str__(self) -> str:
        if not self.exists:
            return f"{self.index}: MISSING — run glasses search-setup"
        return f"{self.index}: {self.total} docs ({self.capture} capture, {self.seed} seed)"


class Indexer:
    """Builds and maintains the derived index."""

    def __init__(
        self,
        client: ElasticClient,
        config: ElasticConfig,
        *,
        semantic: bool = False,
    ) -> None:
        self.client = client
        self.config = config
        self.semantic = semantic

    # --- schema ------------------------------------------------------------

    def ensure_indices(self) -> dict[str, str]:
        """Create what is missing, update the mapping of what is there.

        Mapping updates are additive in Elasticsearch, so this makes a new
        field appear on an existing index but will refuse a changed type —
        correctly. A type change means ``search-setup --recreate``, because
        the only honest way to change a derived index's shape is to rebuild
        it from the source.
        """
        plans = (
            (self.config.segments_index, segment_mappings(semantic=self.semantic)),
            (self.config.facts_index, fact_mappings(semantic=self.semantic)),
        )
        outcome: dict[str, str] = {}
        for index, mappings in plans:
            if self.client.index_exists(index):
                self.client.put_mapping(index, mappings)
                outcome[index] = "mapping updated"
            else:
                self.client.create_index(index, mappings)
                outcome[index] = "created"
        return outcome

    def recreate_indices(self) -> dict[str, str]:
        """Drop and rebuild both indices. Safe because this is a copy."""
        for index in self.config.indices:
            try:
                self.client.delete_index(index)
            except ElasticError as error:
                if error.status != 404:
                    raise
        return self.ensure_indices()

    # --- writing -----------------------------------------------------------

    def index_segments(
        self,
        segments: Iterable[TranscriptSegment],
        *,
        batch_size: int = DEFAULT_BATCH,
    ) -> IndexReport:
        return self._bulk(self.config.segments_index, segments, batch_size)

    def index_facts(
        self,
        facts: Iterable[PersonFact],
        *,
        batch_size: int = DEFAULT_BATCH,
    ) -> IndexReport:
        return self._bulk(self.config.facts_index, facts, batch_size)

    def _bulk(
        self,
        index: str,
        documents: Iterable[TranscriptSegment | PersonFact],
        batch_size: int,
    ) -> IndexReport:
        report = IndexReport(index=index)
        stream = iter(documents)
        while batch := list(islice(stream, batch_size)):
            report.batches += 1
            response = self.client.bulk(bulk_body(index, batch, semantic=self.semantic))
            indexed, failures = _read_bulk(response, batch)
            report.indexed += indexed
            report.failures.extend(failures)
        return report

    # --- reading and removal ----------------------------------------------

    def state(self) -> list[IndexState]:
        return [self._state(index) for index in self.config.indices]

    def _state(self, index: str) -> IndexState:
        if not self.client.index_exists(index):
            return IndexState(index=index, exists=False)
        return IndexState(
            index=index,
            exists=True,
            total=self.client.count(index),
            capture=self.client.count(index, {"term": {"source": SOURCE_CAPTURE}}),
            seed=self.client.count(index, {"term": {"source": SOURCE_SEED}}),
        )

    def clear(self, source: str) -> dict[str, int]:
        """Delete every document from one source.

        This is why ``source`` is on every document: removing the synthetic
        demo data is one call per index and cannot catch a real capture in
        the blast radius.
        """
        removed: dict[str, int] = {}
        for index in self.config.indices:
            if not self.client.index_exists(index):
                continue
            response = self.client.delete_by_query(index, {"term": {"source": source}})
            removed[index] = int(response.get("deleted", 0))
        return removed


def _read_bulk(
    response: Any,
    batch: Sequence[TranscriptSegment | PersonFact],
) -> tuple[int, list[tuple[str, str]]]:
    """Count what landed, and name what did not.

    A bulk response is per-item: ``errors: true`` says only that at least
    one failed, so the items have to be walked to find which.
    """
    items = (response or {}).get("items") or []
    if not items:
        # No item list and no error: the request was accepted whole. This
        # is the fake-transport path and the "errors": false shortcut.
        return (len(batch) if response is not None and not response.get("errors") else 0), []

    indexed = 0
    failures: list[tuple[str, str]] = []
    for position, item in enumerate(items):
        result = next(iter(item.values())) if item else {}
        error = result.get("error")
        if error is None:
            indexed += 1
            continue
        doc_id = result.get("_id") or (
            batch[position].doc_id if position < len(batch) else "unknown"
        )
        reason = error.get("reason") if isinstance(error, dict) else str(error)
        failures.append((str(doc_id), str(reason)))
    return indexed, failures
