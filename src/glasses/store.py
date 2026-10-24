"""A local, file-backed store for M0.

M0 runs entirely on a laptop, so requiring Postgres to try the pipeline would
be friction for nothing. This keeps the gallery in an ``.npz`` beside a JSON
sidecar, exposes the same ``Gallery`` interface the pipeline already takes, and
can be thrown away and rebuilt.

It is not a stand-in for ``db/schema.sql``. It holds no encounters, no facts
and no voiceprints, and it is single-process. When ingest arrives, this becomes
the thing you migrate off.

Consent is recorded rather than assumed: ``enroll`` will not write a person
without it. That is the PRD's posture — the enrollment flow is a real feature,
not a database insert — and it is cheaper to honour from the first row than to
retrofit onto a store you should not have had.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .identity.gallery import GalleryEntry, InMemoryGallery
from .identity.types import Embedding


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Person:
    id: str
    display_name: str
    consented_at: str
    consent_note: str = ""
    created_at: str = field(default_factory=_now)


@dataclass
class TemplateRecord:
    id: str
    person_id: str
    model_name: str
    model_version: str
    quality_score: float
    source: str
    created_at: str = field(default_factory=_now)
    #: "face" or "voice". Defaulted so stores written before voice existed
    #: still load, and because a bare template is a face template
    #: historically. Galleries are always built per-kind.
    kind: str = "face"
    #: Set on the wearer's own voiceprints. Their speech is never attributed
    #: to a conversational partner, so it must be distinguishable.
    is_wearer: bool = False


class LocalStore:
    """People plus their face templates, persisted to a directory."""

    VECTORS = "templates.npz"
    INDEX = "index.json"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.people: dict[str, Person] = {}
        self.templates: list[TemplateRecord] = []
        self._vectors: dict[str, np.ndarray] = {}

    # --- persistence -------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str) -> LocalStore:
        """Load, or return an empty store if nothing is there yet."""
        store = cls(path)
        index_path = store.path / cls.INDEX
        if not index_path.exists():
            return store

        index = json.loads(index_path.read_text())
        store.people = {p["id"]: Person(**p) for p in index["people"]}
        store.templates = [TemplateRecord(**t) for t in index["templates"]]

        vectors_path = store.path / cls.VECTORS
        if vectors_path.exists():
            with np.load(vectors_path) as data:
                store._vectors = {key: data[key] for key in data.files}
        return store

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / self.INDEX).write_text(
            json.dumps(
                {
                    "people": [asdict(p) for p in self.people.values()],
                    "templates": [asdict(t) for t in self.templates],
                },
                indent=2,
            )
        )
        np.savez(self.path / self.VECTORS, **self._vectors)

    # --- writing -----------------------------------------------------------

    def enroll(
        self,
        display_name: str,
        embeddings: list[Embedding],
        *,
        consent_note: str,
        quality_scores: list[float] | None = None,
        source: str = "",
        person_id: str | None = None,
        kind: str = "face",
        is_wearer: bool = False,
    ) -> str:
        """Add templates for a person, creating them if they are new.

        ``consent_note`` is required and must say something. A person enrolled
        without a record of who agreed and when is exactly the row the privacy
        section exists to prevent.
        """
        if not consent_note.strip():
            raise ValueError("consent_note is required: record who agreed to be enrolled and when")
        if not embeddings:
            raise ValueError("no embeddings to enroll")

        scores = quality_scores or [0.0] * len(embeddings)
        if len(scores) != len(embeddings):
            raise ValueError(f"got {len(embeddings)} embeddings but {len(scores)} quality scores")

        if person_id is None:
            person_id = uuid.uuid4().hex
            self.people[person_id] = Person(
                id=person_id,
                display_name=display_name,
                consented_at=_now(),
                consent_note=consent_note,
            )
        elif person_id not in self.people:
            raise KeyError(f"no such person: {person_id}")

        for embedding, score in zip(embeddings, scores, strict=True):
            template_id = uuid.uuid4().hex
            self.templates.append(
                TemplateRecord(
                    id=template_id,
                    person_id=person_id,
                    model_name=embedding.model_name,
                    model_version=embedding.model_version,
                    quality_score=float(score),
                    source=source,
                    kind=kind,
                    is_wearer=is_wearer,
                )
            )
            self._vectors[template_id] = np.asarray(embedding.vector, dtype=np.float32)
        return person_id

    def forget(self, person_id: str) -> int:
        """Remove a person and every template of theirs, in one operation.

        This is the deletion-on-request path. It is a single call on purpose:
        a deletion that has to be remembered in three places is one that will
        eventually be done in two.
        """
        if person_id not in self.people:
            raise KeyError(f"no such person: {person_id}")

        removed = [t for t in self.templates if t.person_id == person_id]
        for template in removed:
            self._vectors.pop(template.id, None)
        self.templates = [t for t in self.templates if t.person_id != person_id]
        del self.people[person_id]
        return len(removed)

    # --- reading -----------------------------------------------------------

    def gallery(self, kind: str = "face") -> InMemoryGallery:
        """Templates of one kind, as a searchable gallery.

        Always scoped by kind. CAM++ voiceprints and ArcFace faces are both
        512-d, so a gallery holding the two would compare them happily and
        return a plausible number; model_name filtering in the search would
        catch it, but relying on that alone leaves no margin.

        The wearer is excluded: they are not someone you are trying to
        recognise across encounters, and leaving them in means every one of
        their own utterances matches them as a "conversational partner".
        """
        return InMemoryGallery(
            [
                GalleryEntry(
                    template_id=t.id,
                    person_id=t.person_id,
                    embedding=Embedding(
                        vector=self._vectors[t.id],
                        model_name=t.model_name,
                        model_version=t.model_version,
                    ),
                )
                for t in self.templates
                if t.id in self._vectors and t.kind == kind and not t.is_wearer
            ]
        )

    def wearer_embeddings(self) -> list[Embedding]:
        """The wearer's enrolled voiceprints, for the pipeline's wearer check."""
        return [
            Embedding(
                vector=self._vectors[t.id],
                model_name=t.model_name,
                model_version=t.model_version,
            )
            for t in self.templates
            if t.is_wearer and t.id in self._vectors
        ]

    def wearer(self) -> Person | None:
        ids = {t.person_id for t in self.templates if t.is_wearer}
        return self.people.get(next(iter(ids))) if len(ids) == 1 else None

    def name_of(self, person_id: str | None) -> str:
        if person_id is None:
            return "unknown"
        person = self.people.get(person_id)
        return person.display_name if person else person_id[:8]

    def find_by_name(self, display_name: str) -> Person | None:
        needle = display_name.strip().casefold()
        return next((p for p in self.people.values() if p.display_name.casefold() == needle), None)

    def template_count(self, person_id: str, kind: str | None = None) -> int:
        return sum(
            1
            for t in self.templates
            if t.person_id == person_id and (kind is None or t.kind == kind)
        )

    def template_count_by_kind(self, kind: str) -> int:
        return sum(1 for t in self.templates if t.kind == kind)

    def models(self, kind: str | None = None) -> set[tuple[str, str]]:
        """Which (model_name, model_version) pairs the store holds.

        More than one means part of the gallery is invisible to any given
        probe, which is the symptom of a half-finished re-embedding.
        """
        return {
            (t.model_name, t.model_version)
            for t in self.templates
            if kind is None or t.kind == kind
        }

    def __len__(self) -> int:
        return len(self.people)
