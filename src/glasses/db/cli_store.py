"""CLI-compatible view of the Postgres store, without local persistence."""

from __future__ import annotations

from ..store import Person, TemplateRecord
from .store import PgStore


class PgCliStore(PgStore):
    @property
    def people(self) -> dict[str, Person]:
        rows = self.connection.execute(
            "SELECT id::text, display_name, consented_at, consent_note, created_at "
            "FROM person WHERE deleted_at IS NULL ORDER BY created_at"
        ).fetchall()
        return {
            row[0]: Person(
                row[0],
                row[1] or row[0][:8],
                row[2].isoformat() if row[2] else "",
                row[3] or "",
                row[4].isoformat(),
            )
            for row in rows
        }

    @property
    def templates(self) -> list[TemplateRecord]:
        rows = self.connection.execute(
            "SELECT id::text, person_id::text, model_name, model_version, "
            "quality_score, source, created_at, 'face', false FROM face_template "
            "WHERE deleted_at IS NULL UNION ALL "
            "SELECT id::text, person_id::text, model_name, model_version, "
            "quality_score, source, created_at, 'voice', is_wearer FROM voice_template "
            "WHERE deleted_at IS NULL"
        ).fetchall()
        return [TemplateRecord(*row[:6], row[6].isoformat(), *row[7:]) for row in rows]

    def save(self) -> None:
        """Writes commit in the database; nothing is written to disk."""

    def enroll(self, display_name, embeddings, *, source="", **kwargs):
        if not (kwargs.get("consent_note") or "").strip():
            raise ValueError("consent_note is required")
        return super().enroll(display_name, embeddings, source=source, **kwargs)

    def find_by_name(self, display_name):
        needle = display_name.strip().casefold()
        return next((p for p in self.people.values() if p.display_name.casefold() == needle), None)

    def template_count(self, person_id=None, kind=None):
        return (
            sum(
                t.person_id == person_id and (kind is None or t.kind == kind)
                for t in self.templates
            )
            if person_id
            else sum(kind is None or t.kind == kind for t in self.templates)
        )

    def template_count_by_kind(self, kind):
        return self.template_count(kind=kind)

    def models(self, kind=None):
        return {
            (t.model_name, t.model_version)
            for t in self.templates
            if kind is None or t.kind == kind
        }

    def wearer(self):
        ids = {t.person_id for t in self.templates if t.is_wearer}
        return self.people.get(next(iter(ids))) if len(ids) == 1 else None

    def __len__(self):
        return len(self.people)
