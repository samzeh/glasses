"""A Postgres-backed store: the same operations as ``LocalStore``, on Tiger.

``LocalStore`` is the offline store. It holds people and
templates in an ``.npz`` beside a JSON index, which is right for a laptop and
wrong for everything else: it has no encounters, no facts, no observation
history, and it is single-process.

This is the other end. People and templates live in the relational tables of
``schema.sql``, the gallery search runs in pgvector against the same rows, and
the observation streams go into the hypertables — one database holding the
profile and the sensor history, which is the whole argument for using this one
rather than a vector store beside a time-series store beside Postgres.

The CLI uses ``PgCliStore`` when DATABASE_URL is configured. The offline
file-backed path remains available through an explicit --store argument.

Consent is recorded when a caller supplies it and nothing is required: the
columns exist, they are nullable, and this code does not ask.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from ..encounter.clock import Clock
from ..encounter.types import Encounter, EncounterStatus, FaceObservation, SpeechObservation
from ..identity.gallery import PgVectorGallery, to_pgvector
from ..identity.types import Embedding
from .connect import connect

FACE = "face"
VOICE = "voice"

#: Which table holds which channel's templates. The two differ by more than
#: vector width — ``face_template`` carries a quality score, ``voice_template``
#: carries the wearer flag — so every write has to know which it is writing.
TEMPLATE_TABLE = {FACE: "face_template", VOICE: "voice_template"}

#: ``EncounterStatus`` has four states and ``encounter_status`` has three.
#: That is not an oversight: *closed* is a confirmed encounter that has ended,
#: and the end is already recorded in ``ended_at``. Storing it as a fourth
#: status would make "confirmed with an ended_at" and "closed" two spellings of
#: one fact, and they would eventually disagree.
_DB_STATUS = {
    EncounterStatus.PROVISIONAL: "provisional",
    EncounterStatus.CONFIRMED: "confirmed",
    EncounterStatus.CLOSED: "confirmed",
    EncounterStatus.EXPIRED: "expired",
}


@dataclass(frozen=True)
class PgPerson:
    """A row of ``person``. ``display_name`` is nullable because a person is
    useful before anyone says their name."""

    id: str
    display_name: str | None
    status: str
    consented_at: datetime | None
    consent_note: str | None

    @property
    def name(self) -> str:
        return self.display_name or self.id[:8]


class PgStore:
    """People, templates and observation streams in one database."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    @classmethod
    def open(cls, url: str | None = None) -> PgStore:
        """Connect in autocommit. Writes here are each one statement or one
        explicit transaction, and refreshing a continuous aggregate cannot run
        inside a transaction block at all."""
        return cls(connect(url, autocommit=True))

    @property
    def connection(self) -> Any:
        return self._connection

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> PgStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- people and templates ---------------------------------------------

    def add_person(
        self,
        display_name: str | None = None,
        *,
        consent_note: str | None = None,
        created_from_encounter: str | None = None,
    ) -> str:
        """Create a person. Nothing is required, not even a name."""
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO person (display_name, consent_note, consented_at,
                                    created_from_encounter)
                VALUES (%s, %s, CASE WHEN %s::text IS NULL THEN NULL ELSE now() END, %s::uuid)
                RETURNING id::text
                """,
                (display_name, consent_note, consent_note, created_from_encounter),
            )
            return str(cursor.fetchone()[0])

    def enroll(
        self,
        display_name: str | None,
        embeddings: list[Embedding],
        *,
        kind: str = FACE,
        quality_scores: list[float] | None = None,
        person_id: str | None = None,
        is_wearer: bool = False,
        source_encounter_id: str | None = None,
        consent_note: str | None = None,
        source: str = "",
    ) -> str:
        """Add templates for a person, creating them if they are new.

        Returns the person id. One transaction: a person with no templates is
        a row that will later look like a failed enrollment, and half a
        gallery is worse than none.
        """
        if not embeddings:
            raise ValueError("no embeddings to enroll")
        if kind not in TEMPLATE_TABLE:
            raise ValueError(f"unknown kind {kind!r}; expected one of {sorted(TEMPLATE_TABLE)}")

        scores = quality_scores or [0.0] * len(embeddings)
        if len(scores) != len(embeddings):
            raise ValueError(f"got {len(embeddings)} embeddings but {len(scores)} quality scores")

        with self._connection.transaction():
            if kind == VOICE and is_wearer:
                self._connection.execute("SELECT pg_advisory_xact_lock(674521903)")
                other = self._connection.execute(
                    "SELECT 1 FROM voice_template WHERE is_wearer AND deleted_at IS NULL "
                    "AND person_id IS DISTINCT FROM %s::uuid LIMIT 1",
                    (person_id,),
                ).fetchone()
                if other:
                    raise ValueError("A different wearer is already enrolled")
            if person_id is None:
                person_id = self.add_person(display_name, consent_note=consent_note)
            with self._connection.cursor() as cursor:
                for embedding, score in zip(embeddings, scores, strict=True):
                    if kind == FACE:
                        cursor.execute(
                            """
                            INSERT INTO face_template (person_id, embedding, model_name,
                                model_version, quality_score, source_encounter_id, source)
                            VALUES (%s, %s::vector, %s, %s, %s, %s, %s)
                            """,
                            (
                                person_id,
                                to_pgvector(embedding.vector),
                                embedding.model_name,
                                embedding.model_version,
                                float(score),
                                source_encounter_id,
                                source,
                            ),
                        )
                    else:
                        cursor.execute(
                            """
                            INSERT INTO voice_template (person_id, embedding, model_name,
                                model_version, is_wearer, source_encounter_id,
                                source, quality_score)
                            VALUES (%s, %s::vector, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                person_id,
                                to_pgvector(embedding.vector),
                                embedding.model_name,
                                embedding.model_version,
                                is_wearer,
                                source_encounter_id,
                                source,
                                float(score),
                            ),
                        )
        return person_id

    def gallery(self, kind: str = FACE) -> PgVectorGallery:
        """The searchable gallery for one channel.

        Scoped by kind, like ``LocalStore.gallery``: CAM++ voiceprints and
        ArcFace faces are both 512-d, so they live in separate tables and the
        search never has the chance to compare them.
        """
        if kind not in TEMPLATE_TABLE:
            raise ValueError(f"unknown kind {kind!r}; expected one of {sorted(TEMPLATE_TABLE)}")
        return PgVectorGallery(self._connection, TEMPLATE_TABLE[kind])

    def people(self) -> list[PgPerson]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id::text, display_name, status::text, consented_at, consent_note
                FROM person WHERE deleted_at IS NULL ORDER BY created_at
                """
            )
            return [PgPerson(*row) for row in cursor.fetchall()]

    def find_by_name(self, display_name: str) -> PgPerson | None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id::text, display_name, status::text, consented_at, consent_note
                FROM person
                WHERE deleted_at IS NULL AND lower(display_name) = lower(%s)
                ORDER BY created_at
                LIMIT 1
                """,
                (display_name.strip(),),
            )
            row = cursor.fetchone()
            return PgPerson(*row) if row else None

    def name_of(self, person_id: str | None) -> str:
        if person_id is None:
            return "unknown"
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT display_name FROM person WHERE id = %s", (person_id,))
            row = cursor.fetchone()
        if row is None:
            return person_id[:8]
        return row[0] or person_id[:8]

    def template_count(self, person_id: str | None = None, kind: str = FACE) -> int:
        table = TEMPLATE_TABLE[kind]
        clause = "AND person_id = %s" if person_id else ""
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"SELECT count(*) FROM {table} WHERE deleted_at IS NULL {clause}",
                (person_id,) if person_id else (),
            )
            return int(cursor.fetchone()[0])

    def wearer_embeddings(self) -> list[Embedding]:
        """The wearer's enrolled voiceprints, for the pipeline's wearer check.

        Returned as vectors rather than left in the gallery: the wearer is
        excluded from every search, so this is the only way to reach them.
        """
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT embedding::text, model_name, model_version
                FROM voice_template
                WHERE is_wearer AND deleted_at IS NULL
                """
            )
            rows = cursor.fetchall()
        return [
            Embedding(
                vector=np.array(row[0].strip("[]").split(","), dtype=np.float32),
                model_name=row[1],
                model_version=row[2],
            )
            for row in rows
        ]

    def forget(self, person_id: str) -> int:
        """Delete a person and everything keyed to them, in one operation.

        The deletion-on-request path, and a hard delete: the FK cascade takes
        the templates, and the observation streams keep their rows with
        ``person_id`` set to null, so the history of *what the sensors saw*
        survives without saying who it was. Returns the templates removed.
        """
        with self._connection.transaction(), self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT (SELECT count(*) FROM face_template WHERE person_id = %(id)s)
                     + (SELECT count(*) FROM voice_template WHERE person_id = %(id)s)
                """,
                {"id": person_id},
            )
            removed = int(cursor.fetchone()[0])
            # Archives may contain shared gallery files. Erase the originals
            # on a deletion request; other people's canonical rows stay intact.
            cursor.execute(
                "DELETE FROM local_import_file WHERE %s::uuid = ANY(person_ids)", (person_id,)
            )
            cursor.execute("DELETE FROM person WHERE id = %s", (person_id,))
            if cursor.rowcount == 0:
                raise KeyError(f"no such person: {person_id}")
        return removed

    def forget_templates(self, person_id: str, *, kind: str) -> int:
        """Retire one channel's templates for a person, keeping the rest.

        A soft delete, unlike ``forget``: this is repair after a bad
        enrollment, and the rows that produced a wrong match are exactly the
        rows worth still being able to look at. Every search filters on
        ``deleted_at IS NULL``, so they are gone from matching immediately.
        """
        table = TEMPLATE_TABLE[kind]
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {table} SET deleted_at = now() "
                "WHERE person_id = %s AND deleted_at IS NULL",
                (person_id,),
            )
            return int(cursor.rowcount)

    # --- encounters and the observation streams ----------------------------

    def open_encounter(
        self,
        device_id: str,
        started_at: datetime,
        *,
        encounter_id: str | None = None,
        location: str | None = None,
    ) -> str:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO encounter (id, device_id, started_at, location)
                VALUES (coalesce(%s::uuid, gen_random_uuid()), %s, %s, %s)
                RETURNING id::text
                """,
                (encounter_id, device_id, started_at, location),
            )
            return str(cursor.fetchone()[0])

    def record_encounter(self, encounter: Encounter, clock: Clock) -> str:
        """Persist an encounter and every observation in it.

        The bridge between ``glasses.encounter`` and Tiger: the session state
        machine runs on device-clock microseconds and knows nothing about SQL,
        and this writes what it produced. The clock supplies the single wall
        anchor each observation's ``observed_at`` is derived through — the only
        place wall time enters, and never the thing binding reads back.

        One transaction, because an encounter whose observations landed
        without it is indistinguishable from one that was never captured.
        """
        with self._connection.transaction():
            self.open_encounter(
                encounter.device_id,
                clock.to_wall(encounter.started_us),
                encounter_id=encounter.id,
                location=encounter.location,
            )
            with self._connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE encounter SET status = %s, ended_at = %s WHERE id = %s",
                    (
                        _DB_STATUS[encounter.status],
                        clock.to_wall(encounter.ended_us) if encounter.ended_us else None,
                        encounter.id,
                    ),
                )
            self.record_faces(encounter.id, encounter.faces, clock)
            self.record_speech(encounter.id, encounter.speech, clock)
        return encounter.id

    def record_faces(
        self,
        encounter_id: str,
        observations: list[FaceObservation],
        clock: Clock,
    ) -> int:
        """Append face observations to the hypertable."""
        if not observations:
            return 0
        rows = [
            (
                encounter_id,
                clock.to_wall(o.span.start_us),
                o.span.start_us,
                o.span.end_us,
                o.person_id,
                float(o.confidence),
                int(o.frame_count),
                bool(o.is_primary),
                bool(o.centred),
            )
            for o in observations
        ]
        with self._connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO face_observation (encounter_id, observed_at,
                    start_device_clock_us, end_device_clock_us, person_id,
                    confidence, frame_count, is_primary, centred)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
        return len(rows)

    def record_speech(
        self,
        encounter_id: str,
        observations: list[SpeechObservation],
        clock: Clock,
    ) -> int:
        """Append speech observations to the hypertable."""
        if not observations:
            return 0
        rows = [
            (
                encounter_id,
                clock.to_wall(o.span.start_us),
                o.span.start_us,
                o.span.end_us,
                o.person_id,
                o.speaker,
                float(o.confidence),
                bool(o.is_wearer),
                o.text,
            )
            for o in observations
        ]
        with self._connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO speech_observation (encounter_id, observed_at,
                    start_device_clock_us, end_device_clock_us, person_id,
                    speaker, confidence, is_wearer, text)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
        return len(rows)

    def faces_during(
        self,
        encounter_id: str,
        start_us: int,
        end_us: int,
    ) -> list[tuple[str | None, int, int, float]]:
        """Which faces were visible over a device-clock span.

        The primitive the voice-to-face binding rests on, asked of the
        database rather than of a list in memory. Overlap, not containment: a
        face visible from before the sentence started until after it ended
        overlaps it and contains nothing.
        """
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT person_id::text, start_device_clock_us, end_device_clock_us, confidence
                FROM face_observation
                WHERE encounter_id = %s
                  AND start_device_clock_us < %s
                  AND end_device_clock_us > %s
                ORDER BY start_device_clock_us
                """,
                (encounter_id, end_us, start_us),
            )
            return [(row[0], int(row[1]), int(row[2]), float(row[3])) for row in cursor.fetchall()]

    # --- facts -------------------------------------------------------------

    def add_facts(self, encounter_id: str, person_id: str | None, facts: list[Any]) -> int:
        """Store extracted ``PersonFact``s against one person and encounter.

        A value this person already has is skipped, so repeating yourself in
        a second conversation does not grow the profile.
        """
        added = 0
        with self._connection.transaction(), self._connection.cursor() as cursor:
            for fact in facts:
                value = fact.value.strip()
                if not value:
                    continue
                cursor.execute(
                    "SELECT 1 FROM fact WHERE person_id IS NOT DISTINCT FROM %s::uuid "
                    "AND kind = %s AND lower(value) = lower(%s) AND superseded_by IS NULL",
                    (person_id, fact.kind, value),
                )
                if cursor.fetchone():
                    continue
                cursor.execute(
                    """
                    INSERT INTO fact (person_id, encounter_id, kind, value, evidence_span,
                                      evidence_start_ms, evidence_end_ms, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        person_id,
                        encounter_id,
                        fact.kind,
                        value,
                        fact.evidence_span,
                        fact.evidence_start_ms,
                        fact.evidence_end_ms,
                        float(fact.confidence),
                    ),
                )
                added += 1
        return added

    def facts_of(self, person_id: str) -> list[dict[str, Any]]:
        """A person's current facts, oldest first."""
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT kind, value, confidence, extracted_at
                FROM fact WHERE person_id = %s AND superseded_by IS NULL
                ORDER BY extracted_at
                """,
                (person_id,),
            )
            columns = [c.name for c in cursor.description]
            return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    def first_met(self, person_id: str) -> datetime | None:
        """When this person was first seen: their oldest fact or their creation."""
        with self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT least((SELECT min(extracted_at) FROM fact WHERE person_id = %s), "
                "created_at) FROM person WHERE id = %s",
                (person_id, person_id),
            )
            row = cursor.fetchone()
        return row[0] if row else None

    # --- the live view -----------------------------------------------------

    def refresh_aggregates(self, since: datetime | None = None) -> None:
        """Materialize the continuous aggregates now.

        The refresh policy runs once a minute; this is for a demo that will
        not wait, and for a test that wants the materialized path rather than
        the real-time one. Cannot run inside a transaction block, which is why
        this store connects in autocommit.
        """
        with self._connection.cursor() as cursor:
            for view in ("face_activity_per_minute", "speech_activity_per_minute"):
                cursor.execute(
                    f"CALL refresh_continuous_aggregate('{view}', %s::timestamptz, NULL)",
                    (since,),
                )

    def minutes(self, encounter_id: str | None = None, *, limit: int = 60) -> list[dict[str, Any]]:
        """The per-minute view a live demo renders, newest first."""
        clause = "WHERE encounter_id = %s" if encounter_id else ""
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT bucket, encounter_id::text, face_observations, distinct_faces,
                       face_visible_s, stranger_visible_s, speech_segments,
                       distinct_voices, partner_speech_s, wearer_speech_s
                FROM encounter_minute {clause}
                ORDER BY bucket DESC
                LIMIT {int(limit)}
                """,
                (encounter_id,) if encounter_id else (),
            )
            columns = [c.name for c in cursor.description]
            return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


__all__ = ["FACE", "TEMPLATE_TABLE", "VOICE", "PgPerson", "PgStore"]
