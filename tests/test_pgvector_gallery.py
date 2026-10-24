"""``PgVectorGallery`` and ``PgStore``, against a real Postgres.

Skipped entirely unless ``DATABASE_URL`` is set. That is not politeness about
CI: the default suite is numpy-only with no network and no containers, and a
test that silently started needing Docker would make every other test in this
repo harder to trust.

With a URL set, these run against the real thing — TimescaleDB, pgvector, the
actual ``schema.sql``. Each module run gets its own schema
(``glasses_test_<pid>``) and drops it afterwards, so the tests cannot eat the
database you were demoing from.

    docker run -d --name glasses-tiger -p 55432:5432 \\
        -e POSTGRES_PASSWORD=glasses -e POSTGRES_DB=glasses \\
        timescale/timescaledb-ha:pg17
    DATABASE_URL=postgresql://postgres:glasses@127.0.0.1:55432/glasses \\
        .venv/bin/python -m pytest tests/test_pgvector_gallery.py
"""

from __future__ import annotations

import itertools
import os
import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from glasses.db.connect import ENV_VAR, apply_schema, database_url
from glasses.encounter.clock import FixedClock, seconds_to_us
from glasses.encounter.types import (
    Encounter,
    EncounterStatus,
    FaceObservation,
    Interval,
    SpeechObservation,
)
from glasses.identity.config import MatchConfig
from glasses.identity.gallery import DEFAULT_EF_SEARCH, PgVectorGallery
from glasses.identity.match import match_embedding
from glasses.identity.types import Embedding, MatchStatus

pytestmark = pytest.mark.skipif(
    not database_url(),
    reason=f"needs a Postgres with TimescaleDB and pgvector; set {ENV_VAR}",
)

psycopg = pytest.importorskip("psycopg", reason="needs the db extra")

FACE_DIM = 512
MODEL = ("arcface/buffalo_l", "1")


def direction(index: int, dim: int = FACE_DIM) -> np.ndarray:
    """A unit vector along one axis. Orthogonal to every other one, so
    similarities are exactly 1.0 or exactly 0.0 and a test can assert on them
    rather than on a tolerance."""
    vector = np.zeros(dim, dtype=np.float32)
    vector[index] = 1.0
    return vector


def embedding(vector: np.ndarray, model: tuple[str, str] = MODEL) -> Embedding:
    arr = np.asarray(vector, dtype=np.float32)
    return Embedding(vector=arr / np.linalg.norm(arr), model_name=model[0], model_version=model[1])


@pytest.fixture(scope="module")
def connection():
    """An isolated schema with ``schema.sql`` applied, dropped at the end.

    Autocommit, because a continuous aggregate cannot be created inside a
    transaction block — which is also why ``apply_schema`` runs the file one
    statement at a time.
    """
    schema = f"glasses_test_{os.getpid()}"
    with psycopg.connect(database_url(), autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        conn.execute(f'CREATE SCHEMA "{schema}"')
        # public stays on the path for the extensions, which are per-database
        # and installed there by `glasses db init`.
        conn.execute(f'SET search_path TO "{schema}", public')
        try:
            apply_schema(conn)
            yield conn
        finally:
            conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


@pytest.fixture
def store(connection):
    """A store on a wiped schema. Function-scoped: these tests count rows, and
    a gallery is only predictable if it holds what this test put in it."""
    from glasses.db.store import PgStore

    connection.execute("TRUNCATE face_observation, speech_observation")
    connection.execute("TRUNCATE person, encounter CASCADE")
    return PgStore(connection)


#: Where the fixed clocks start. Each test gets its own hour, and that is not
#: tidiness: refreshing a continuous aggregate advances a watermark past the
#: buckets it materialized, and real-time aggregation only adds rows *newer*
#: than that watermark. Truncating the hypertable does not remove what was
#: already materialized — so a later test writing into a minute an earlier
#: test refreshed would read the earlier test's numbers instead of its own.
_BASE_WALL = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
_HOURS = itertools.count()


@pytest.fixture
def clock() -> FixedClock:
    """A clock of this test's own, an hour after the last one's."""
    return FixedClock(wall=_BASE_WALL + timedelta(hours=next(_HOURS)))


# --- the schema itself -----------------------------------------------------


def test_the_streams_are_hypertables_with_compression_and_retention(connection):
    """The three things the storage layer is actually for. A plain table would
    pass every other test in this file."""
    rows = connection.execute(
        """
        SELECT hypertable_name, compression_enabled
        FROM timescaledb_information.hypertables
        WHERE hypertable_schema = current_schema()
        ORDER BY hypertable_name
        """
    ).fetchall()
    assert rows == [("face_observation", True), ("speech_observation", True)]

    policies = connection.execute(
        """
        SELECT DISTINCT proc_name FROM timescaledb_information.jobs
        WHERE hypertable_schema = current_schema()
        """
    ).fetchall()
    names = {row[0] for row in policies}
    assert "policy_compression" in names
    assert "policy_retention" in names
    assert "policy_refresh_continuous_aggregate" in names


def test_the_continuous_aggregates_answer_in_real_time(connection, store):
    """``materialized_only = false`` is the setting that makes the live view
    live: with the default, the current minute is invisible until the refresh
    job has been round, which is the one minute a demo is looking at."""
    rows = connection.execute(
        """
        SELECT view_name, materialized_only
        FROM timescaledb_information.continuous_aggregates
        WHERE view_schema = current_schema()
        ORDER BY view_name
        """
    ).fetchall()
    assert rows == [
        ("face_activity_per_minute", False),
        ("speech_activity_per_minute", False),
    ]


# --- the gallery -----------------------------------------------------------


def test_search_returns_templates_most_similar_first(store):
    sam = store.enroll("Sam", [embedding(direction(0)), embedding(direction(0) * 0.9 + 0.1)])
    dan = store.enroll("Dan", [embedding(direction(1))])

    hits = store.gallery().search(embedding(direction(0)), limit=10)

    assert [hit.person_id for hit in hits][:2] == [sam, sam]
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)
    assert hits[-1].person_id == dan
    assert hits[-1].similarity == pytest.approx(0.0, abs=1e-5)


def test_a_different_model_version_is_invisible_to_the_search(store):
    """The same guarantee ``InMemoryGallery`` gives, now enforced in SQL. A
    search spanning two model versions returns meaningless similarities, and
    it reads as a threshold problem rather than a schema one."""
    store.enroll("Sam", [embedding(direction(0), ("arcface/buffalo_l", "1"))])

    gallery = store.gallery()
    assert gallery.search(embedding(direction(0), ("arcface/buffalo_l", "1"))) != []
    assert gallery.search(embedding(direction(0), ("arcface/buffalo_l", "2"))) == []
    assert gallery.search(embedding(direction(0), ("adaface/ir101", "1"))) == []


def test_the_requested_limit_is_the_limit_actually_served(store):
    """The bug this file was written to catch.

    ``hnsw.ef_search`` defaults to 40, so an unmodified ``LIMIT 64`` — and 64
    is ``MatchConfig.search_limit``, the default every match runs with — comes
    back with 40 rows the moment the planner chooses the vector index. Nothing
    errors; the candidate list is quietly truncated and the top-3 aggregation
    scores people on whatever survived.

    The plan is asserted before the count because the count only means
    something on the HNSW path: a btree scan of (model_name, model_version)
    plus a sort is exact and would pass this test while the bug was still
    there. At this table size that is what the planner prefers, so the two
    settings take it away — which is a stand-in for the real gallery being
    large enough that it chooses HNSW on its own.
    """
    wanted = MatchConfig().search_limit
    assert wanted > DEFAULT_EF_SEARCH, "this test only means something above the pgvector default"

    rng = np.random.default_rng(0)
    store.enroll("Crowd", [embedding(rng.normal(size=FACE_DIM)) for _ in range(wanted + 60)])

    gallery = store.gallery()
    probe = embedding(rng.normal(size=FACE_DIM))
    store.connection.execute("ANALYZE face_template")
    store.connection.execute("SET enable_seqscan = off")
    store.connection.execute("SET enable_sort = off")
    try:
        plan = "\n".join(gallery.explain(probe, limit=wanted))
        assert "face_template_embedding_idx" in plan, f"not the HNSW path:\n{plan}"
        hits = gallery.search(probe, limit=wanted)
    finally:
        store.connection.execute("RESET enable_seqscan")
        store.connection.execute("RESET enable_sort")

    assert len(hits) == wanted


def test_a_probe_of_the_wrong_width_is_refused_the_way_numpy_refuses_it(store):
    """The two galleries are interchangeable, and that has to include how they
    fail: ``InMemoryGallery`` raises ValueError, so a driver exception leaking
    out of here would be a difference callers have to know about."""
    store.enroll("Sam", [embedding(direction(0))])

    with pytest.raises(ValueError, match="dim"):
        store.gallery().search(embedding(np.array([1.0, 0.0, 0.0])))


def test_a_zero_vector_is_refused_rather_than_silently_becoming_nan(store):
    store.enroll("Sam", [embedding(direction(0))])

    with pytest.raises(ValueError, match="zero vector"):
        store.gallery().search(
            Embedding(np.zeros(FACE_DIM, dtype=np.float32), *MODEL)  # type: ignore[arg-type]
        )


def test_searching_an_empty_gallery_is_not_an_error(store):
    assert store.gallery().search(embedding(direction(0))) == []


def test_a_retired_template_stops_matching(store):
    sam = store.enroll("Sam", [embedding(direction(0))])

    assert store.gallery().search(embedding(direction(0))) != []
    assert store.forget_templates(sam, kind="face") == 1
    assert store.gallery().search(embedding(direction(0))) == []


def test_the_wearer_is_not_a_candidate_in_the_voice_gallery(store):
    """The wearer's own voice is the loudest thing the mic hears. Left in the
    gallery, every one of their utterances matches them as the conversational
    partner and their speech becomes facts about the person opposite."""
    store.enroll("Sam", [embedding(direction(0))], kind="voice", is_wearer=True)
    dan = store.enroll("Dan", [embedding(direction(1))], kind="voice")

    hits = store.gallery("voice").search(embedding(direction(0)), limit=10)

    assert [hit.person_id for hit in hits] == [dan]

    # Reachable on purpose, for the pipeline's wearer check, which has to
    # compare against them rather than ignore them.
    unfiltered = PgVectorGallery(store.connection, "voice_template", exclude_wearer=False)
    assert len(unfiltered.search(embedding(direction(0)), limit=10)) == 2


def test_the_matcher_cannot_tell_which_gallery_it_is_searching(store):
    """The point of the Protocol. ``match_embedding`` is the production path
    and it takes the pgvector gallery unchanged."""
    sam = store.enroll("Sam", [embedding(direction(0)), embedding(direction(0))])
    store.enroll("Dan", [embedding(direction(1))])

    result = match_embedding(embedding(direction(0)), store.gallery())

    assert result.status is MatchStatus.MATCH
    assert result.person_id == sam
    assert result.similarity == pytest.approx(1.0, abs=1e-5)

    stranger = match_embedding(embedding(direction(7)), store.gallery())
    assert stranger.status is MatchStatus.UNKNOWN


# --- the store -------------------------------------------------------------


def test_enrolling_needs_no_consent_note(store):
    """Consent is recorded when it is supplied and required nowhere. The
    columns stay; the demand is gone."""
    person_id = store.enroll(None, [embedding(direction(0))])

    person = store.people()[0]
    assert person.id == person_id
    assert person.display_name is None
    assert person.consented_at is None
    assert store.template_count(person_id) == 1


def test_a_consent_note_is_stored_when_there_is_one(store):
    person_id = store.enroll("Sam", [embedding(direction(0))], consent_note="myself")

    person = store.find_by_name("sam")
    assert person is not None
    assert person.id == person_id
    assert person.consent_note == "myself"
    assert person.consented_at is not None


def test_forgetting_a_person_removes_every_template_in_one_call(store):
    sam = store.enroll("Sam", [embedding(direction(0)), embedding(direction(1))])
    store.enroll("Sam", [embedding(direction(2))], kind="voice", person_id=sam)
    dan = store.enroll("Dan", [embedding(direction(3))])

    assert store.forget(sam) == 3

    assert [p.id for p in store.people()] == [dan]
    assert store.template_count(kind="face") == 1
    assert store.template_count(kind="voice") == 0
    # Dan's one template is still there and still answers — orthogonally, at
    # similarity 0. What must be gone is every row of Sam's.
    assert [h.person_id for h in store.gallery().search(embedding(direction(0)))] == [dan]


def test_forgetting_someone_who_is_not_there_is_an_error(store):
    with pytest.raises(KeyError):
        store.forget(str(uuid.uuid4()))


def test_the_wearer_embeddings_come_back_as_vectors(store):
    """They are excluded from every search, so this is the only way to reach
    them — and the pipeline's wearer check needs the vectors themselves."""
    store.enroll("Sam", [embedding(direction(5))], kind="voice", is_wearer=True)

    vectors = store.wearer_embeddings()

    assert len(vectors) == 1
    assert vectors[0].model_name == MODEL[0]
    assert float(vectors[0].vector @ direction(5)) == pytest.approx(1.0, abs=1e-5)


# --- the streams -----------------------------------------------------------


def _encounter(clock: FixedClock, *, person_id: str | None) -> Encounter:
    """Half a minute of one person in front of you, talking."""
    start = clock.now_us()
    faces = [
        FaceObservation(
            span=Interval(start + seconds_to_us(i * 2), start + seconds_to_us(i * 2 + 2)),
            person_id=person_id,
            confidence=0.7,
            frame_count=2,
            is_primary=True,
        )
        for i in range(5)
    ]
    speech = [
        SpeechObservation(
            span=Interval(start + seconds_to_us(1), start + seconds_to_us(5)),
            person_id=person_id,
            speaker=1,
            confidence=0.8,
            is_wearer=False,
            text="hello",
        ),
        SpeechObservation(
            span=Interval(start + seconds_to_us(6), start + seconds_to_us(8)),
            person_id=None,
            speaker=0,
            confidence=0.9,
            is_wearer=True,
            text="hello back",
        ),
    ]
    return Encounter(
        id=uuid.uuid4().hex,
        device_id="laptop",
        started_us=start,
        status=EncounterStatus.CLOSED,
        ended_us=start + seconds_to_us(10),
        faces=faces,
        speech=speech,
    )


def test_an_encounter_and_its_observations_are_written_together(store, clock):
    sam = store.enroll("Sam", [embedding(direction(0))])
    encounter = _encounter(clock, person_id=sam)

    store.record_encounter(encounter, clock)

    rows = store.connection.execute(
        "SELECT count(*) FROM face_observation WHERE encounter_id = %s", (encounter.id,)
    ).fetchone()
    assert rows[0] == 5
    speech = store.connection.execute(
        "SELECT count(*) FROM speech_observation WHERE encounter_id = %s AND NOT is_wearer",
        (encounter.id,),
    ).fetchone()
    assert speech[0] == 1
    status = store.connection.execute(
        "SELECT status::text, ended_at IS NOT NULL FROM encounter WHERE id = %s",
        (encounter.id,),
    ).fetchone()
    assert status == ("confirmed", True)


def test_which_faces_were_visible_while_something_was_said(store, clock):
    """The primitive binding rests on, asked of the hypertable. Overlap, not
    containment: a face visible from before the sentence until after it
    contains nothing and still answers the question."""
    sam = store.enroll("Sam", [embedding(direction(0))])
    encounter = _encounter(clock, person_id=sam)
    store.record_encounter(encounter, clock)

    sentence = encounter.speech[0].span
    during = store.faces_during(encounter.id, sentence.start_us, sentence.end_us)

    assert [row[0] for row in during] == [sam, sam, sam]
    assert all(start < sentence.end_us and end > sentence.start_us for _, start, end, _ in during)


def test_the_live_view_counts_observations_and_distinct_faces_per_minute(store, clock):
    """The demo query, and the reason the aggregate groups by person_id:
    continuous aggregates cannot hold a count(DISTINCT), so the distinct count
    is taken one level up over rows the aggregate has already collapsed."""
    sam = store.enroll("Sam", [embedding(direction(0))])
    dan = store.enroll("Dan", [embedding(direction(1))])

    first = _encounter(clock, person_id=sam)
    store.record_encounter(first, clock)
    clock.advance(600)  # ten minutes later, a different person
    second = _encounter(clock, person_id=dan)
    store.record_encounter(second, clock)

    # The streams store canonical uuids; the session's ids are the same
    # value written without dashes.
    minutes = {uuid.UUID(row["encounter_id"]).hex: row for row in store.minutes()}

    assert len(minutes) == 2
    for encounter in (first, second):
        row = minutes[encounter.id]
        assert row["face_observations"] == 5
        assert row["distinct_faces"] == 1
        assert row["face_visible_s"] == pytest.approx(10.0)
        assert row["speech_segments"] == 2
        assert row["partner_speech_s"] == pytest.approx(4.0)
        assert row["wearer_speech_s"] == pytest.approx(2.0)


def test_the_live_view_still_answers_after_the_aggregate_is_materialized(store, clock):
    """Real-time aggregation and the materialized path have to agree, or the
    numbers change a minute after the demo."""
    sam = store.enroll("Sam", [embedding(direction(0))])
    encounter = _encounter(clock, person_id=sam)
    store.record_encounter(encounter, clock)

    before = store.minutes(encounter.id)
    store.refresh_aggregates()
    after = store.minutes(encounter.id)

    assert before == after
    assert before[0]["distinct_faces"] == 1


def test_a_stranger_is_counted_as_seen_without_being_counted_as_someone(store, clock):
    """An unidentified face is still an observation, and the thing that makes
    a binding ambiguous. Counting it as a person would be the bug."""
    encounter = _encounter(clock, person_id=None)

    store.record_encounter(encounter, clock)
    row = store.minutes(encounter.id)[0]

    assert row["face_observations"] == 5
    assert row["distinct_faces"] == 0
    assert row["stranger_visible_s"] == pytest.approx(10.0)


def test_chunks_compress_and_stay_queryable(store, clock):
    """Compression is configured by the schema; this proves the configuration
    is usable rather than merely present. Nothing in a demo is seven days old,
    so the policy is bypassed and the chunk is compressed by hand — which is
    exactly what the README tells you to do to see it work."""
    sam = store.enroll("Sam", [embedding(direction(0))])
    encounter = _encounter(clock, person_id=sam)
    store.record_encounter(encounter, clock)

    chunks = store.connection.execute(
        "SELECT compress_chunk(c) FROM show_chunks('face_observation') c"
    ).fetchall()
    assert chunks, "no chunk to compress — did anything get written?"

    compressed = store.connection.execute(
        """
        SELECT count(*) FROM timescaledb_information.chunks
        WHERE hypertable_schema = current_schema()
          AND hypertable_name = 'face_observation' AND is_compressed
        """
    ).fetchone()
    assert compressed[0] == 1

    during = store.faces_during(
        encounter.id, encounter.speech[0].span.start_us, encounter.speech[0].span.end_us
    )
    assert len(during) == 3
    assert store.minutes(encounter.id)[0]["face_observations"] == 5
