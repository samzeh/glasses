"""The Elastic layer: config, client, documents, indexer.

No network. Everything runs through the fake transport, which is also the
thing being asserted on — the failure mode of a layer like this is a
malformed request, so the tests check what went out, not only what came
back.
"""

from __future__ import annotations

import json

import pytest

from glasses.search.client import ElasticClient, ElasticError
from glasses.search.config import SOURCE_CAPTURE, SOURCE_SEED, ConfigError, ElasticConfig
from glasses.search.documents import (
    PersonFact,
    TranscriptSegment,
    bulk_body,
    segments_from_utterances,
)
from glasses.search.indexer import Indexer
from glasses.search.mappings import FACT_TYPE, SEGMENT_TYPE, SEMANTIC_FIELD
from glasses.search.seed import seed_facts, seed_segments

from .elastic_fakes import FakeTransport, bulk_ok, bulk_with_failure, not_found

ENV = {
    "ELASTIC_URL": "https://demo.es.us-east-1.aws.elastic.cloud",
    "ELASTIC_API_KEY": "a-very-secret-key-value",
}


def config(**overrides) -> ElasticConfig:
    return ElasticConfig.from_env({**ENV, **overrides})


# --- config ----------------------------------------------------------------


def test_missing_credentials_name_the_variable_to_set():
    with pytest.raises(ConfigError) as error:
        ElasticConfig.from_env({})

    message = str(error.value)
    assert "ELASTIC_URL" in message
    assert "ELASTIC_API_KEY" in message


def test_a_half_configured_environment_is_still_an_error():
    """A URL with no key fails here rather than as a 401 later."""
    with pytest.raises(ConfigError) as error:
        ElasticConfig.from_env({"ELASTIC_URL": ENV["ELASTIC_URL"]})

    assert "ELASTIC_API_KEY" in str(error.value)
    assert "ELASTIC_URL" not in str(error.value).split("not set")[0]


def test_the_kibana_url_is_derived_from_the_elasticsearch_one():
    """Agent Builder is served by Kibana, and the two endpoints differ by
    one DNS label on Elastic Cloud. Getting this wrong is the most common
    misconfiguration, so it is derived rather than asked for."""
    assert config().kibana_url == "https://demo.kb.us-east-1.aws.elastic.cloud"


def test_an_explicit_kibana_url_wins():
    resolved = config(ELASTIC_KIBANA_URL="https://kibana.internal:5601/")
    assert resolved.kibana_url == "https://kibana.internal:5601"


def test_a_url_that_is_not_elastic_cloud_is_left_alone():
    """Guessing a self-managed Kibana's port would be worse than being
    visibly wrong in search-check."""
    resolved = ElasticConfig.from_env({**ENV, "ELASTIC_URL": "https://localhost:9200"})
    assert resolved.kibana_url == "https://localhost:9200"


def test_index_names_follow_the_prefix():
    resolved = config(ELASTIC_INDEX_PREFIX="demo")
    assert resolved.segments_index == "demo-transcript-segments"
    assert resolved.facts_index == "demo-person-facts"


def test_describe_never_prints_the_whole_key():
    described = config().describe()
    assert ENV["ELASTIC_API_KEY"] not in described
    assert "a-ve" in described


def test_a_trailing_slash_does_not_double_up_in_a_url():
    resolved = ElasticConfig.from_env({**ENV, "ELASTIC_URL": ENV["ELASTIC_URL"] + "/"})
    transport = FakeTransport()
    ElasticClient.from_config(resolved, transport=transport).info()

    assert transport.calls[0].url == ENV["ELASTIC_URL"] + "/"


# --- client ----------------------------------------------------------------


def test_the_api_key_is_sent_as_an_apikey_header():
    transport = FakeTransport()
    ElasticClient.from_config(config(), transport=transport).info()

    assert transport.calls[0].headers["Authorization"] == f"ApiKey {ENV['ELASTIC_API_KEY']}"


def test_an_error_response_carries_elastics_own_reason():
    """``index_not_found_exception`` is the diagnosis. A bare status code
    is not."""
    transport = FakeTransport(handler=lambda call: not_found())
    client = ElasticClient.from_config(config(), transport=transport)

    with pytest.raises(ElasticError) as error:
        client.search("missing", {"query": {"match_all": {}}})

    assert error.value.status == 404
    assert "index_not_found_exception" in str(error.value)


def test_index_exists_turns_a_404_into_false_but_not_a_403():
    """A missing index is an answer; a permissions failure is not, and
    swallowing it would make setup appear to work."""
    client = ElasticClient.from_config(
        config(), transport=FakeTransport(handler=lambda call: not_found())
    )
    assert client.index_exists("anything") is False

    forbidden = FakeTransport(handler=lambda call: (403, {"error": "security_exception"}))
    with pytest.raises(ElasticError):
        ElasticClient.from_config(config(), transport=forbidden).index_exists("anything")


def test_bulk_is_sent_as_ndjson_with_refresh():
    transport = FakeTransport(routes={"POST /_bulk": bulk_ok(0)})
    ElasticClient.from_config(config(), transport=transport).bulk('{"index":{}}\n{}\n')

    call = transport.calls[0]
    assert call.headers["Content-Type"] == "application/x-ndjson"
    assert "refresh=true" in call.url


# --- documents -------------------------------------------------------------


def test_a_segments_id_is_stable_across_processes():
    """Re-transcribing with a better model must update the sentence in
    place, not leave two versions of it in the index."""
    first = TranscriptSegment(encounter_id="e1", start_ms=1000, end_ms=4000, text="hello")
    again = TranscriptSegment(encounter_id="e1", start_ms=1000, end_ms=4000, text="hello there")

    assert first.doc_id == again.doc_id


def test_two_segments_in_one_encounter_do_not_collide():
    first = TranscriptSegment(encounter_id="e1", start_ms=0, end_ms=3000, text="a")
    second = TranscriptSegment(encounter_id="e1", start_ms=4000, end_ms=7000, text="b")

    assert first.doc_id != second.doc_id


def test_a_facts_id_is_a_digest_of_what_identifies_it():
    """A fact has no natural key, so re-extraction has to produce the same
    id or the index fills with duplicates of one claim."""
    fact = PersonFact(encounter_id="e1", kind="employer", value="Helix Bio", person_id="p1")
    same = PersonFact(
        encounter_id="e1",
        kind="employer",
        value="Helix Bio",
        person_id="p1",
        confidence=0.4,
        evidence_span="different quote",
    )
    other = PersonFact(encounter_id="e1", kind="employer", value="Verida", person_id="p1")

    assert fact.doc_id == same.doc_id
    assert fact.doc_id != other.doc_id


def test_the_same_fact_on_two_people_is_two_documents():
    at_helix = PersonFact(encounter_id="e1", kind="industry", value="biotech", person_id="p1")
    also = PersonFact(encounter_id="e1", kind="industry", value="biotech", person_id="p2")

    assert at_helix.doc_id != also.doc_id


def test_nulls_are_omitted_rather_than_indexed():
    """``dynamic: strict`` accepts a missing optional field; a null adds
    nothing a query can use."""
    doc = TranscriptSegment(encounter_id="e1", start_ms=0, end_ms=1, text="hi").to_doc()

    assert "person_id" not in doc
    assert "location" not in doc
    assert doc["doc_type"] == SEGMENT_TYPE


def test_an_unassigned_fact_still_indexes():
    """Two faces visible at once means the fact belongs to the encounter
    and waits. That state has to be searchable, not an error."""
    doc = PersonFact(encounter_id="e1", kind="interest", value="sailing").to_doc()

    assert doc["doc_type"] == FACT_TYPE
    assert "person_id" not in doc
    assert doc["encounter_id"] == "e1"


def test_duration_is_derived_rather_than_trusted():
    doc = TranscriptSegment(encounter_id="e1", start_ms=1000, end_ms=5500, text="hi").to_doc()
    assert doc["duration_ms"] == 4500


def test_the_semantic_field_appears_only_when_asked_for():
    segment = TranscriptSegment(encounter_id="e1", start_ms=0, end_ms=1, text="biotech")

    assert SEMANTIC_FIELD not in segment.to_doc()
    assert segment.to_doc(semantic=True)[SEMANTIC_FIELD] == "biotech"


def test_a_fact_row_from_the_database_maps_without_a_shim():
    """The eventual Postgres source selects the schema's own column names,
    so the index shape does not have to change when it arrives."""
    row = {
        "encounter_id": "6f1",
        "person_id": "ab2",
        "display_name": "Dan Kim",
        "kind": "commitment",
        "value": "send the preprint",
        "evidence_span": "I'll send you the preprint",
        "evidence_start_ms": 31000,
        "evidence_end_ms": 34000,
        "confidence": 0.88,
        "extracted_at": "2026-10-24T18:12:00Z",
    }

    fact = PersonFact.from_row(row)

    assert fact.person_name == "Dan Kim"
    assert fact.occurred_at == "2026-10-24T18:12:00Z"
    assert fact.evidence_start_ms == 31000


class _Utterance:
    """The shape ``transcribe_segments`` returns, without the voice stack."""

    def __init__(self, text, start_s, end_s, person_id=None, is_wearer=False, speaker=None):
        self.text = text
        self.start_s = start_s
        self.end_s = end_s
        self.person_id = person_id
        self.person_name = None
        self.is_wearer = is_wearer
        self.speaker = speaker


def test_utterances_become_segments_with_names_resolved():
    utterances = [
        _Utterance("I work in biotech.", 3.25, 7.5, person_id="p1", speaker=0),
        _Utterance("Nice to meet you.", 8.0, 10.0, is_wearer=True),
    ]

    segments = segments_from_utterances(
        utterances,
        encounter_id="e1",
        occurred_at="2026-10-24T18:12:00Z",
        location="Cal Hacks",
        name_of=lambda person_id: "Dan Kim",
    )

    assert [s.start_ms for s in segments] == [3250, 8000]
    assert segments[0].person_name == "Dan Kim"
    assert segments[0].location == "Cal Hacks"
    # The wearer's own speech is kept and flagged, not dropped: "what did I
    # promise him?" is a question too.
    assert segments[1].is_wearer is True
    assert segments[1].person_name is None


def test_bulk_body_pairs_an_action_line_with_each_document():
    segments = [
        TranscriptSegment(encounter_id="e1", start_ms=0, end_ms=1, text="a"),
        TranscriptSegment(encounter_id="e1", start_ms=2, end_ms=3, text="b"),
    ]

    lines = [json.loads(line) for line in bulk_body("idx", segments).splitlines()]

    assert len(lines) == 4
    assert lines[0] == {"index": {"_index": "idx", "_id": segments[0].doc_id}}
    assert lines[1]["text"] == "a"
    assert bulk_body("idx", segments).endswith("\n")


def test_bulk_uses_index_rather_than_create():
    """The derived copy is rebuildable, so a re-run must overwrite. With
    ``create`` a retry after a partial sync would fail on every document
    that had already landed."""
    body = bulk_body("idx", [TranscriptSegment(encounter_id="e", start_ms=0, end_ms=1, text="a")])
    assert "create" not in body


# --- indexer ---------------------------------------------------------------


def test_setup_creates_both_indices_when_they_are_missing():
    transport = FakeTransport(
        handler=lambda call: not_found() if call.method == "GET" else {"acknowledged": True}
    )
    indexer = Indexer(ElasticClient.from_config(config(), transport=transport), config())

    outcome = indexer.ensure_indices()

    assert set(outcome.values()) == {"created"}
    created = transport.all_matching("PUT /glasses-transcript-segments")
    assert created[0].body["mappings"]["dynamic"] == "strict"


def test_setup_on_an_existing_index_updates_the_mapping_instead():
    transport = FakeTransport()
    indexer = Indexer(ElasticClient.from_config(config(), transport=transport), config())

    outcome = indexer.ensure_indices()

    assert set(outcome.values()) == {"mapping updated"}
    assert transport.first("PUT /glasses-person-facts/_mapping").body["properties"]["kind"]


def test_recreate_tolerates_an_index_that_was_never_there():
    def handler(call):
        return not_found() if call.method == "DELETE" else {"acknowledged": True}

    transport = FakeTransport(handler=handler)
    indexer = Indexer(ElasticClient.from_config(config(), transport=transport), config())

    assert set(indexer.recreate_indices().values()) == {"mapping updated"}


def test_indexing_reports_what_landed():
    transport = FakeTransport(routes={"POST /_bulk": bulk_ok(2)})
    indexer = Indexer(ElasticClient.from_config(config(), transport=transport), config())

    report = indexer.index_facts(seed_facts()[:2])

    assert report.indexed == 2
    assert report.ok
    assert report.index == "glasses-person-facts"


def test_one_bad_document_does_not_lose_the_batch():
    transport = FakeTransport(routes={"POST /_bulk": bulk_with_failure(3, failing=1)})
    indexer = Indexer(ElasticClient.from_config(config(), transport=transport), config())

    report = indexer.index_segments(seed_segments()[:3])

    assert report.indexed == 2
    assert len(report.failures) == 1
    assert "mapper error" in report.failures[0][1]
    assert not report.ok


def test_indexing_batches_rather_than_sending_one_huge_request():
    transport = FakeTransport(routes={"POST /_bulk": bulk_ok(5)})
    indexer = Indexer(ElasticClient.from_config(config(), transport=transport), config())

    report = indexer.index_segments(seed_segments(), batch_size=5)

    assert report.batches == len(transport.all_matching("POST /_bulk"))
    assert report.batches > 1


def test_nothing_to_index_sends_no_requests():
    transport = FakeTransport()
    indexer = Indexer(ElasticClient.from_config(config(), transport=transport), config())

    report = indexer.index_segments([])

    assert report.batches == 0
    assert transport.calls == []


def test_clearing_seed_data_cannot_touch_a_capture():
    transport = FakeTransport(routes={"POST /*": {"deleted": 7}})
    indexer = Indexer(ElasticClient.from_config(config(), transport=transport), config())

    removed = indexer.clear(SOURCE_SEED)

    assert sum(removed.values()) == 14
    deleted = transport.first("POST /glasses-transcript-segments/_delete_by_query")
    assert deleted.body == {"query": {"term": {"source": SOURCE_SEED}}}


def test_state_counts_capture_and_seed_separately():
    """So nobody demos synthetic data believing it is real."""

    def handler(call):
        if call.method == "GET":
            return {"ok": True}
        query = (call.body or {}).get("query", {}).get("term", {})
        if query.get("source") == SOURCE_SEED:
            return {"count": 9}
        if query.get("source") == SOURCE_CAPTURE:
            return {"count": 2}
        return {"count": 11}

    transport = FakeTransport(handler=handler)
    indexer = Indexer(ElasticClient.from_config(config(), transport=transport), config())

    state = indexer.state()

    assert [s.total for s in state] == [11, 11]
    assert [s.seed for s in state] == [9, 9]
    assert "2 capture" in str(state[0])


def test_a_missing_index_is_reported_with_the_command_that_fixes_it():
    transport = FakeTransport(handler=lambda call: not_found())
    indexer = Indexer(ElasticClient.from_config(config(), transport=transport), config())

    assert "search-setup" in str(indexer.state()[0])


# --- seed data -------------------------------------------------------------


def test_every_seed_document_says_it_is_synthetic():
    for document in [*seed_segments(), *seed_facts()]:
        assert document.source == SOURCE_SEED
        assert document.encounter_id.startswith("seed-encounter-")


def test_seed_people_are_identifiable_as_fake_by_id():
    for segment in seed_segments():
        assert segment.person_id is not None
        assert segment.person_id.startswith("seed-person-")


def test_the_seed_answers_the_three_demo_questions():
    facts = seed_facts()
    segments = seed_segments()

    biotech = {f.person_name for f in facts if f.kind == "industry" and f.value == "biotech"}
    assert len(biotech) >= 2, "'who works in biotech' needs more than one answer to be a search"

    dan_commitments = [
        f for f in facts if f.person_name == "Dan Kim" and f.kind == "commitment"
    ]
    assert any("send" in f.value for f in dan_commitments)
    assert all(f.evidence_span for f in dan_commitments), "a fact with no quote is undebuggable"

    cal_hacks = {s.person_name for s in segments if s.location == "Cal Hacks"}
    assert {"Dan Kim", "Priya Raman", "Marcus Webb"} <= cal_hacks
    assert "Elena Garcia" not in cal_hacks, "the location filter needs something to exclude"


def test_seed_segments_do_not_overlap_within_an_encounter():
    """Spans are what the real pipeline produces, so the fake ones have to
    be monotonic and disjoint or the demo teaches a wrong shape."""
    by_encounter: dict[str, list[TranscriptSegment]] = {}
    for segment in seed_segments():
        by_encounter.setdefault(segment.encounter_id, []).append(segment)

    for segments in by_encounter.values():
        ordered = sorted(segments, key=lambda s: s.start_ms)
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            assert earlier.end_ms <= later.start_ms


def test_the_wearer_is_flagged_in_the_seed_conversations():
    """Their words are indexed; the flag is what keeps them from becoming
    facts about the person opposite."""
    segments = seed_segments()

    assert any(s.is_wearer for s in segments)
    assert all(f.person_name != "Sam" for f in seed_facts())
