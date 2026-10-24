"""``glasses ask`` and the search subcommands.

The commands that need models are exercised by hand; these cover argument
parsing, the paths that touch Elastic through the fake transport, and the
two things that have to work on a laptop with no credentials at all — the
``--dry-run`` definition dumps, which are how the Agent Builder config
gets reviewed.
"""

from __future__ import annotations

import json

import pytest

from glasses.cli import build_parser
from glasses.search import agentbuilder
from glasses.search.agentbuilder import CONVERSE_PATH, AgentBuilderClient
from glasses.search_cli import (
    _paragraphs,
    cmd_agent_setup,
    cmd_ask,
    cmd_search_seed,
    cmd_search_setup,
)

from .elastic_fakes import FakeTransport, bulk_ok
from .test_agent_builder import CONVERSE_RESPONSE

ENV = {
    "ELASTIC_URL": "https://demo.es.us-east-1.aws.elastic.cloud",
    "ELASTIC_API_KEY": "key",
}


def parse(*argv: str):
    return build_parser().parse_args(argv)


@pytest.fixture
def configured(monkeypatch):
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("ELASTIC_KIBANA_URL", raising=False)
    monkeypatch.delenv("ELASTIC_INDEX_PREFIX", raising=False)
    monkeypatch.delenv("ELASTIC_AGENT_ID", raising=False)


@pytest.fixture
def unconfigured(monkeypatch):
    for name in ("ELASTIC_URL", "ELASTIC_API_KEY", "ELASTIC_KIBANA_URL"):
        monkeypatch.delenv(name, raising=False)


def wired(monkeypatch, transport: FakeTransport) -> FakeTransport:
    """Point every client the CLI builds at one fake transport."""
    import glasses.search_cli as cli

    monkeypatch.setattr(
        cli.ElasticClient,
        "from_config",
        classmethod(lambda cls, config, **kw: cls(config.url, config.api_key, transport=transport)),
    )
    monkeypatch.setattr(
        AgentBuilderClient,
        "from_config",
        classmethod(
            lambda cls, config, **kw: cls(
                config.kibana_url,
                config.api_key,
                transport=transport,
                extra_headers={"kbn-xsrf": "true"},
            )
        ),
    )
    return transport


# --- parsing ---------------------------------------------------------------


def test_every_search_subcommand_parses():
    for argv in (
        ("ask", "who have I met who works in biotech"),
        ("ask", "what", "did", "Dan", "say"),
        ("ask", "--json", "--conversation", "abc", "who did I meet"),
        ("search-check",),
        ("search-setup", "--recreate", "--semantic"),
        ("search-setup", "--dry-run"),
        ("agent-setup", "--dry-run"),
        ("agent-setup", "--agent", "other"),
        ("search-seed",),
        ("search-seed", "--clear"),
        ("search-index", "a.wav", "--location", "Cal Hacks"),
    ):
        assert parse(*argv).command == argv[0]


def test_ask_needs_a_question():
    with pytest.raises(SystemExit):
        parse("ask")


def test_a_question_does_not_need_quoting():
    """``glasses ask who did I meet at Cal Hacks`` is what people type."""
    assert parse("ask", "who", "did", "I", "meet").question == ["who", "did", "I", "meet"]


# --- no credentials --------------------------------------------------------


def test_a_missing_environment_names_the_variables(unconfigured, capsys):
    with pytest.raises(SystemExit) as exit_code:
        cmd_ask(parse("ask", "anything"))

    assert exit_code.value.code == 2
    error = capsys.readouterr().err
    assert "ELASTIC_URL" in error
    assert "README" in error


def test_the_tool_definitions_can_be_dumped_without_credentials(unconfigured, capsys):
    """This is how the Agent Builder configuration is reviewed in a pull
    request: it is code, not something clicked into a UI."""
    assert cmd_agent_setup(parse("agent-setup", "--dry-run")) == 0

    dumped = json.loads(capsys.readouterr().out)
    assert len(dumped["tools"]) == 6
    assert dumped["agent"]["configuration"]["instructions"]


def test_the_mappings_can_be_dumped_without_credentials(unconfigured, capsys):
    assert cmd_search_setup(parse("search-setup", "--dry-run")) == 0

    dumped = json.loads(capsys.readouterr().out)
    assert dumped["transcript_segment"]["dynamic"] == "strict"
    assert "text_semantic" not in dumped["person_fact"]["properties"]


def test_the_semantic_dry_run_adds_the_field(unconfigured, capsys):
    cmd_search_setup(parse("search-setup", "--dry-run", "--semantic"))

    dumped = json.loads(capsys.readouterr().out)
    assert "text_semantic" in dumped["transcript_segment"]["properties"]


# --- ask -------------------------------------------------------------------


def test_ask_prints_the_answer_and_the_tools_behind_it(configured, monkeypatch, capsys):
    transport = wired(
        monkeypatch, FakeTransport(routes={f"POST {CONVERSE_PATH}": CONVERSE_RESPONSE})
    )

    assert cmd_ask(parse("ask", "who", "works", "in", "biotech")) == 0

    out = capsys.readouterr().out
    assert "who works in biotech" in out
    assert "Dan Kim works at Helix Bio" in out
    assert "glasses_find_people_by_fact" in out
    assert "conversation e5b4e3ff-f5c9-4f52-9001-87b403df1a6a" in out
    assert transport.first(f"POST {CONVERSE_PATH}").body["agent_id"] == "glasses_recall"


def test_ask_passes_an_overridden_agent_id(configured, monkeypatch, capsys):
    transport = wired(
        monkeypatch, FakeTransport(routes={f"POST {CONVERSE_PATH}": CONVERSE_RESPONSE})
    )
    cmd_ask(parse("ask", "--agent", "other_agent", "anything"))

    assert transport.first(f"POST {CONVERSE_PATH}").body["agent_id"] == "other_agent"


def test_ask_json_prints_the_raw_round(configured, monkeypatch, capsys):
    """Including the steps, which is what you read when the answer is
    wrong."""
    wired(monkeypatch, FakeTransport(routes={f"POST {CONVERSE_PATH}": CONVERSE_RESPONSE}))
    cmd_ask(parse("ask", "--json", "anything"))

    assert json.loads(capsys.readouterr().out)["steps"][1]["tool_id"]


def test_asking_a_missing_agent_says_how_to_create_it(configured, monkeypatch, capsys):
    wired(monkeypatch, FakeTransport(handler=lambda call: (404, {"error": "not found"})))

    assert cmd_ask(parse("ask", "anything")) == 1
    assert "glasses agent-setup" in capsys.readouterr().err


def test_an_empty_answer_is_a_failure_rather_than_a_blank_screen(
    configured, monkeypatch, capsys
):
    wired(monkeypatch, FakeTransport(routes={f"POST {CONVERSE_PATH}": {"status": "completed"}}))

    assert cmd_ask(parse("ask", "anything")) == 1
    assert "no answer" in capsys.readouterr().out


# --- seeding ---------------------------------------------------------------


def test_seeding_indexes_both_document_types(configured, monkeypatch, capsys):
    transport = wired(monkeypatch, FakeTransport(routes={"POST /_bulk": bulk_ok(400)}))

    assert cmd_search_seed(parse("search-seed")) == 0

    indices = {
        line["index"]["_index"]
        for call in transport.all_matching("POST /_bulk")
        for line in call.ndjson
        if "index" in line
    }
    assert indices == {"glasses-transcript-segments", "glasses-person-facts"}
    assert "synthetic encounters" in capsys.readouterr().out


def test_clearing_the_seed_says_real_captures_are_safe(configured, monkeypatch, capsys):
    transport = wired(monkeypatch, FakeTransport(routes={"POST /*": {"deleted": 3}}))

    assert cmd_search_seed(parse("search-seed", "--clear")) == 0

    assert "untouched" in capsys.readouterr().out
    assert transport.first("POST /glasses-person-facts/_delete_by_query").body == {
        "query": {"term": {"source": "seed"}}
    }


# --- presentation ----------------------------------------------------------


def test_wrapping_keeps_the_models_own_line_breaks():
    """An answer that lists three people as three lines should stay three
    lines."""
    wrapped = _paragraphs("Dan Kim — biotech\nMarcus Webb — biotech")

    assert wrapped.splitlines() == ["  Dan Kim — biotech", "  Marcus Webb — biotech"]


def test_wrapping_reflows_an_over_long_line():
    wrapped = _paragraphs("word " * 60)

    assert len(wrapped.splitlines()) > 1
    assert all(len(line) <= 90 for line in wrapped.splitlines())


def test_the_instructions_are_a_single_string_not_a_list():
    """The API takes ``configuration.instructions`` as a string; a list
    would be accepted as JSON and then ignored."""
    assert isinstance(agentbuilder.INSTRUCTIONS, str)
