"""The Agent Builder tool and agent definitions, and the converse call.

The definitions are the product here — they are what the agent can do —
so they are tested as data: ids unique, every ES|QL parameter actually
referenced by its query, every index name coming from config rather than
hardcoded, and the request bodies matching the shapes the Agent Builder
API documents.

None of this talks to Elastic. What it cannot test is whether Elastic
accepts the definitions; see the runbook in README.md.
"""

from __future__ import annotations

import re

import pytest

from glasses.search.agentbuilder import (
    AGENT_PATH,
    BUILTIN_TOOL_IDS,
    CONVERSE_PATH,
    TOOL_PATH,
    AgentBuilderClient,
    Answer,
    agent_definition,
    apply_definitions,
    tool_definitions,
)
from glasses.search.client import ElasticError
from glasses.search.config import ElasticConfig

from .elastic_fakes import FakeTransport, not_found

ENV = {
    "ELASTIC_URL": "https://demo.es.us-east-1.aws.elastic.cloud",
    "ELASTIC_API_KEY": "key",
}

#: The documented converse response, trimmed to the fields that matter.
CONVERSE_RESPONSE = {
    "conversation_id": "e5b4e3ff-f5c9-4f52-9001-87b403df1a6a",
    "round_id": "145aeacf",
    "status": "completed",
    "steps": [
        {"type": "reasoning", "reasoning": "The user is asking about biotech…"},
        {
            "type": "tool_call",
            "tool_id": "glasses_find_people_by_fact",
            "params": {"value": "*biotech*", "kind": "industry"},
            "results": [],
        },
    ],
    "model_usage": {"input_tokens": 30345, "output_tokens": 704},
    "response": {"message": "Test Person works at Helix Bio, a biotech company."},
}


def config(**overrides) -> ElasticConfig:
    return ElasticConfig.from_env({**ENV, **overrides})


def client(transport: FakeTransport) -> AgentBuilderClient:
    return AgentBuilderClient.from_config(config(), transport=transport)


# --- tool definitions ------------------------------------------------------


def test_the_tools_have_unique_ids():
    ids = [tool["id"] for tool in tool_definitions(config())]
    assert len(ids) == len(set(ids))


def test_every_tool_is_one_of_the_two_types_used():
    types = {tool["type"] for tool in tool_definitions(config())}
    assert types == {"esql", "index_search"}


def test_every_esql_parameter_is_referenced_by_its_query():
    """A declared parameter the query never uses is dead weight the model
    still has to fill in, and a referenced parameter that is not declared
    is a 400 at tool-creation time."""
    for tool in tool_definitions(config()):
        if tool["type"] != "esql":
            continue
        query = tool["configuration"]["query"]
        declared = set(tool["configuration"]["params"])
        used = set(re.findall(r"\?([A-Za-z_][A-Za-z0-9_]*)", query))
        assert declared == used, tool["id"]


#: Agent Builder's own parameter types, which are NOT Elasticsearch field
#: types. "keyword" is a mapping type and is refused here with a 400 naming
#: this exact list, which is how the tools failed to create the first time.
PARAM_TYPES = ("string", "integer", "float", "boolean", "date", "array")


def test_every_esql_parameter_is_described():
    """The description is the only thing telling the model that these are
    lowercase wildcard patterns."""
    for tool in tool_definitions(config()):
        for name, spec in tool.get("configuration", {}).get("params", {}).items():
            assert spec["description"], f"{tool['id']}.{name}"
            assert spec["type"] in PARAM_TYPES


def test_queries_read_the_configured_indices_and_nothing_else():
    """Index names come from config, so a project with a different prefix
    needs no edit here."""
    resolved = config(ELASTIC_INDEX_PREFIX="other")
    for tool in tool_definitions(resolved):
        if tool["type"] == "esql":
            source = tool["configuration"]["query"].splitlines()[0]
            assert source in (f"FROM {resolved.segments_index}", f"FROM {resolved.facts_index}")
        else:
            assert tool["configuration"]["pattern"] in resolved.indices


def test_person_filters_are_case_insensitive():
    """ES|QL LIKE is case-sensitive and the model will not reliably
    reproduce someone's capitalisation."""
    for tool in tool_definitions(config()):
        query = tool.get("configuration", {}).get("query", "")
        for line in query.splitlines():
            if "LIKE ?" in line:
                assert "TO_LOWER(" in line, tool["id"]


def test_the_people_met_at_tool_excludes_the_wearers_own_speech():
    """Otherwise the wearer turns up in the list of people they met."""
    tool = _tool("glasses_people_met_at")
    assert "is_wearer == false" in tool["configuration"]["query"]


def test_the_conversation_tools_return_the_evidence_columns():
    """An answer without the quote is not checkable, which is the whole
    point of indexing the transcript rather than a summary of it."""
    assert "text" in _tool("glasses_person_conversations")["configuration"]["query"]
    assert "evidence_span" in _tool("glasses_get_person_facts")["configuration"]["query"]


def test_every_query_is_bounded():
    for tool in tool_definitions(config()):
        query = tool.get("configuration", {}).get("query")
        if query:
            assert "| LIMIT " in query, tool["id"]


def test_every_tool_describes_itself():
    """The description is the only thing the model uses to choose."""
    for tool in tool_definitions(config()):
        assert len(tool["description"]) > 40, tool["id"]


def _tool(tool_id: str) -> dict:
    return next(t for t in tool_definitions(config()) if t["id"] == tool_id)


# --- agent definition ------------------------------------------------------


def test_the_agent_selects_every_custom_tool_and_the_builtins():
    definition = agent_definition(config())
    selected = definition["configuration"]["tools"][0]["tool_ids"]

    assert {tool["id"] for tool in tool_definitions(config())} <= set(selected)
    assert set(BUILTIN_TOOL_IDS) <= set(selected)


def test_the_agent_names_its_tools_rather_than_taking_a_wildcard():
    """A wildcard agent silently gains whatever anyone else builds in the
    project, which changes its behaviour with no diff."""
    selected = agent_definition(config())["configuration"]["tools"][0]["tool_ids"]
    assert "*" not in selected


def test_the_agent_id_follows_config():
    assert agent_definition(config(ELASTIC_AGENT_ID="other"))["id"] == "other"


def test_the_instructions_cover_the_three_mistakes_that_matter():
    instructions = agent_definition(config())["configuration"]["instructions"]

    # The wearer's own speech must not become a fact about someone else.
    assert "is_wearer" in instructions
    # Synthetic data must be declared as synthetic.
    assert "seed" in instructions
    # An unidentified speaker is a real state, not an invitation to guess.
    assert "guess" in instructions
    # And the evidence quote is what makes an answer checkable.
    assert "evidence_span" in instructions


# --- the client ------------------------------------------------------------


def test_agent_builder_talks_to_kibana_not_elasticsearch():
    """The single most common misconfiguration: the Agent Builder API is a
    Kibana API, and on Serverless that is a different hostname."""
    transport = FakeTransport()
    client(transport).get_agent("glasses_recall")

    assert transport.calls[0].url.startswith("https://demo.kb.")


def test_state_changing_requests_carry_the_kibana_xsrf_header():
    transport = FakeTransport()
    client(transport).create_tool({"id": "x"})

    assert transport.calls[0].headers["kbn-xsrf"] == "true"


def test_converse_sends_the_documented_body():
    transport = FakeTransport(routes={f"POST {CONVERSE_PATH}": CONVERSE_RESPONSE})
    client(transport).converse("who works in biotech", agent_id="glasses_recall")

    call = transport.first(f"POST {CONVERSE_PATH}")
    assert call.body == {"input": "who works in biotech", "agent_id": "glasses_recall"}


def test_a_conversation_id_is_sent_only_when_continuing_one():
    """Sending a null conversation_id is how a fresh question gets attached
    to the wrong conversation."""
    transport = FakeTransport(routes={f"POST {CONVERSE_PATH}": CONVERSE_RESPONSE})
    client(transport).converse("and Priya?", agent_id="a", conversation_id="abc")

    assert transport.calls[0].body["conversation_id"] == "abc"


def test_the_answer_is_read_out_of_the_documented_place():
    answer = Answer.from_response(CONVERSE_RESPONSE)

    assert answer.text == "Test Person works at Helix Bio, a biotech company."
    assert answer.conversation_id == "e5b4e3ff-f5c9-4f52-9001-87b403df1a6a"
    assert answer.status == "completed"


def test_the_answer_reports_which_tools_ran():
    """The demo's credibility: an answer that names the tool calls behind
    it can be checked, and it is the first place to look when it is
    wrong."""
    answer = Answer.from_response(CONVERSE_RESPONSE)

    assert len(answer.tool_calls) == 1
    assert "glasses_find_people_by_fact" in answer.tool_calls[0]
    assert "kind='industry'" in answer.tool_calls[0]


def test_reasoning_steps_are_not_reported_as_tool_calls():
    answer = Answer.from_response(CONVERSE_RESPONSE)
    assert not any("reasoning" in call for call in answer.tool_calls)


def test_an_empty_response_is_an_empty_answer_rather_than_a_crash():
    assert Answer.from_response({}).text == ""
    assert Answer.from_response(None).text == ""
    assert Answer.from_response({"status": "error"}).status == "error"


def test_the_simple_response_mode_is_also_understood():
    """``response_mode: simple`` returns ``answer`` instead of
    ``response.message``. Accepted so that switching modes does not
    silently print nothing."""
    assert Answer.from_response({"answer": "Test Person."}).text == "Test Person."


# --- applying the definitions ---------------------------------------------


def test_applying_to_an_empty_project_creates_everything():
    def handler(call):
        return not_found() if call.method == "GET" else {"ok": True}

    transport = FakeTransport(handler=handler)
    outcome = apply_definitions(client(transport), config())

    assert set(outcome.values()) == {"created", "agent created"}
    posts = transport.all_matching(f"POST {TOOL_PATH}")
    assert len(posts) == len(tool_definitions(config()))
    assert transport.first(f"POST {AGENT_PATH}").body["id"] == "glasses_recall"


def test_re_applying_updates_in_place():
    """This is how the configuration is edited: change the definition in
    the repo, run agent-setup again. It has to converge, not fail."""
    transport = FakeTransport()
    outcome = apply_definitions(client(transport), config())

    assert set(outcome.values()) == {"updated", "agent updated"}
    assert not transport.all_matching(f"POST {TOOL_PATH}")


def test_an_update_never_sends_the_immutable_fields():
    """``PUT /tools/{id}`` takes description, tags and configuration only.
    Sending id or type is a 400."""
    transport = FakeTransport()
    apply_definitions(client(transport), config())

    updates = transport.all_matching(f"PUT {TOOL_PATH}/*")
    assert len(updates) == len(tool_definitions(config()))
    for call in updates:
        assert set(call.body) <= {"description", "tags", "configuration", "confirmation"}


def test_a_tool_whose_type_changed_is_replaced():
    """An immutable field cannot be updated, so the only way to converge
    is delete and recreate — safe, because the definition in the repo is
    the only copy that matters."""

    def handler(call):
        if call.method == "PUT" and call.path.startswith(TOOL_PATH):
            return (400, {"error": "cannot change type"})
        return {"ok": True}

    transport = FakeTransport(handler=handler)
    outcome = apply_definitions(client(transport), config())

    assert all(value in ("replaced", "agent updated") for value in outcome.values())
    replaced = transport.all_matching(f"DELETE {TOOL_PATH}/*")
    assert len(replaced) == len(tool_definitions(config()))


def test_a_permissions_failure_is_not_mistaken_for_a_missing_tool():
    """A 403 that reads as 'not there yet' would make setup report success
    while having created nothing."""
    transport = FakeTransport(handler=lambda call: (403, {"error": "security_exception"}))

    with pytest.raises(ElasticError):
        apply_definitions(client(transport), config())
