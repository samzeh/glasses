"""The Agent Builder tools and agent, as code.

Agent Builder can be configured by clicking around Kibana. This file
exists so that it is not: the tools and the agent are definitions in the
repository, applied by ``glasses agent-setup``, reviewable in a diff and
re-appliable to a fresh Serverless project in one command. A demo whose
brain lives in someone's browser session is a demo that cannot be rebuilt.

API shapes verified against the Elastic docs (Kibana 9.x / Serverless):

* ``POST   {kibana}/api/agent_builder/tools``          create a tool
* ``PUT    {kibana}/api/agent_builder/tools/{id}``     update one — takes
  ``description``, ``tags`` and ``configuration`` only; ``id`` and
  ``type`` are immutable and must not be in the body
* ``POST   {kibana}/api/agent_builder/agents``         create an agent
* ``PUT    {kibana}/api/agent_builder/agents/{id}``    update one
* ``POST   {kibana}/api/agent_builder/converse``       ask a question
* headers: ``Authorization: ApiKey …`` and ``kbn-xsrf: true``

Note the base URL: **Agent Builder is a Kibana API.** Pointing it at the
Elasticsearch endpoint of a Serverless project gives a 404 that reads like
a missing agent, which is why ``ElasticConfig`` resolves the two
separately and ``search-check`` prints both.

Two tool types are used, and the split is deliberate:

* ``index_search`` for the open question. It takes a natural-language
  query and picks keyword, semantic or hybrid retrieval itself, which is
  strictly better than a hand-written ES|QL ``MATCH`` for "who works in
  biotech" — and it keeps the demo off any full-text ES|QL semantics that
  would have to be tested against a live cluster to be trusted.
* ``esql`` for the precise questions — everything about one person, one
  location, one kind of fact. These are exact filters, ``LIKE`` patterns
  and ``STATS``, so they are predictable and cheap, and they return the
  evidence columns rather than a relevance-ranked blob.

Every ES|QL filter is written as ``TO_LOWER(field) LIKE ?param``. ES|QL
``LIKE`` is case-sensitive and the agent will not reliably reproduce
someone's capitalisation, so the comparison is lowered on both sides and
the parameter is documented as a lowercase wildcard pattern. "Dan" is
``*dan*``, which also makes a first name match a full display name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .client import ElasticError, HttpClient, Transport
from .config import ElasticConfig

#: Built-in tools, included so the agent can orient itself when a question
#: does not fit one of ours — list the indices, read a mapping, run a
#: search. Verified ids from the Agent Builder API tutorial.
BUILTIN_TOOL_IDS = (
    "platform.core.search",
    "platform.core.list_indices",
    "platform.core.get_index_mapping",
    "platform.core.get_document_by_id",
)

AGENT_PATH = "/api/agent_builder/agents"
TOOL_PATH = "/api/agent_builder/tools"
CONVERSE_PATH = "/api/agent_builder/converse"

INSTRUCTIONS = """
You answer questions for the wearer of a pair of recording glasses, about
the people they have met. Everything you know comes from transcripts of
those conversations and facts extracted from them.

Two indices, both a derived copy of the system of record:

* transcript segments — one attributed utterance each: text, person_name,
  is_wearer, encounter_id, location, occurred_at, start_ms.
* person facts — kind (name, employer, role, industry, commitment,
  interest), value, person_name, confidence, and evidence_span: the quote
  the fact was read out of.

How to answer:

1. Name people. "Dan Kim" and "Marcus Webb", not "two people".
2. Quote the evidence. Every claim should come with the sentence it came
   from and where and when it was said. The wearer needs to be able to
   tell a real memory from a plausible one.
3. is_wearer: true is the wearer's own speech. Never turn it into a fact
   about the person they were talking to. Use it when they ask what *they*
   said or promised.
4. A missing person_name or person_id means the speaker was never
   identified. Say so. Do not guess who it was.
5. If the documents you found all have source: "seed", say that the answer
   comes from synthetic demo data rather than from real recordings.
6. If you find nothing, say you have nothing on it. Do not fill the gap.

Using the tools: the person, location and value parameters are lowercase
wildcard patterns, so search for "*dan*" rather than "Dan Kim". Start with
glasses_search_conversations or glasses_find_people_by_fact to find who is
relevant, then glasses_get_person_facts and
glasses_person_conversations for the detail on each one.
""".strip()


def _esql(*lines: str) -> str:
    return "\n".join(lines)


def tool_definitions(config: ElasticConfig) -> list[dict[str, Any]]:
    """The six custom tools, as request bodies for ``POST /tools``."""
    segments, facts = config.segments_index, config.facts_index
    person_param = {
        "type": "string",
        "description": (
            "Lowercase wildcard pattern for the person's display name. "
            'Use "*dan*" to match a first name against a full name.'
        ),
    }
    return [
        {
            "id": "glasses_search_conversations",
            "type": "index_search",
            "description": (
                "Search everything anyone said, in the wearer's own transcripts. "
                "Natural-language query; returns attributed utterances with who "
                "said them, when, and where. Use this first for open questions "
                "like 'who talked about biotech' or 'who did I meet at a "
                "hackathon'. Set is_wearer aside: those utterances are the "
                "wearer's own speech."
            ),
            "tags": ["glasses", "transcripts"],
            "configuration": {"pattern": segments},
        },
        {
            "id": "glasses_search_facts",
            "type": "index_search",
            "description": (
                "Search the facts extracted about people the wearer has met — "
                "employer, role, industry, interests and commitments, each with "
                "the quote it was read out of. Natural-language query."
            ),
            "tags": ["glasses", "facts"],
            "configuration": {"pattern": facts},
        },
        {
            "id": "glasses_get_person_facts",
            "type": "esql",
            "description": (
                "Everything known about one person: their employer, role, "
                "industry, interests and what they committed to, with the "
                "evidence quote and a confidence for each."
            ),
            "tags": ["glasses", "facts"],
            "configuration": {
                "query": _esql(
                    f"FROM {facts}",
                    "| WHERE TO_LOWER(person_name) LIKE ?person",
                    "| KEEP person_name, kind, value, confidence, evidence_span,"
                    " occurred_at, encounter_id, source",
                    "| SORT confidence DESC",
                    "| LIMIT 100",
                ),
                "params": {"person": person_param},
            },
        },
        {
            "id": "glasses_person_conversations",
            "type": "esql",
            "description": (
                "Every utterance by one person, in order, across every encounter "
                "with them. Use it to answer what somebody actually said — for "
                "example what they promised to send."
            ),
            "tags": ["glasses", "transcripts"],
            "configuration": {
                "query": _esql(
                    f"FROM {segments}",
                    "| WHERE TO_LOWER(person_name) LIKE ?person",
                    "| KEEP occurred_at, location, person_name, is_wearer, text,"
                    " encounter_id, start_ms, source",
                    "| SORT occurred_at ASC, start_ms ASC",
                    "| LIMIT 200",
                ),
                "params": {"person": person_param},
            },
        },
        {
            "id": "glasses_people_met_at",
            "type": "esql",
            "description": (
                "Who the wearer met at a place or event, with how much each "
                "person said. Answers 'who did I meet at Cal Hacks'. Excludes "
                "the wearer's own speech."
            ),
            "tags": ["glasses", "transcripts"],
            "configuration": {
                "query": _esql(
                    f"FROM {segments}",
                    "| WHERE TO_LOWER(location) LIKE ?location AND is_wearer == false",
                    "| STATS utterances = COUNT(*)"
                    " BY person_name, location, encounter_id, occurred_at",
                    "| SORT occurred_at ASC",
                    "| LIMIT 100",
                ),
                "params": {
                    "location": {
                        "type": "string",
                        "description": (
                            "Lowercase wildcard pattern for the place or event, "
                            'e.g. "*cal hacks*".'
                        ),
                    }
                },
            },
        },
        {
            "id": "glasses_find_people_by_fact",
            "type": "esql",
            "description": (
                "Find everyone whose facts match a value — 'who works in "
                "biotech', 'who said they would send me something'. kind "
                'narrows to one sort of fact (industry, employer, role, '
                'commitment, interest); pass "*" for any kind.'
            ),
            "tags": ["glasses", "facts"],
            "configuration": {
                "query": _esql(
                    f"FROM {facts}",
                    "| WHERE TO_LOWER(value.keyword) LIKE ?value AND kind LIKE ?kind",
                    "| KEEP person_name, kind, value, confidence, evidence_span,"
                    " occurred_at, encounter_id, source",
                    "| SORT confidence DESC",
                    "| LIMIT 100",
                ),
                "params": {
                    "value": {
                        "type": "string",
                        "description": (
                            "Lowercase wildcard pattern for the fact's value, "
                            'e.g. "*biotech*".'
                        ),
                    },
                    "kind": {
                        "type": "string",
                        "description": (
                            "Fact kind to restrict to: name, employer, role, "
                            'industry, commitment or interest. "*" for any.'
                        ),
                    },
                },
            },
        },
    ]


def agent_definition(
    config: ElasticConfig,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The agent, as a request body for ``POST /agents``.

    Tool ids are listed explicitly rather than with a wildcard: an agent
    that silently gains every tool in the project as other people build
    things is an agent whose behaviour changes without a diff.
    """
    custom = [tool["id"] for tool in (tools if tools is not None else tool_definitions(config))]
    return {
        "id": config.agent_id,
        "name": "Glasses Recall",
        "description": (
            "Answers questions about the people you have met, from the "
            "transcripts and facts your glasses recorded."
        ),
        "labels": ["glasses", "recall"],
        "avatar_color": "#BFDBFF",
        "avatar_symbol": "👓",
        "configuration": {
            "instructions": INSTRUCTIONS,
            "tools": [{"tool_ids": [*custom, *BUILTIN_TOOL_IDS]}],
        },
    }


@dataclass(frozen=True)
class Answer:
    """What the agent said, plus how it got there.

    ``tool_calls`` is kept because it is the demo's credibility: an
    answer that names the tools it ran and the parameters it ran them with
    is one the audience can check, and the first thing to look at when the
    answer is wrong.
    """

    text: str
    conversation_id: str | None = None
    status: str | None = None
    tool_calls: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(cls, payload: Any) -> Answer:
        body = payload or {}
        response = body.get("response") or {}
        # "response": {"message": …} is the documented shape. "answer" is
        # what the experimental response_mode: simple returns, and is
        # accepted here so that switching modes does not silently print
        # nothing.
        text = response.get("message") or body.get("answer") or ""
        calls = []
        for step in body.get("steps") or []:
            if step.get("type") != "tool_call":
                continue
            params = step.get("params") or {}
            rendered = ", ".join(f"{key}={value!r}" for key, value in sorted(params.items()))
            calls.append(f"{step.get('tool_id', 'tool')}({rendered})")
        return cls(
            text=str(text).strip(),
            conversation_id=body.get("conversation_id"),
            status=body.get("status"),
            tool_calls=tuple(calls),
            raw=body,
        )


class AgentBuilderClient(HttpClient):
    """The Agent Builder half of the API, on the Kibana endpoint."""

    @classmethod
    def from_config(
        cls, config: ElasticConfig, *, transport: Transport | None = None
    ) -> AgentBuilderClient:
        return cls(
            config.kibana_url,
            config.api_key,
            transport=transport,
            timeout_s=config.timeout_s,
            # Kibana rejects state-changing requests without it.
            extra_headers={"kbn-xsrf": "true"},
        )

    # --- tools -------------------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        response = self.request("GET", TOOL_PATH)
        return (response or {}).get("results") or (response or {}).get("tools") or []

    def get_tool(self, tool_id: str) -> dict[str, Any]:
        return self.request("GET", f"{TOOL_PATH}/{tool_id}")

    def create_tool(self, definition: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", TOOL_PATH, body=definition)

    def update_tool(self, tool_id: str, definition: dict[str, Any]) -> dict[str, Any]:
        """``id`` and ``type`` are immutable, so they are stripped here."""
        body = {
            key: value
            for key, value in definition.items()
            if key in ("description", "tags", "configuration", "confirmation")
        }
        return self.request("PUT", f"{TOOL_PATH}/{tool_id}", body=body)

    def delete_tool(self, tool_id: str) -> Any:
        return self.request("DELETE", f"{TOOL_PATH}/{tool_id}")

    # --- agents ------------------------------------------------------------

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        return self.request("GET", f"{AGENT_PATH}/{agent_id}")

    def create_agent(self, definition: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", AGENT_PATH, body=definition)

    def update_agent(self, agent_id: str, definition: dict[str, Any]) -> dict[str, Any]:
        body = {key: value for key, value in definition.items() if key != "id"}
        return self.request("PUT", f"{AGENT_PATH}/{agent_id}", body=body)

    def delete_agent(self, agent_id: str) -> Any:
        return self.request("DELETE", f"{AGENT_PATH}/{agent_id}")

    # --- asking ------------------------------------------------------------

    def converse(
        self,
        question: str,
        *,
        agent_id: str,
        conversation_id: str | None = None,
    ) -> Answer:
        """One question, one answer. Synchronous.

        The streaming endpoint (``/converse/async``) would let the terminal
        show reasoning as it arrives, which is nicer on a projector, but it
        is server-sent events and that is a parser plus a failure mode for
        a demo that takes two seconds. Not worth it yet.
        """
        body: dict[str, Any] = {"input": question, "agent_id": agent_id}
        if conversation_id:
            body["conversation_id"] = conversation_id
        return Answer.from_response(self.request("POST", CONVERSE_PATH, body=body))


def apply_definitions(
    client: AgentBuilderClient,
    config: ElasticConfig,
) -> dict[str, str]:
    """Create or update every tool and the agent. Idempotent.

    Re-running must converge rather than fail, because this is how the
    configuration is edited: change the definition in this file, run
    ``glasses agent-setup`` again. A tool whose immutable ``type`` changed
    cannot be updated in place, so it is replaced — which is safe, since
    the definition here is the only copy that matters.
    """
    outcome: dict[str, str] = {}
    for definition in tool_definitions(config):
        tool_id = definition["id"]
        outcome[tool_id] = _apply_tool(client, tool_id, definition)

    agent = agent_definition(config)
    try:
        client.get_agent(config.agent_id)
    except ElasticError as error:
        if error.status != 404:
            raise
        client.create_agent(agent)
        outcome[config.agent_id] = "agent created"
    else:
        client.update_agent(config.agent_id, agent)
        outcome[config.agent_id] = "agent updated"
    return outcome


def _apply_tool(client: AgentBuilderClient, tool_id: str, definition: dict[str, Any]) -> str:
    try:
        client.get_tool(tool_id)
    except ElasticError as error:
        if error.status != 404:
            raise
        client.create_tool(definition)
        return "created"

    try:
        client.update_tool(tool_id, definition)
    except ElasticError as error:
        if error.status not in (400, 409):
            raise
        # Almost always the tool's type changed, which an update cannot do.
        client.delete_tool(tool_id)
        client.create_tool(definition)
        return "replaced"
    return "updated"
