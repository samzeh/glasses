"""Elastic: the search layer, and the Agent Builder agent on top of it.

The question this answers is the one the glasses exist for and the rest of
the pipeline cannot answer: *who have I met who works in biotech?* Face and
voice recognition tell you who is in front of you. Recall means asking
about everyone who is not.

The shape, which is the important part:

    Postgres/TigerData          Elastic                 Agent Builder
    (system of record)   →   (derived index)   →   (tools + one agent)
      people, encounters        transcript_segment      search_conversations
      facts, templates          person_fact             get_person_facts
                                                        glasses ask "…"

One-way sync, always. Elastic is a searchable copy that can be dropped and
rebuilt; nothing reads an answer out of it and stores it as truth. That is
what makes it safe to put an LLM agent in front of: an agent querying a
derived copy can be wrong about what it found, while an agent querying the
system of record can be wrong about what is true.

The source today is the transcription pipeline plus ``LocalStore``, because
that is what exists. ``TranscriptSegment.from_row`` and
``PersonFact.from_row`` take the schema's own column names, so adding a
Postgres source is a new reader rather than a new index.
"""

from __future__ import annotations

from .agentbuilder import AgentBuilderClient, agent_definition, tool_definitions
from .client import ElasticClient, ElasticError, Transport, UrllibTransport
from .config import SOURCE_CAPTURE, SOURCE_SEED, ConfigError, ElasticConfig
from .documents import PersonFact, TranscriptSegment, segments_from_utterances
from .indexer import Indexer, IndexReport, IndexState
from .mappings import all_mappings, fact_mappings, segment_mappings

__all__ = [
    "SOURCE_CAPTURE",
    "SOURCE_SEED",
    "AgentBuilderClient",
    "ConfigError",
    "ElasticClient",
    "ElasticConfig",
    "ElasticError",
    "IndexReport",
    "IndexState",
    "Indexer",
    "PersonFact",
    "TranscriptSegment",
    "Transport",
    "UrllibTransport",
    "agent_definition",
    "all_mappings",
    "fact_mappings",
    "segment_mappings",
    "segments_from_utterances",
    "tool_definitions",
]
