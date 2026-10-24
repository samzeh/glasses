"""The search and recall subcommands: ``glasses ask`` and the plumbing.

``ask`` is the demo. Everything else here exists so that ``ask`` has
something to answer from:

    glasses search-setup     create the two indices
    glasses agent-setup      apply the Agent Builder tools and agent
    glasses search-seed      load the synthetic demo encounters
    glasses search-index     index real transcripts from audio
    glasses search-check     what is configured, and what is in it
    glasses ask "…"          ask the agent a question

The two ``setup`` commands are idempotent and take ``--dry-run``, which
prints exactly what would be sent. That is deliberate: the Agent Builder
configuration is code in this repo, not something clicked into a UI, and
``--dry-run`` lets the whole definition be reviewed — and tested — with no
credentials at all.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import textwrap
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from .search.agentbuilder import (
    AgentBuilderClient,
    agent_definition,
    apply_definitions,
    tool_definitions,
)
from .search.client import ElasticClient, ElasticError
from .search.config import SOURCE_CAPTURE, SOURCE_SEED, ConfigError, ElasticConfig
from .search.documents import segments_from_utterances
from .search.indexer import Indexer
from .search.mappings import all_mappings
from .search.seed import seed_facts, seed_segments

# --- terminal niceties ------------------------------------------------------


def _colour(enabled: bool | None = None):  # type: ignore[no-untyped-def]
    """ANSI helpers that turn themselves off when piped.

    ``glasses ask | tee`` should produce plain text; a demo on a projector
    should not.
    """
    on = sys.stdout.isatty() and not os.environ.get("NO_COLOR") if enabled is None else enabled

    def wrap(code: str):  # type: ignore[no-untyped-def]
        return (lambda text: f"\033[{code}m{text}\033[0m") if on else (lambda text: text)

    return {
        "bold": wrap("1"),
        "dim": wrap("2"),
        "cyan": wrap("36"),
        "green": wrap("32"),
        "yellow": wrap("33"),
    }


def _width(limit: int = 88) -> int:
    return min(limit, max(48, shutil.get_terminal_size((80, 24)).columns - 2))


def _paragraphs(text: str, indent: str = "  ") -> str:
    """Wrap to the terminal, preserving the model's own line breaks.

    An answer that lists three people as three lines should stay three
    lines; only over-long lines are reflowed.
    """
    width = _width()
    out: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            out.append("")
            continue
        out.extend(
            textwrap.wrap(
                line,
                width=width,
                initial_indent=indent,
                subsequent_indent=indent + ("  " if line.lstrip().startswith(("-", "*")) else ""),
            )
        )
    return "\n".join(out)


def _config(args: argparse.Namespace) -> ElasticConfig:
    """Resolve the environment, or exit with the runbook in the message."""
    try:
        config = ElasticConfig.from_env()
    except ConfigError as error:
        print(error, file=sys.stderr)
        raise SystemExit(2) from error
    if getattr(args, "agent", None):
        config = replace(config, agent_id=args.agent)
    return config


def _elastic(config: ElasticConfig) -> ElasticClient:
    return ElasticClient.from_config(config)


def _indexer(config: ElasticConfig, *, semantic: bool = False) -> Indexer:
    return Indexer(_elastic(config), config, semantic=semantic)


# --- commands --------------------------------------------------------------


def cmd_search_check(args: argparse.Namespace) -> int:
    """What is configured, whether it answers, and what is in it."""
    style = _colour()
    config = _config(args)
    print(config.describe())

    print()
    try:
        info = _elastic(config).info()
        version = (info or {}).get("version", {})
        flavour = version.get("build_flavor", "")
        print(f"reachable     {version.get('number', '?')} {flavour}".rstrip())
    except ElasticError as error:
        print(style["yellow"](f"unreachable   {error}"), file=sys.stderr)
        return 1

    for state in _indexer(config).state():
        print(f"  {state}")

    agent = AgentBuilderClient.from_config(config)
    try:
        definition = agent.get_agent(config.agent_id)
        tools = definition.get("configuration", {}).get("tools", [])
        print(f"\nagent         {config.agent_id} present, {len(tools)} tool selection(s)")
    except ElasticError as error:
        hint = "run glasses agent-setup" if error.status == 404 else str(error)
        print(f"\nagent         {config.agent_id} not usable — {hint}")
        return 1
    return 0


def cmd_search_setup(args: argparse.Namespace) -> int:
    """Create or update the two indices."""
    if args.dry_run:
        # Checked before the environment: reviewing the mappings has to
        # work on a laptop with no Elastic at all.
        print(json.dumps(all_mappings(semantic=args.semantic), indent=2))
        return 0

    indexer = _indexer(_config(args), semantic=args.semantic)
    outcome = indexer.recreate_indices() if args.recreate else indexer.ensure_indices()
    for index, what in outcome.items():
        print(f"  {index}  {what}")
    if args.semantic:
        print(
            "\nsemantic_text is on. It needs an inference endpoint in the "
            "project; if indexing fails with a missing inference id, re-run "
            "without --semantic."
        )
    print("\nNext: glasses agent-setup, then glasses search-seed")
    return 0


def cmd_agent_setup(args: argparse.Namespace) -> int:
    """Apply the Agent Builder tools and agent, idempotently."""
    config = _config(args) if not args.dry_run else _dry_run_config()
    tools = tool_definitions(config)
    agent = agent_definition(config, tools)

    if args.dry_run:
        print(json.dumps({"tools": tools, "agent": agent}, indent=2))
        return 0

    client = AgentBuilderClient.from_config(config)
    for name, what in apply_definitions(client, config).items():
        print(f"  {name}  {what}")
    print('\nNext: glasses ask "who have I met who works in biotech"')
    return 0


def _dry_run_config() -> ElasticConfig:
    """A config that needs no credentials, for printing definitions.

    ``--dry-run`` has to work on a laptop with no Elastic at all — that is
    how the definitions get reviewed in a pull request.
    """
    return ElasticConfig(url="", api_key="", kibana_url="")


def cmd_search_seed(args: argparse.Namespace) -> int:
    """Load (or remove) the synthetic demo encounters."""
    config = _config(args)
    indexer = _indexer(config, semantic=args.semantic)

    if args.clear:
        removed = indexer.clear(SOURCE_SEED)
        for index, count in removed.items():
            print(f"  {index}  {count} seed document(s) removed")
        print("\nReal captures are untouched: the delete is filtered on source=seed.")
        return 0

    segments, facts = seed_segments(), seed_facts()
    print(indexer.index_segments(segments))
    print(indexer.index_facts(facts))
    print(
        f"\n{len(segments)} utterance(s) and {len(facts)} fact(s) across "
        f"{len({s.encounter_id for s in segments})} synthetic encounters.\n"
        "Every one carries source=seed; remove them with: glasses search-seed --clear"
    )
    return 0


def cmd_search_index(args: argparse.Namespace) -> int:
    """Transcribe real recordings and index what was said.

    Facts are not extracted here, because nothing extracts them yet — the
    ``fact`` table exists and the mapping for it exists, but the extractor
    does not. Segments alone already answer "what did Dan say he'd send
    me?"; claiming to index facts we do not have would be worse than the
    gap.
    """
    from .store import LocalStore
    from .voice import WhisperTranscriber, transcribe_segments
    from .voice.audio import load_audio
    from .voice_cli import build_voice_pipeline, expand_audio

    config = _config(args)
    store = LocalStore.load(args.store)
    pipeline = build_voice_pipeline(store, diarize=args.diarize, num_speakers=args.speakers)
    transcriber = WhisperTranscriber(args.model)
    indexer = _indexer(config, semantic=args.semantic)

    paths = expand_audio(args.audio)
    if not paths:
        print("no audio files found", file=sys.stderr)
        return 1

    total = 0
    for path in paths:
        result = pipeline.process(load_audio(path))
        usable = [item for item in result.segments if item.embedding is not None]
        utterances = transcribe_segments(
            transcriber,
            [item.segment for item in usable],
            labels=[
                (
                    item.match.person_id if item.match else None,
                    store.name_of(item.match.person_id)
                    if item.match and item.match.person_id
                    else None,
                    item.is_wearer,
                )
                for item in usable
            ],
        )
        segments = segments_from_utterances(
            utterances,
            encounter_id=args.encounter or _encounter_id(path),
            occurred_at=args.at or _modified_at(path),
            location=args.location,
            source=SOURCE_CAPTURE,
            model=transcriber.model_name,
            name_of=store.name_of,
        )
        report = indexer.index_segments(segments)
        print(f"  {path.name}  {report}")
        for doc_id, reason in report.failures:
            print(f"      ! {doc_id}: {reason}", file=sys.stderr)
        total += report.indexed

    print(f"\n{total} segment(s) indexed as source={SOURCE_CAPTURE}")
    return 0


def _encounter_id(path: Path) -> str:
    """A stable id for a recording, so re-indexing it overwrites.

    Derived from the file rather than random: running the command twice on
    the same WAV must not double the transcript in the index.
    """
    stamp = _modified_at(path)[:10].replace("-", "")
    return f"capture-{stamp}-{path.stem}"


def _modified_at(path: Path) -> str:
    try:
        when = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    except OSError:
        when = datetime.now(UTC)
    return when.isoformat().replace("+00:00", "Z")


def cmd_ask(args: argparse.Namespace) -> int:
    """Ask the agent a question about the people you have met."""
    style = _colour()
    config = _config(args)
    client = AgentBuilderClient.from_config(config)
    question = " ".join(args.question).strip()
    if not question:
        print("ask what?", file=sys.stderr)
        return 1

    if not args.json:
        # Echoed before the request, so the question is on screen while
        # the agent thinks. --json stays machine-readable.
        print(f"\n{style['cyan']('?')} {style['bold'](question)}\n")
    started = time.perf_counter()
    try:
        answer = client.converse(
            question,
            agent_id=config.agent_id,
            conversation_id=args.conversation,
        )
    except ElasticError as error:
        print(style["yellow"](str(error)), file=sys.stderr)
        if error.status == 404:
            print("\nNo such agent. Run: glasses agent-setup", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started

    if args.json:
        print(json.dumps(answer.raw, indent=2))
        return 0

    print(_paragraphs(answer.text) if answer.text else style["yellow"]("  (no answer)"))

    if answer.tool_calls:
        print()
        for call in answer.tool_calls:
            print(style["dim"](f"  · {call}"))
    footer = f"  {elapsed:.1f}s"
    if answer.conversation_id:
        footer += f"  ·  conversation {answer.conversation_id}"
        footer += f"\n  continue with: glasses ask --conversation {answer.conversation_id} \"…\""
    print(style["dim"](f"\n{footer}"))
    return 0 if answer.text else 1


# --- wiring ----------------------------------------------------------------


def register(subparsers) -> None:  # type: ignore[no-untyped-def]
    ask = subparsers.add_parser("ask", help="ask the agent about people you have met")
    ask.add_argument("question", nargs="+", help='e.g. "who have I met who works in biotech"')
    ask.add_argument("--agent", help="agent id (default: $ELASTIC_AGENT_ID or glasses_recall)")
    ask.add_argument("--conversation", help="continue an existing conversation")
    ask.add_argument("--json", action="store_true", help="print the raw response")
    ask.set_defaults(func=cmd_ask)

    check = subparsers.add_parser(
        "search-check", help="what Elastic is configured, and what is in it"
    )
    check.add_argument("--agent", help="agent id to look for")
    check.set_defaults(func=cmd_search_check)

    setup = subparsers.add_parser("search-setup", help="create the Elastic indices")
    setup.add_argument("--recreate", action="store_true", help="drop and rebuild both indices")
    setup.add_argument("--semantic", action="store_true", help="add a semantic_text field")
    setup.add_argument("--dry-run", action="store_true", help="print the mappings and stop")
    setup.set_defaults(func=cmd_search_setup)

    agent_setup = subparsers.add_parser(
        "agent-setup", help="apply the Agent Builder tools and agent"
    )
    agent_setup.add_argument("--agent", help="agent id to create (default: glasses_recall)")
    agent_setup.add_argument(
        "--dry-run",
        action="store_true",
        help="print the tool and agent definitions without sending them — "
        "works with no credentials",
    )
    agent_setup.set_defaults(func=cmd_agent_setup)

    seed = subparsers.add_parser("search-seed", help="load the synthetic demo encounters")
    seed.add_argument("--clear", action="store_true", help="remove every source=seed document")
    seed.add_argument("--semantic", action="store_true", help="also write the semantic field")
    seed.set_defaults(func=cmd_search_seed)

    index = subparsers.add_parser("search-index", help="transcribe recordings and index them")
    index.add_argument("audio", nargs="+")
    index.add_argument("--encounter", help="encounter id (default: derived from the file)")
    index.add_argument("--at", help="when it happened, ISO 8601 (default: the file's mtime)")
    index.add_argument("--location", help="where it happened, e.g. 'Cal Hacks'")
    index.add_argument("--diarize", action="store_true", help="recording has several voices")
    index.add_argument("--speakers", type=int, default=-1, help="pin the speaker count")
    index.add_argument("--model", default="base.en", help="whisper size (default base.en)")
    index.add_argument("--semantic", action="store_true", help="also write the semantic field")
    index.set_defaults(func=cmd_search_index)
