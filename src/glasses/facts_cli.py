"""``glasses remember`` — a conversation becomes something recallable.

The cold loop end to end, in one command: identify the speakers, transcribe
each segment on its own, extract durable facts, and put both halves in the
derived index. ``glasses ask`` is the other end of it.

It is one command rather than four because the four only make sense
together and because the intermediate artefacts are not worth a file. A
transcript nobody queries is a wall of text, and a fact with no transcript
behind it cannot be checked.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .autoenroll import MIN_NAME_CONFIDENCE
from .store import LocalStore


def register(subparsers) -> None:  # type: ignore[no-untyped-def]
    remember = subparsers.add_parser(
        "remember",
        help="transcribe a conversation, extract facts, and index both",
        description=__doc__,
    )
    remember.add_argument("audio", nargs="*", type=Path, help="audio files")
    remember.add_argument("--mic", action="store_true", help="record now instead")
    remember.add_argument("--seconds", type=int, default=60, help="how long to record")
    remember.add_argument("--device", default=None, help="input device for --mic")
    remember.add_argument("--save", type=Path, default=None, help="keep the recording")
    remember.add_argument("--location", default=None, help="where this happened")
    remember.add_argument("--encounter", default=None, help="encounter id (default: a new one)")
    remember.add_argument("--diarize", action="store_true", help="several speakers")
    remember.add_argument("--speakers", type=int, default=None, help="how many, if known")
    remember.add_argument("--model", default=None, help="local Whisper model for the fallback")
    remember.add_argument("--language", default="en", help="spoken language")
    remember.add_argument("--local", action="store_true", help="keep transcription on this machine")
    remember.add_argument(
        "--gemini-model", default=None, help="Gemini model for extraction"
    )
    remember.add_argument(
        "--no-enroll",
        action="store_true",
        help="do not create people from introductions",
    )
    remember.add_argument(
        "--name-confidence",
        type=float,
        default=MIN_NAME_CONFIDENCE,
        help=f"how sure a heard name must be to enrol someone (default {MIN_NAME_CONFIDENCE})",
    )
    remember.add_argument(
        "--dry-run",
        action="store_true",
        help="print the facts without indexing them",
    )
    remember.set_defaults(func=cmd_remember)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def cmd_remember(args: argparse.Namespace) -> int:
    from .autoenroll import enroll_from_conversation, relabel_utterances
    from .facts import DEFAULT_MODEL, GeminiError, GeminiExtractor, has_api_key
    from .search.documents import segments_from_utterances
    from .voice import WhisperTranscriber, build_transcriber, render
    from .voice.audio import load_audio, write_wav
    from .voice_cli import (
        _label_of,
        build_voice_pipeline,
        expand_audio,
        transcribe_with_fallback,
    )

    if not has_api_key():
        print(
            "No Gemini API key, so there is nothing to extract facts with.\n"
            "Set GEMINI_API_KEY (a .env beside the repo is read at startup).\n"
            "Get one at https://aistudio.google.com/apikey",
            file=sys.stderr,
        )
        return 2
    if args.mic == bool(args.audio):
        print("Give either --mic or audio paths, not both and not neither.", file=sys.stderr)
        return 2

    store = LocalStore.load(args.store)
    wearer = store.wearer()
    wearer_name = wearer.display_name if wearer else "the wearer"

    pipeline = build_voice_pipeline(store, diarize=args.diarize, num_speakers=args.speakers)
    transcriber = build_transcriber(
        local_model=args.model or "base.en",
        language=args.language,
        local_only=args.local,
    )
    extractor = GeminiExtractor(model=args.gemini_model or DEFAULT_MODEL)
    print(f"transcribing with {transcriber.model_name}, extracting with {extractor.model_name}")

    if args.mic:
        from .voice.capture import record_with_meter

        print(f"\nListening for {args.seconds}s.\n")
        samples = record_with_meter(args.seconds, device=args.device)
        if args.save:
            write_wav(args.save, samples)
        recordings = [("microphone", samples)]
    else:
        recordings = [(p.name, load_audio(p)) for p in expand_audio(args.audio)]

    all_segments = []
    all_facts = []
    for label, samples in recordings:
        encounter_id = args.encounter or f"enc-{uuid.uuid4().hex[:12]}"
        occurred_at = _now_iso()

        result = pipeline.process(samples)
        usable = [i for i in result.segments if i.embedding is not None]
        speaker_labels = [_label_of(i, store) for i in usable]
        utterances, transcriber = transcribe_with_fallback(
            transcriber,
            [i.segment for i in usable],
            speaker_labels,
            fallback=lambda: WhisperTranscriber(args.model or "base.en"),
        )

        print(f"\n--- {label}  ({encounter_id}) ---")
        if not utterances:
            print("  (nothing intelligible)")
            continue
        print(render(utterances))

        try:
            facts = extractor.extract(
                utterances,
                encounter_id=encounter_id,
                wearer_name=wearer_name,
                occurred_at=occurred_at,
            )
        except GeminiError as error:
            # The transcript is the expensive half and it already exists.
            # Losing it because extraction failed would be the wrong trade.
            print(f"\n  ! extraction failed: {error}", file=sys.stderr)
            facts = []

        # Someone who introduced themselves can be enrolled from the
        # conversation itself, which is the difference between facts about
        # "Speaker 1" and facts about a person the next encounter will
        # recognise. Runs before printing so the facts print with the name
        # they will be indexed under.
        if not args.no_enroll:
            enrolled, facts = enroll_from_conversation(
                store,
                facts,
                usable,
                speaker_labels,
                utterances=utterances,
                min_confidence=args.name_confidence,
                source=f"{label} ({encounter_id})",
            )
            if enrolled:
                utterances = relabel_utterances(utterances, enrolled)
                print(f"\n  enrolled {len(enrolled)} new person/people from this conversation:")
                for item in enrolled:
                    print(f"    {item}")

        print(f"\n  {len(facts)} fact(s):")
        for fact in facts:
            who = fact.person_name or "unattributed"
            print(f"    {who:<16} {fact.kind:<12} {fact.value}  ({fact.confidence:.2f})")
            if fact.evidence_span:
                print(f'      "{fact.evidence_span}"')

        all_facts.extend(facts)
        all_segments.extend(
            segments_from_utterances(
                utterances,
                encounter_id=encounter_id,
                occurred_at=occurred_at,
                location=args.location,
                model=transcriber.model_name,
                name_of=store.name_of,
            )
        )

    if args.dry_run:
        print(f"\n--dry-run: {len(all_segments)} segment(s), {len(all_facts)} fact(s) not indexed")
        return 0
    if not all_segments and not all_facts:
        print("\nNothing to index.")
        return 0

    from .search.config import ConfigError, ElasticConfig
    from .search_cli import _indexer

    try:
        config = ElasticConfig.from_env()
    except ConfigError as error:
        print(f"\n  ! not indexed: {error}", file=sys.stderr)
        print("  The transcript and facts above are still correct.", file=sys.stderr)
        return 1

    indexer = _indexer(config)
    indexer.ensure_indices()
    segment_report = indexer.index_segments(all_segments)
    fact_report = indexer.index_facts(all_facts)
    print(f"\nindexed  {segment_report}")
    print(f"indexed  {fact_report}")
    print('\nNow ask: glasses ask "what did they say they would send me"')
    return 0
