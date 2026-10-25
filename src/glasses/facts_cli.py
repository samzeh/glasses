"""Extract and save structured person memories from a conversation.

The transcript exists only during Gemini extraction. The local store keeps
selected facts by person and category for later Gemini queries.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .autoenroll import MIN_NAME_CONFIDENCE
from .storage import open_store


def register(subparsers) -> None:  # type: ignore[no-untyped-def]
    remember = subparsers.add_parser(
        "remember",
        help="transcribe a conversation and save selected person facts",
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
    remember.add_argument(
        "--partners",
        type=int,
        default=None,
        help="how many other people were present, to merge surplus speaker clusters",
    )
    remember.add_argument("--gemini-model", default=None, help="Gemini model for memory processing")
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
        help="preview facts without saving them",
    )
    remember.set_defaults(func=cmd_remember)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def cmd_remember(args: argparse.Namespace) -> int:
    from .attribution import (
        apply_remap,
        identities_by_speaker,
        labels_for,
        partner_remap,
        transcribable,
    )
    from .autoenroll import enroll_from_conversation, relabel_utterances
    from .facts import DEFAULT_MODEL, GeminiError, GeminiExtractor, has_api_key
    from .profiles import profile_path, save_encounter
    from .voice.audio import load_audio, write_wav
    from .voice.gemini import GeminiTranscriber
    from .voice.transcribe import transcribe_segments
    from .voice_cli import build_voice_pipeline, expand_audio

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

    store = open_store(args.store)
    wearer = store.wearer()
    wearer_name = wearer.display_name if wearer else "the wearer"

    pipeline = build_voice_pipeline(store, diarize=args.diarize, num_speakers=args.speakers)
    transcriber = GeminiTranscriber(model=args.gemini_model or DEFAULT_MODEL)
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

    saved_people = 0
    for label, samples in recordings:
        encounter_id = args.encounter or f"enc-{uuid.uuid4().hex[:12]}"
        occurred_at = _now_iso()

        result = pipeline.process(samples)
        # Everything audible is transcribed; only the segments long enough
        # to embed decide identity, and they lend it to their short
        # siblings. An introduction is short, and losing it loses the name.
        identities = identities_by_speaker(result.segments, store)
        # The diarizer over-splits; --partners says how many other people
        # were actually there, and folds the surplus clusters together.
        remap = partner_remap(result.segments, identities, partners=args.partners)
        identities = apply_remap(identities, remap)
        usable = transcribable(result.segments)
        speaker_labels = labels_for(usable, identities, remap)
        if remap:
            print(f"  folded {len(remap)} surplus speaker cluster(s) into {args.partners}")
        try:
            utterances = transcribe_segments(
                transcriber, [i.segment for i in usable], speaker_labels
            )
        except GeminiError as error:
            print(f"  ! transcription failed: {error}", file=sys.stderr)
            return 1

        print(f"\n--- {label}  ({encounter_id}) ---")
        if not utterances:
            print("  (nothing intelligible)")
            continue

        try:
            facts = extractor.extract(
                utterances,
                encounter_id=encounter_id,
                wearer_name=wearer_name,
                occurred_at=occurred_at,
            )
        except GeminiError as error:
            print(f"\n  ! extraction failed: {error}", file=sys.stderr)
            return 1

        # Someone who introduced themselves can be enrolled from the
        # conversation itself, which is the difference between facts about
        # "Speaker 1" and facts about a person the next encounter will
        # recognise. Runs before printing so the facts print with the name
        # they will be saved under.
        if not args.no_enroll and not args.dry_run:
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

        if not args.dry_run:
            save_encounter(
                profile_path(store),
                utterances,
                facts,
                encounter_id=encounter_id,
                occurred_at=occurred_at,
                location=args.location,
            )
            saved_people += len(
                {u.person_id for u in utterances if u.person_id and not u.is_wearer}
            )

    if args.dry_run:
        print("\n--dry-run: no memories saved")
        return 0
    print(f"\nsaved structured memories for {saved_people} known person/people")
    print('Now ask: glasses ask "what is this person planning?"')
    return 0
