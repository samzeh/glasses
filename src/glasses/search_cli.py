"""Recall selected person memories with Gemini or inspect them locally."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .facts import DEFAULT_MODEL, GeminiError
from .memory_agent import answer
from .profiles import find_profiles, profile_path

DEMO_PERSON_ID = "seed-person-dan"


def cmd_demo_person(args: argparse.Namespace) -> int:
    """Add or remove an obviously synthetic profile for recall testing."""
    from datetime import UTC, datetime

    from .profiles import load_profiles, remove_profile, save_encounter
    from .search.config import SOURCE_SEED
    from .search.documents import PersonFact
    from .voice.transcribe import Utterance

    path = _memory_path(args)
    if args.clear:
        removed = remove_profile(path, DEMO_PERSON_ID)
        print("Demo profile removed." if removed else "No demo profile to remove.")
        return 0
    existing = load_profiles(path).get(DEMO_PERSON_ID)
    if existing and existing.source != SOURCE_SEED:
        print("Refusing to overwrite a real profile.", file=sys.stderr)
        return 1
    details = (
        ("name", None, "Dan Kim"),
        ("age", None, "24"),
        ("school", None, "UC Berkeley"),
        ("employer", None, "Helix Bio"),
        ("occupation", None, "machine learning engineer"),
        ("hobby", None, "bouldering"),
        ("family", "sister", "Maya"),
        ("plan", "next summer", "visit Japan"),
        ("favorite", "movie", "Spirited Away"),
        ("project", None, "a cryo-EM map viewer"),
        ("goal", None, "start a biotech company"),
    )
    facts = [
        PersonFact(
            encounter_id="seed-encounter-dan",
            person_id=DEMO_PERSON_ID,
            person_name="Dan Kim (demo)",
            kind=kind,
            topic=topic,
            value=value,
            confidence=1.0,
            source=SOURCE_SEED,
        )
        for kind, topic, value in details
    ]
    save_encounter(
        path,
        [Utterance("", 0, 0, person_id=DEMO_PERSON_ID, person_name="Dan Kim (demo)")],
        facts,
        encounter_id="seed-encounter-dan",
        occurred_at=datetime.now(UTC).isoformat(),
        location="Cal Hacks (demo)",
    )
    print("Saved synthetic Dan Kim profile for CLI testing (no face enrolled).")
    print('Try: glasses person "Dan Kim" or glasses ask --person "Dan Kim" "What is his plan?"')
    return 0


def _memory_path(args: argparse.Namespace) -> Path:
    return profile_path(args.store or Path("glasses-store"))


def cmd_ask(args: argparse.Namespace) -> int:
    """Ask Gemini about saved, categorized person memories."""
    question = " ".join(args.question).strip()
    if not question:
        print("ask what?", file=sys.stderr)
        return 1
    try:
        response = answer(question, _memory_path(args), model=args.model, person=args.person)
    except GeminiError as error:
        print(error, file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"answer": response}, indent=2))
    else:
        print(response)
    return 0


def cmd_person(args: argparse.Namespace) -> int:
    """Inspect one person's facts without a network call."""
    matches = find_profiles(_memory_path(args), args.name)
    if not matches:
        print("No saved person matches that name.")
        return 1
    for profile in matches:
        marker = " [synthetic demo]" if profile.source == "seed" else ""
        print(f"{profile.name} ({profile.person_id}){marker}")
        print(f"  First met: {profile.met_at} at {profile.location or 'unknown location'}")
        for kind, items in sorted(profile.facts.items()):
            if args.kind and kind != args.kind:
                continue
            print(f"  {kind}:")
            for item in items:
                topic = f"{item['topic']}: " if item.get("topic") else ""
                print(f"    {topic}{item['value']}")
    return 0


def register(subparsers) -> None:  # type: ignore[no-untyped-def]
    ask = subparsers.add_parser("ask", help="ask Gemini about saved person facts")
    ask.add_argument("question", nargs="+", help='e.g. "what is Dan building?"')
    ask.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model for recall")
    ask.add_argument("--person", help="limit context to a name or person ID")
    ask.add_argument("--json", action="store_true", help="print a JSON answer")
    ask.set_defaults(func=cmd_ask)

    person = subparsers.add_parser("person", help="show saved facts for one person")
    person.add_argument("name", help="name fragment or person ID")
    person.add_argument("--kind", help="show one fact category, e.g. plan or favorite")
    person.set_defaults(func=cmd_person)

    demo = subparsers.add_parser("demo-person", help="load synthetic Dan Kim memories")
    demo.add_argument("--clear", action="store_true", help="remove the synthetic profile")
    demo.set_defaults(func=cmd_demo_person)
