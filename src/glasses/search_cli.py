"""Recall selected person memories with Gemini or inspect them locally."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .facts import DEFAULT_MODEL, GeminiError
from .memory_agent import answer
from .profiles import find_profiles, profile_path


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
        print(f"{profile.name} ({profile.person_id})")
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
    ask.add_argument("question", nargs="+", help='e.g. "what is Sam building?"')
    ask.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model for recall")
    ask.add_argument("--person", help="limit context to a name or person ID")
    ask.add_argument("--json", action="store_true", help="print a JSON answer")
    ask.set_defaults(func=cmd_ask)

    person = subparsers.add_parser("person", help="show saved facts for one person")
    person.add_argument("name", help="name fragment or person ID")
    person.add_argument("--kind", help="show one fact category, e.g. plan or favorite")
    person.set_defaults(func=cmd_person)
