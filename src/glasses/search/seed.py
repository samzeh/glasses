"""Synthetic demo data: four encounters that never happened.

A recall demo needs a past. Two real recordings is not a past, and the
honest way to get one is to fabricate it openly rather than to record
strangers to make a demo look better.

Every document written from here carries ``source: "seed"``, every person
id starts with ``seed-person-`` and every encounter id with
``seed-encounter-``. So:

* ``glasses search-seed --clear`` removes all of it in one call per index
  and cannot touch a real capture;
* ``glasses search-state`` always prints the capture and seed counts
  separately, so nobody demos synthetic data believing it is real;
* any answer the agent gives can be traced to an id that says where it
  came from.

The display names are ordinary names on purpose — "Dan Kim (DEMO)" would
make the demo read as a test harness, and the separation that matters is
the one a query can filter on, not one in the prose. The three questions
this data exists to answer:

    who have I met who works in biotech?
    what did Dan say he'd send me?
    who did I meet at Cal Hacks?
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import SOURCE_SEED
from .documents import PersonFact, TranscriptSegment

WEARER = "you"

#: key -> (person_id, display name). The wearer is in here too: their own
#: speech is indexed, flagged, and excluded from every "about a person"
#: query by the agent's tools rather than by being thrown away.
PEOPLE: dict[str, tuple[str, str]] = {
    WEARER: ("seed-person-sam", "Sam"),
    "dan": ("seed-person-dan", "Dan Kim"),
    "priya": ("seed-person-priya", "Priya Raman"),
    "marcus": ("seed-person-marcus", "Marcus Webb"),
    "elena": ("seed-person-elena", "Elena Garcia"),
}


@dataclass(frozen=True)
class SeedEncounter:
    """One fabricated conversation.

    ``turns`` is ``(person key, text)`` in order. Spans are laid out
    end-to-end at a fixed cadence: the exact milliseconds are not the
    interesting part of a transcript, but having them be monotonic,
    non-overlapping and attached to an encounter is, because that is the
    shape the real pipeline produces.
    """

    id: str
    location: str
    occurred_at: str
    turns: tuple[tuple[str, str], ...]
    turn_ms: int = 6_000


ENCOUNTERS: tuple[SeedEncounter, ...] = (
    SeedEncounter(
        id="seed-encounter-calhacks-dan",
        location="Cal Hacks",
        occurred_at="2026-10-24T18:12:00Z",
        turns=(
            (WEARER, "Hey, I'm Sam. Are you here with a team or solo?"),
            ("dan", "I'm Dan, Dan Kim. Solo this time. I work at Helix Bio, "
             "we're a biotech company doing protein design."),
            (WEARER, "Nice. What are you building this weekend?"),
            ("dan", "A viewer for cryo-EM density maps. Mostly so I stop "
             "squinting at PyMOL screenshots at work."),
            (WEARER, "That sounds genuinely useful. Is the dataset public?"),
            ("dan", "Some of it. I'll send you the benchmark dataset link "
             "and our preprint on Monday, if you want them."),
            (WEARER, "Yes please, that would be great."),
            ("dan", "Cool. I'll email it over. I also promised to introduce "
             "you to Marcus, he's at the table by the window."),
        ),
    ),
    SeedEncounter(
        id="seed-encounter-calhacks-priya",
        location="Cal Hacks",
        occurred_at="2026-10-24T20:40:00Z",
        turns=(
            ("priya", "You're Sam, right? Dan mentioned you. I'm Priya Raman."),
            (WEARER, "I am. What do you work on?"),
            ("priya", "I founded Northbeam Robotics. We do warehouse "
             "picking arms — eleven people, Series A last spring."),
            (WEARER, "How did you end up in robotics?"),
            ("priya", "Mechanical engineering, then four years at a "
             "self-driving company. I got tired of simulation."),
            (WEARER, "Are you hiring?"),
            ("priya", "Always, for perception. Send me anyone who has "
             "shipped a real-time tracker."),
        ),
    ),
    SeedEncounter(
        id="seed-encounter-calhacks-marcus",
        location="Cal Hacks",
        occurred_at="2026-10-24T21:15:00Z",
        turns=(
            (WEARER, "Dan said I should talk to you. Sam."),
            ("marcus", "Marcus Webb. I'm a research scientist at Verida "
             "Therapeutics — biotech, immunology side."),
            (WEARER, "What brings a research scientist to a hackathon?"),
            ("marcus", "My wife is on the organising committee, honestly. "
             "But I'm also looking for ML people who understand assays."),
            (WEARER, "Do you two know each other through work?"),
            ("marcus", "Dan and I overlapped at a conference in Basel. "
             "He's the one who got me interested in protein design."),
        ),
    ),
    SeedEncounter(
        id="seed-encounter-coffee-elena",
        location="Blue Bottle, Oakland",
        occurred_at="2026-09-30T16:05:00Z",
        turns=(
            ("elena", "Elena Garcia. I think we were both at the climate "
             "meetup in June?"),
            (WEARER, "We were. You were talking about rooftop solar."),
            ("elena", "Still am. I run operations at Sunline Energy, we do "
             "residential installs across the East Bay."),
            (WEARER, "How is that going?"),
            ("elena", "Busy. I'll introduce you to our data lead, she has "
             "the same problem you were describing with sensor drift."),
        ),
    ),
)

#: Facts, as the extractor would eventually produce them: a kind, a value,
#: the quote they came from, and a confidence that is not 1.0 because
#: nothing read out of a conversation ever is. ``person_id`` is filled from
#: ``PEOPLE``; an unassigned fact would carry ``None`` here, which the
#: mapping allows and the agent's tools report rather than hide.
FACTS: tuple[tuple[str, str, str, str, str, float], ...] = (
    # (person key, encounter id, kind, value, evidence, confidence)
    ("dan", "seed-encounter-calhacks-dan", "name", "Dan Kim",
     "I'm Dan, Dan Kim.", 0.95),
    ("dan", "seed-encounter-calhacks-dan", "employer", "Helix Bio",
     "I work at Helix Bio, we're a biotech company doing protein design.", 0.92),
    ("dan", "seed-encounter-calhacks-dan", "industry", "biotech",
     "we're a biotech company doing protein design", 0.9),
    ("dan", "seed-encounter-calhacks-dan", "role", "machine learning engineer",
     "A viewer for cryo-EM density maps.", 0.55),
    ("dan", "seed-encounter-calhacks-dan", "commitment",
     "send the benchmark dataset link and the preprint on Monday",
     "I'll send you the benchmark dataset link and our preprint on Monday", 0.88),
    ("dan", "seed-encounter-calhacks-dan", "commitment",
     "introduce Sam to Marcus",
     "I also promised to introduce you to Marcus", 0.72),
    ("dan", "seed-encounter-calhacks-dan", "interest", "protein design",
     "doing protein design", 0.8),
    ("priya", "seed-encounter-calhacks-priya", "name", "Priya Raman",
     "I'm Priya Raman.", 0.95),
    ("priya", "seed-encounter-calhacks-priya", "employer", "Northbeam Robotics",
     "I founded Northbeam Robotics.", 0.93),
    ("priya", "seed-encounter-calhacks-priya", "role", "founder",
     "I founded Northbeam Robotics.", 0.9),
    ("priya", "seed-encounter-calhacks-priya", "industry", "robotics",
     "We do warehouse picking arms", 0.85),
    ("priya", "seed-encounter-calhacks-priya", "commitment",
     "wants referrals for perception engineers",
     "Always, for perception. Send me anyone who has shipped a real-time tracker.", 0.6),
    ("marcus", "seed-encounter-calhacks-marcus", "name", "Marcus Webb",
     "Marcus Webb.", 0.95),
    ("marcus", "seed-encounter-calhacks-marcus", "employer", "Verida Therapeutics",
     "I'm a research scientist at Verida Therapeutics — biotech, immunology side.", 0.93),
    ("marcus", "seed-encounter-calhacks-marcus", "industry", "biotech",
     "biotech, immunology side", 0.91),
    ("marcus", "seed-encounter-calhacks-marcus", "role", "research scientist",
     "I'm a research scientist at Verida Therapeutics", 0.92),
    ("marcus", "seed-encounter-calhacks-marcus", "interest", "immunology",
     "biotech, immunology side", 0.78),
    ("elena", "seed-encounter-coffee-elena", "name", "Elena Garcia",
     "Elena Garcia.", 0.95),
    ("elena", "seed-encounter-coffee-elena", "employer", "Sunline Energy",
     "I run operations at Sunline Energy", 0.9),
    ("elena", "seed-encounter-coffee-elena", "role", "head of operations",
     "I run operations at Sunline Energy", 0.86),
    ("elena", "seed-encounter-coffee-elena", "industry", "residential solar",
     "we do residential installs across the East Bay", 0.84),
    ("elena", "seed-encounter-coffee-elena", "commitment",
     "introduce Sam to her data lead",
     "I'll introduce you to our data lead", 0.8),
)


def seed_segments() -> list[TranscriptSegment]:
    """Every fabricated utterance, on non-overlapping spans."""
    out: list[TranscriptSegment] = []
    for encounter in ENCOUNTERS:
        for index, (key, text) in enumerate(encounter.turns):
            person_id, name = PEOPLE[key]
            start = index * encounter.turn_ms
            out.append(
                TranscriptSegment(
                    encounter_id=encounter.id,
                    start_ms=start,
                    # Leaves a second of silence between turns, which is
                    # what VAD would have produced.
                    end_ms=start + encounter.turn_ms - 1_000,
                    text=text,
                    person_id=person_id,
                    person_name=name,
                    is_wearer=key == WEARER,
                    speaker=str(index % 2),
                    occurred_at=encounter.occurred_at,
                    location=encounter.location,
                    source=SOURCE_SEED,
                    model="seed",
                )
            )
    return out


def seed_facts() -> list[PersonFact]:
    """Every fabricated fact, with the quote it came from."""
    at = {encounter.id: encounter.occurred_at for encounter in ENCOUNTERS}
    out: list[PersonFact] = []
    for key, encounter_id, kind, value, evidence, confidence in FACTS:
        person_id, name = PEOPLE[key]
        out.append(
            PersonFact(
                encounter_id=encounter_id,
                kind=kind,
                value=value,
                person_id=person_id,
                person_name=name,
                evidence_span=evidence,
                confidence=confidence,
                occurred_at=at[encounter_id],
                source=SOURCE_SEED,
            )
        )
    return out
