"""Small, local encounter summaries for immediate face-triggered recall.

The search index keeps the full transcript and facts. This file keeps the
four short fields the glasses need when a face is recognised, keyed by the
gallery's person ID rather than by a name that may collide.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Profile:
    person_id: str
    name: str
    met_at: str
    location: str | None
    fun_fact: str | None
    encounter_id: str


def profile_path(store: Any) -> Path:
    return Path(getattr(store, "path", "glasses-store")) / "encounters.json"


def load_profiles(path: Path) -> dict[str, Profile]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {person_id: Profile(**item) for person_id, item in data.items()}


def save_encounter(
    path: Path,
    utterances: list[Any],
    facts: list[Any],
    *,
    encounter_id: str,
    occurred_at: str,
    location: str | None = None,
    choose: Any = random.choice,
) -> dict[str, Profile]:
    """Remember each identified participant, even when no fact was extracted.

    The first meeting date and its location stay fixed. A later conversation
    may refresh the fun fact. Reprocessing the same encounter is idempotent.
    """
    profiles = load_profiles(path)
    participants = {
        u.person_id: u.person_name
        for u in utterances
        if u.person_id and u.person_name and not u.is_wearer
    }
    for person_id, name in participants.items():
        old = profiles.get(person_id)
        candidates = [
            f"{fact.kind.capitalize()}: {fact.value.strip()}"
            for fact in facts
            if fact.person_id == person_id
            and fact.kind in {"interest", "role", "employer", "project", "industry"}
            and fact.value.strip()
        ]
        # Keep the chosen fact stable when the same recording is retried.
        fun_fact = old.fun_fact if old and old.encounter_id == encounter_id else None
        if fun_fact is None:
            fun_fact = choose(candidates) if candidates else old.fun_fact if old else None
        first = old is None or datetime.fromisoformat(occurred_at) < datetime.fromisoformat(
            old.met_at
        )
        profiles[person_id] = Profile(
            person_id=person_id,
            name=name,
            met_at=occurred_at if first else old.met_at,
            location=location if first else old.location,
            fun_fact=fun_fact,
            encounter_id=encounter_id,
        )
    if participants:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({key: asdict(value) for key, value in profiles.items()}, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
    return profiles


def display_fields(profile: Profile) -> tuple[str, str, str, str]:
    date = datetime.fromisoformat(profile.met_at).astimezone().strftime("%b %d, %Y")
    return profile.name, date, profile.location or "Unknown", profile.fun_fact or "None saved"


def serial_message(profile: Profile) -> bytes:
    """One line with four tab-separated, display-safe fields for the ESP32."""
    fields = [" ".join(value.split()) for value in display_fields(profile)]
    return ("PROFILE:" + "\t".join(fields) + "\n").encode("ascii", errors="replace")
