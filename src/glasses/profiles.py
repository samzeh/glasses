"""Structured person memories for local recall and the OLED.

Only selected facts are written. Transcripts and audio stay in memory during
extraction and are not part of this store.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
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
    facts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    source: str = "capture"


def profile_path(store: Any) -> Path:
    root = store if isinstance(store, (Path, str)) else getattr(store, "path", "glasses-store")
    return Path(root) / "encounters.json"


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
        grouped = {kind: list(items) for kind, items in old.facts.items()} if old else {}
        for fact in facts:
            if fact.person_id != person_id or not fact.value.strip():
                continue
            entry = {
                "topic": fact.topic,
                "value": fact.value.strip(),
                "confidence": fact.confidence,
                "observed_at": occurred_at,
                "encounter_id": encounter_id,
                "source": fact.source,
            }
            bucket = grouped.setdefault(fact.kind, [])
            key = (str(fact.topic or "").casefold(), fact.value.strip().casefold())
            existing = next(
                (
                    index
                    for index, item in enumerate(bucket)
                    if (str(item.get("topic") or "").casefold(), item["value"].casefold()) == key
                ),
                None,
            )
            if existing is None:
                bucket.append(entry)
            else:
                entry["confidence"] = max(bucket[existing]["confidence"], fact.confidence)
                bucket[existing] = entry
        candidates = [
            f"{fact.topic.capitalize() if fact.topic else fact.kind.capitalize()}: "
            f"{fact.value.strip()}"
            for fact in facts
            if fact.person_id == person_id
            and fact.kind
            in {"hobby", "favorite", "interest", "project", "role", "occupation", "employer"}
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
            facts=grouped,
            source="capture",
        )
    if participants:
        _write_profiles(path, profiles)
    return profiles


def _write_profiles(path: Path, profiles: dict[str, Profile]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({key: asdict(value) for key, value in profiles.items()}, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def find_profiles(path: Path, query: str) -> list[Profile]:
    """Look up people by exact ID or a case-insensitive name fragment."""
    needle = query.strip().casefold()
    if not needle:
        return []
    return [
        profile
        for profile in load_profiles(path).values()
        if profile.person_id.casefold() == needle or needle in profile.name.casefold()
    ]


def display_fields(profile: Profile) -> tuple[str, str, str, str]:
    date = datetime.fromisoformat(profile.met_at).astimezone().strftime("%b %d, %Y")
    return profile.name, date, profile.location or "Unknown", profile.fun_fact or "None saved"


def serial_message(profile: Profile) -> bytes:
    """One line with four tab-separated, display-safe fields for the ESP32."""
    fields = [" ".join(value.split()) for value in display_fields(profile)]
    return ("PROFILE:" + "\t".join(fields) + "\n").encode("ascii", errors="replace")
