"""Gemini answers questions from structured person memories only."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from .facts import DEFAULT_MODEL, GeminiError, GeminiExtractor
from .profiles import Profile, load_profiles

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

PROMPT = """\
You help the wearer recall useful facts about people they have met. Answer
the question using ONLY the structured memories below. They contain no full
conversation transcript. If a detail is absent, say it is not saved. Do not
guess, and do not treat a past plan or goal as completed. When useful, suggest
a natural follow-up question about a saved plan, project, or goal.
If facts conflict, prefer the most recently observed one and say what remains uncertain.
If a memory has source "seed", clearly say it is synthetic demo data.

Question: {question}

Memories (JSON):
{memories}
"""


def _context(profiles: dict[str, Profile]) -> list[dict]:
    return [
        {
            "person_id": profile.person_id,
            "name": profile.name,
            "first_met_at": profile.met_at,
            "first_met_location": profile.location,
            "source": profile.source,
            "facts": profile.facts,
        }
        for profile in profiles.values()
    ]


def answer(
    question: str,
    path: Path,
    *,
    key: str | None = None,
    post: Callable[[str, dict[str, str], bytes], bytes] | None = None,
    model: str = DEFAULT_MODEL,
    person: str | None = None,
) -> str:
    """Answer from saved facts. Gemini sees no transcript or audio."""
    profiles = load_profiles(path)
    if person:
        needle = person.strip().casefold()
        profiles = {
            person_id: profile
            for person_id, profile in profiles.items()
            if profile.person_id.casefold() == needle or needle in profile.name.casefold()
        }
    if not profiles:
        return "No person memories are saved yet."
    prompt = PROMPT.format(
        question=question.strip(),
        memories=json.dumps(_context(profiles), ensure_ascii=False),
    )
    client = GeminiExtractor(key, model=model, post=post)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": ANSWER_SCHEMA,
            "temperature": 0.0,
        },
    }
    raw = client._send(json.dumps(payload).encode("utf-8"))
    try:
        body = json.loads(raw)
        content = json.loads(body["candidates"][0]["content"]["parts"][0]["text"])
        result = content["answer"]
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise GeminiError("Gemini returned an invalid recall answer") from error
    if not isinstance(result, str) or not result.strip():
        raise GeminiError("Gemini returned an empty recall answer")
    return result.strip()
