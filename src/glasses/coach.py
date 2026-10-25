"""What the OLED shows for a person: talking points, or what was just learned.

A known face gets the agent's suggestions for what to bring up, from their
saved facts and, while a conversation is being recorded, what has been said
so far. A person met for the first time gets the facts just saved about
them instead: there is nothing to suggest from yet.

Gemini calls take a second or two, so suggestions are worked out on a
background thread and the card is sent when they arrive, unless the face
in front has changed in the meantime.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import Any

from .facts import DEFAULT_MODEL, GeminiError, GeminiExtractor

SUGGEST_SCHEMA = {
    "type": "object",
    "properties": {"suggestions": {"type": "array", "items": {"type": "string"}}},
    "required": ["suggestions"],
}

PROMPT = """\
You help someone wearing smart glasses keep a conversation going with \
{name}. Suggest the single best thing to bring up or ask {name} about next.

Rules:
- If there is a conversation so far, it must follow the topic being talked \
about right now; use the saved facts to make it personal. With no \
conversation yet, pick from the saved facts.
- Prefer follow-ups on plans, projects, goals and interests.
- Do not repeat something the transcript shows was already covered.
- Under 60 characters, written as a prompt to the wearer \
("Ask how the CalHacks demo went").
- Never invent details that are not in the facts or the transcript.
- Refer to {name} by name or "they", never a gendered pronoun.

Saved facts about {name} (JSON):
{facts}

Conversation so far (may be empty):
{transcript}
"""


def suggest(
    name: str, facts: list[dict[str, Any]], transcript: str = "", *, model: str = DEFAULT_MODEL
) -> list[str]:
    """Talking points for this person, or [] when there is nothing to go on."""
    if not facts and not transcript.strip():
        return []
    prompt = PROMPT.format(
        name=name,
        facts=json.dumps(
            [{"kind": f["kind"], "value": f["value"]} for f in facts], ensure_ascii=False
        ),
        transcript=transcript.strip() or "(nothing yet)",
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": SUGGEST_SCHEMA,
            "temperature": 0.4,
        },
    }
    raw = GeminiExtractor(model=model)._send(json.dumps(payload).encode("utf-8"))
    try:
        body = json.loads(raw)
        items = json.loads(body["candidates"][0]["content"]["parts"][0]["text"])["suggestions"]
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise GeminiError("Gemini returned invalid suggestions") from error
    return [s.strip() for s in items if isinstance(s, str) and s.strip()][:1]


def card(name: str, met: str, where: str, body: str) -> bytes:
    """One ``PROFILE:`` line. The firmware labels the fields Name, Met, Where
    and Fact; the last one scrolls, so it carries the list."""
    fields = [" ".join(str(v).split()) or "-" for v in (name, met, where, body)]
    return ("PROFILE:" + "\t".join(fields) + "\n").encode("ascii", errors="replace")


def _date(when: datetime | None) -> str:
    return when.astimezone().strftime("%b %d, %Y") if when else "-"


class Coach:
    def __init__(self, store: Any, link: Any, *, log=print) -> None:
        self.store = store
        self.link = link
        self.log = log
        #: Who the screen is about right now; a late answer for anyone else is dropped.
        self.current: str | None = None
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()

    def show(self, person_id: str) -> None:
        """A known face came into view: their talking points."""
        self.current = person_id
        if person_id in self._cache:
            self._send(person_id, self._cache[person_id])
            return
        self._send(person_id, "Thinking of topics...")
        self.refresh(person_id)

    def refresh(self, person_id: str, transcript: str = "") -> None:
        """Work the suggestions out again, with the conversation so far."""
        threading.Thread(target=self._work, args=(person_id, transcript), daemon=True).start()

    def show_facts(self, person_id: str, facts: list[Any]) -> None:
        """Someone met for the first time: what was just saved about them."""
        self.current = person_id
        values = [f"{f.kind}: {f.value}" for f in facts if f.kind != "name" and f.value.strip()]
        body = " | ".join(values) if values else "Profile saved, no facts yet"
        # Cached so it stays up when they come back into view this run; the
        # next conversation with them refreshes it into talking points.
        self._cache[person_id] = body
        self._send(person_id, body)

    def _work(self, person_id: str, transcript: str) -> None:
        name = self.store.name_of(person_id)
        try:
            ideas = suggest(name, self.store.facts_of(person_id), transcript)
        except GeminiError as error:
            self.log(f"[coach] no suggestions for {name}: {error}")
            return
        body = ideas[0] if ideas else "Nothing saved yet"
        self.log(f"[coach] {name}: {body}")
        self._cache[person_id] = body
        if self.current == person_id:
            self._send(person_id, body)

    def _send(self, person_id: str, body: str) -> None:
        if self.link is None:
            return
        name = self.store.name_of(person_id)
        with self._lock:
            self.link.write(card(name, _date(self.store.first_met(person_id)), "-", body))
