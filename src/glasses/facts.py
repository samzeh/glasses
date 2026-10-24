"""Transcript to facts, with Gemini.

The cold loop's last step. Transcription says who spoke and what they
said; this says what any of it meant, as rows that outlive the recording:
employer, role, commitment, interest. A transcript is a wall of text
nobody rereads. A fact is something the glasses can hand back the next
time they see the face.

Three rules shape the prompt and the schema, and all three come from the
PRD:

* **Every fact carries the sentence it came from.** ``evidence_span`` is
  not decoration. A profile that asserts something wrong is only
  correctable if you can see what caused it, and a fact with no evidence
  is one you cannot argue with.
* **A fact belongs to the speaker it is about, or to nobody.** Facts are
  attached by speaker name and resolved back to ``person_id`` here, in
  code, not by the model. Where the subject is ambiguous, ``person_id``
  stays null and the fact belongs to the encounter — guessing is how a
  profile ends up asserting that your colleague has three children who
  actually belong to someone they were standing next to.
* **Fewer facts, higher confidence.** Far-field audio in a noisy room
  produces garbled transcripts, and a confidently wrong fact surfaced at
  the wrong moment is worse than an empty profile.

The HTTP seam is a ``Post`` callable, as in ``voice.cartesia``, so the
tests exercise the prompt, the schema and the parsing without a network
or a key.
"""

from __future__ import annotations

import functools
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .search.documents import PersonFact
from .voice.transcribe import Utterance

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"


DEFAULT_MODEL = "gemini-3.5-flash"
FALLBACK_MODELS = ("gemini-3.7-flash", "gemini-3.8-flash", "gemini-flash-latest")

KEY_HINT = (
    "Set GEMINI_API_KEY (a .env beside the repo is read at startup). "
    "Get one at https://aistudio.google.com/apikey"
)

#: The kinds worth surfacing next to a face. A closed set on purpose: an
#: open one drifts into near-duplicates ("job", "work", "employer") that
#: split one person's history across three labels nothing joins back up.
FACT_KINDS = (
    "name",
    "employer",
    "role",
    "industry",
    "project",
    "commitment",
    "interest",
    "location",
    "relationship",
)

#: Below this a fact is noise. Deliberately high: see the module docstring.
MIN_CONFIDENCE = 0.55

Post = Callable[[str, dict[str, str], bytes], bytes]


class GeminiError(RuntimeError):
    """A Gemini request that did not produce facts.

    ``retryable`` separates a rate limit or a 5xx from a refused key, so a
    caller does not spend three attempts on a key that is simply wrong.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


def api_key(explicit: str | None = None) -> str | None:
    """The configured key, or None. Blank counts as unset."""
    key = explicit if explicit is not None else os.environ.get("GEMINI_API_KEY", "")
    key = (key or "").strip()
    return key or None


def has_api_key() -> bool:
    return api_key() is not None


#: Gemini returns exactly this shape, so parsing is indexing rather than
#: hoping. ``subject`` is a speaker *name* — the mapping to person_id
#: happens here, where the utterances are.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "Speaker name the fact is about, exactly as labelled.",
                    },
                    "kind": {"type": "string", "enum": list(FACT_KINDS)},
                    "value": {
                        "type": "string",
                        "description": "The fact itself, as a short phrase.",
                    },
                    "evidence": {
                        "type": "string",
                        "description": "The verbatim sentence from the transcript.",
                    },
                    "confidence": {"type": "number"},
                },
                "required": ["subject", "kind", "value", "evidence", "confidence"],
            },
        }
    },
    "required": ["facts"],
}

PROMPT = """\
You are reading a transcript of a conversation recorded by smart glasses \
worn by {wearer}. Extract durable facts about the people {wearer} spoke \
with, so the glasses can recall them at the next meeting.

Rules:
- Only facts stated or clearly implied in the transcript. Never infer from \
general knowledge, and never invent a detail that would be useful if true.
- `evidence` must be a verbatim sentence copied from the transcript.
- `subject` must be one of these exact speaker names: {names}.
- Attribute a fact to the person it is ABOUT, not whoever said it. If \
{wearer} says "you work at Acme", the fact belongs to the other person.
- Skip facts about {wearer} themselves; the glasses already know the wearer.
- Skip small talk, pleasantries and anything true only during this \
conversation ("is sitting down", "is getting coffee").
- `confidence` is 0 to 1. Be strict: a garbled or half-heard line is low \
confidence. Returning three solid facts beats returning ten shaky ones.
- If nothing durable was said, return an empty list.

Transcript:
{transcript}
"""


def render_transcript(utterances: Iterable[Utterance]) -> str:
    """The transcript as the model sees it: one speaker-labelled line each."""
    lines = []
    for utterance in utterances:
        text = utterance.text.strip()
        if text:
            lines.append(f"[{_speaker_label(utterance)}] {text}")
    return "\n".join(lines)


def _speaker_label(utterance: Utterance) -> str:
    """A stable, human name for one speaker.

    Falls back through the identity the pipeline managed to resolve: a
    matched person, then the anonymous within-recording label, then
    nothing. The label is what the model attributes facts to, so it has to
    exist even when the speaker does not.
    """
    if utterance.person_name:
        return utterance.person_name
    if utterance.speaker is not None:
        return f"Speaker {utterance.speaker}"
    return "Unknown speaker"


def speaker_names(utterances: Iterable[Utterance]) -> list[str]:
    """Every distinct speaker label, in first-appearance order."""
    seen: list[str] = []
    for utterance in utterances:
        label = _speaker_label(utterance)
        if label not in seen:
            seen.append(label)
    return seen


def _identities(utterances: Iterable[Utterance]) -> dict[str, tuple[str | None, str | None]]:
    """Speaker label to the (person_id, person_name) behind it."""
    table: dict[str, tuple[str | None, str | None]] = {}
    for utterance in utterances:
        label = _speaker_label(utterance)
        known = table.get(label)
        if known is None or (known[0] is None and utterance.person_id):
            table[label] = (utterance.person_id, utterance.person_name)
    return table


def _evidence_window(
    utterances: Sequence[Utterance], evidence: str
) -> tuple[int | None, int | None]:
    """Where in the recording the evidence was said, in milliseconds.

    Matched on normalised text rather than offsets: the model copies a
    sentence, and a copy that differs by punctuation or case is still the
    same sentence. No match leaves the window null rather than guessing a
    span, because a wrong timestamp is worse than a missing one.
    """
    needle = _normalise(evidence)
    if not needle:
        return (None, None)
    for utterance in utterances:
        haystack = _normalise(utterance.text)
        if needle and (needle in haystack or haystack in needle):
            return (int(utterance.start_s * 1000), int(utterance.end_s * 1000))
    return (None, None)


def _normalise(text: str) -> str:
    return "".join(c for c in text.casefold() if c.isalnum() or c.isspace()).strip()


class GeminiExtractor:
    """Gemini, one request per encounter, answering in a fixed schema."""

    def __init__(
        self,
        key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        fallback_models: Sequence[str] = FALLBACK_MODELS,
        min_confidence: float = MIN_CONFIDENCE,
        timeout_s: float = 60.0,
        attempts: int = 2,
        retry_wait_s: float = 1.0,
        post: Post | None = None,
    ) -> None:
        resolved = api_key(key)
        if resolved is None:
            raise GeminiError(f"no Gemini API key. {KEY_HINT}")
        self._key = resolved
        self.model = model
        # The chain, preferred first, with the caller's model promoted to
        # the front and no duplicates. See _send for why there is one.
        self.models = [model] + [m for m in fallback_models if m != model]
        self.min_confidence = min_confidence
        self._attempts = max(1, attempts)
        self._retry_wait_s = retry_wait_s
        self._post: Post = post or functools.partial(_post, timeout_s=timeout_s)
        #: Which model actually answered. Not the one that was asked for,
        #: whenever the chain had to move on.
        self.served_by: str | None = None

    @property
    def model_name(self) -> str:
        return f"gemini/{self.served_by or self.model}"

    def extract(
        self,
        utterances: Sequence[Utterance],
        *,
        encounter_id: str,
        wearer_name: str = "the wearer",
        occurred_at: str | None = None,
    ) -> list[PersonFact]:
        """Facts from one encounter's transcript.

        An empty transcript never reaches the API: a request that can only
        answer "nothing" is latency and spend for a known answer.
        """
        transcript = render_transcript(utterances)
        if not transcript:
            return []

        names = speaker_names(utterances)
        prompt = PROMPT.format(
            wearer=wearer_name,
            names=", ".join(repr(name) for name in names),
            transcript=transcript,
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": RESPONSE_SCHEMA,
                # Extraction, not composition. The same transcript should
                # give the same facts twice.
                "temperature": 0.0,
            },
        }
        raw = self._send(json.dumps(payload).encode("utf-8"))
        return self._parse(raw, utterances, encounter_id, wearer_name, occurred_at)

    def _send(self, body: bytes) -> bytes:
        """One request, retried, then tried against the next model.

        Google's flash models return 503 "experiencing high demand" in
        bursts that roll between models: the newest can be refusing while
        an older one answers in seven seconds, and a minute later it is the
        other way round. A single model plus a retry loop therefore still
        fails, because the thing that is down is the model and not the
        account. Walking the chain is what actually gets a transcript
        turned into facts during a demo.

        Only retryable failures move on. A refused key is refused
        everywhere, and trying it four more times just delays the message
        that says so.
        """
        last: GeminiError | None = None
        for model in self.models:
            for attempt in range(self._attempts):
                try:
                    raw = self._post(
                        f"{ENDPOINT}/{model}:generateContent",
                        {"Content-Type": "application/json", "x-goog-api-key": self._key},
                        body,
                    )
                except GeminiError as error:
                    if not error.retryable:
                        raise
                    last = error
                    if attempt < self._attempts - 1:
                        time.sleep(self._retry_wait_s * (attempt + 1))
                    continue
                self.served_by = model
                return raw
        raise last or GeminiError("no Gemini model answered", retryable=True)

    def _parse(
        self,
        raw: bytes,
        utterances: Sequence[Utterance],
        encounter_id: str,
        wearer_name: str,
        occurred_at: str | None,
    ) -> list[PersonFact]:
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GeminiError(f"Gemini returned something that is not JSON: {error}") from error

        try:
            text = body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as error:
            blocked = body.get("promptFeedback", {}).get("blockReason")
            if blocked:
                raise GeminiError(f"Gemini blocked the prompt: {blocked}") from error
            raise GeminiError(f"no candidate in Gemini's reply: {body!r:.300}") from error

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            raise GeminiError(f"Gemini's JSON did not parse: {error}") from error

        identities = _identities(utterances)
        facts: list[PersonFact] = []
        for item in parsed.get("facts") or []:
            fact = self._one(item, identities, utterances, encounter_id, wearer_name, occurred_at)
            if fact is not None:
                facts.append(fact)
        return facts

    def _one(
        self,
        item: Any,
        identities: dict[str, tuple[str | None, str | None]],
        utterances: Sequence[Utterance],
        encounter_id: str,
        wearer_name: str,
        occurred_at: str | None,
    ) -> PersonFact | None:
        """One item from the model to one PersonFact, or None if it fails a gate.

        Four things are rejected here rather than indexed: an unknown kind,
        a blank value, a subject that is not a speaker in this recording,
        and anything under the confidence floor. All four are the model
        being creative, and the index is not where to discover that.
        """
        if not isinstance(item, dict):
            return None

        kind = str(item.get("kind") or "").strip()
        value = str(item.get("value") or "").strip()
        subject = str(item.get("subject") or "").strip()
        if kind not in FACT_KINDS or not value or not subject:
            return None

        # A fact about the wearer is one the glasses already have, and
        # letting it through means their own life shows up as a profile.
        if subject.casefold() == wearer_name.casefold():
            return None

        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            return None
        if confidence < self.min_confidence:
            return None

        # An invented speaker means the attribution is guesswork. Keep the
        # fact against the encounter rather than hanging it on a stranger.
        person_id, person_name = identities.get(subject, (None, None))
        if subject not in identities:
            person_name = None

        evidence = str(item.get("evidence") or "").strip() or None
        start_ms, end_ms = _evidence_window(utterances, evidence or "")
        return PersonFact(
            encounter_id=encounter_id,
            kind=kind,
            value=value,
            person_id=person_id,
            person_name=person_name or (subject if subject in identities else None),
            evidence_span=evidence,
            evidence_start_ms=start_ms,
            evidence_end_ms=end_ms,
            confidence=min(1.0, max(0.0, confidence)),
            occurred_at=occurred_at,
        )


def _post(url: str, headers: dict[str, str], body: bytes, timeout_s: float = 60.0) -> bytes:
    """One POST over urllib, with every failure mapped to GeminiError."""
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return bytes(response.read())
    except urllib.error.HTTPError as error:
        detail = _detail(error)
        if error.code in (401, 403):
            raise GeminiError(
                f"Gemini rejected the API key (HTTP {error.code}). {KEY_HINT}{detail}",
                retryable=False,
            ) from error
        raise GeminiError(
            f"Gemini returned HTTP {error.code}.{detail}",
            retryable=error.code == 429 or error.code >= 500,
        ) from error
    except urllib.error.URLError as error:
        raise GeminiError(f"could not reach Gemini: {error.reason}", retryable=True) from error
    except TimeoutError as error:
        # A read that times out mid-response raises the builtin, not a
        # URLError, so it escaped uncaught and took the whole command down
        # — transcript included. Extraction is allowed to fail; the
        # transcript above it is the expensive half.
        raise GeminiError(
            f"Gemini did not answer within {timeout_s:.0f}s", retryable=True
        ) from error
    except OSError as error:
        raise GeminiError(f"could not reach Gemini: {error}", retryable=True) from error


def _detail(error: urllib.error.HTTPError) -> str:
    try:
        body = error.read().decode("utf-8", "replace").strip()
    except Exception:  # noqa: BLE001 - the detail is a nicety, never the failure
        return ""
    return f" {body[:400]}" if body else ""


__all__ = [
    "FACT_KINDS",
    "MIN_CONFIDENCE",
    "GeminiError",
    "GeminiExtractor",
    "api_key",
    "has_api_key",
    "render_transcript",
    "speaker_names",
]
