"""Fact extraction: the prompt, the gates, and the mapping back to people.

Every test runs against an injected ``post``, so the suite needs no key
and no network.
"""

from __future__ import annotations

import json

import pytest

from glasses import facts as F
from glasses.voice.transcribe import Utterance


def utterance(text, start=0.0, end=1.0, speaker=0, person_id=None, person_name=None):
    return Utterance(
        text=text,
        start_s=start,
        end_s=end,
        speaker=speaker,
        person_id=person_id,
        person_name=person_name,
    )


CONVERSATION = [
    utterance("Good to meet you.", 0.0, 1.5, speaker=0, person_id="w1", person_name="Sam"),
    utterance(
        "I work at Helix Bio, we do protein design.",
        1.5,
        4.0,
        speaker=1,
        person_id="p1",
        person_name="Dan Kim",
    ),
    utterance("I'll send you the preprint on Monday.", 4.0, 6.0, speaker=1, person_id="p1",
              person_name="Dan Kim"),
]


def reply(facts):
    """A Gemini response envelope around a facts payload."""
    body = {"candidates": [{"content": {"parts": [{"text": json.dumps({"facts": facts})}]}}]}
    return json.dumps(body).encode("utf-8")


def extractor(facts, *, captured=None, **kwargs):
    def post(url, headers, payload):
        if captured is not None:
            captured.append((url, headers, json.loads(payload.decode())))
        return reply(facts)

    return F.GeminiExtractor("test-key", post=post, **kwargs)


GOOD = {
    "subject": "Dan Kim",
    "kind": "employer",
    "value": "Helix Bio",
    "evidence": "I work at Helix Bio, we do protein design.",
    "confidence": 0.95,
}


# --- the request ---------------------------------------------------------


def test_the_transcript_is_speaker_labelled():
    rendered = F.render_transcript(CONVERSATION)
    assert "[Sam] Good to meet you." in rendered
    assert "[Dan Kim] I work at Helix Bio, we do protein design." in rendered


def test_a_speaker_with_no_identity_still_gets_a_label():
    """An unresolved speaker's words still belong to the encounter."""
    rendered = F.render_transcript([utterance("Hello", speaker=3)])
    assert "[Speaker 3] Hello" in rendered


def test_blank_utterances_are_left_out():
    assert F.render_transcript([utterance("   "), utterance("real")]) == "[Speaker 0] real"


def test_speaker_names_are_in_first_appearance_order():
    assert F.speaker_names(CONVERSATION) == ["Sam", "Dan Kim"]


def test_the_request_pins_the_schema_and_zero_temperature():
    captured = []
    extractor([GOOD], captured=captured).extract(CONVERSATION, encounter_id="e1")

    url, headers, body = captured[0]
    assert f"{F.DEFAULT_MODEL}:generateContent" in url
    assert headers["x-goog-api-key"] == "test-key"
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseSchema"] == F.RESPONSE_SCHEMA
    assert config["temperature"] == 0.0


def test_the_prompt_names_the_speakers_and_the_wearer():
    captured = []
    extractor([GOOD], captured=captured).extract(
        CONVERSATION, encounter_id="e1", wearer_name="Sam"
    )
    prompt = captured[0][2]["contents"][0]["parts"][0]["text"]
    assert "'Dan Kim'" in prompt
    assert "Sam" in prompt


def test_an_empty_transcript_never_reaches_the_api():
    """A request that can only answer 'nothing' is spend for a known answer."""

    def explode(*_):
        raise AssertionError("should not have called Gemini")

    got = F.GeminiExtractor("k", post=explode).extract([], encounter_id="e1")
    assert got == []


# --- the facts that come back --------------------------------------------


def test_a_fact_is_mapped_back_to_the_person_who_said_it():
    [fact] = extractor([GOOD]).extract(CONVERSATION, encounter_id="e1")
    assert (fact.kind, fact.value) == ("employer", "Helix Bio")
    assert fact.person_id == "p1"
    assert fact.person_name == "Dan Kim"
    assert fact.encounter_id == "e1"


def test_the_evidence_span_is_kept():
    """A fact with no evidence is one you cannot argue with."""
    [fact] = extractor([GOOD]).extract(CONVERSATION, encounter_id="e1")
    assert fact.evidence_span == GOOD["evidence"]


def test_the_evidence_is_located_in_the_recording():
    [fact] = extractor([GOOD]).extract(CONVERSATION, encounter_id="e1")
    assert (fact.evidence_start_ms, fact.evidence_end_ms) == (1500, 4000)


def test_evidence_matching_survives_punctuation_and_case():
    item = dict(GOOD, evidence="i work at helix bio we do protein design")
    [fact] = extractor([item]).extract(CONVERSATION, encounter_id="e1")
    assert fact.evidence_start_ms == 1500


def test_unlocatable_evidence_leaves_the_window_null():
    """A wrong timestamp is worse than a missing one."""
    item = dict(GOOD, evidence="a sentence nobody said")
    [fact] = extractor([item]).extract(CONVERSATION, encounter_id="e1")
    assert (fact.evidence_start_ms, fact.evidence_end_ms) == (None, None)


# --- the gates -----------------------------------------------------------


def test_a_low_confidence_fact_is_dropped():
    assert extractor([dict(GOOD, confidence=0.2)]).extract(CONVERSATION, encounter_id="e1") == []


def test_the_confidence_floor_is_configurable():
    item = dict(GOOD, confidence=0.4)
    got = extractor([item], min_confidence=0.3).extract(CONVERSATION, encounter_id="e1")
    assert len(got) == 1


def test_an_unknown_kind_is_dropped():
    """A kind outside the closed set splits one history across labels."""
    assert extractor([dict(GOOD, kind="vibe")]).extract(CONVERSATION, encounter_id="e1") == []


@pytest.mark.parametrize("field", ["value", "subject"])
def test_a_blank_required_field_is_dropped(field):
    assert extractor([dict(GOOD, **{field: "  "})]).extract(CONVERSATION, encounter_id="e1") == []


def test_facts_about_the_wearer_are_dropped():
    """The glasses already know the wearer; their life is not a profile."""
    item = dict(GOOD, subject="Sam")
    got = extractor([item]).extract(CONVERSATION, encounter_id="e1", wearer_name="Sam")
    assert got == []


def test_an_invented_speaker_keeps_the_fact_but_not_the_identity():
    """Ambiguity attaches to the encounter rather than to a stranger."""
    item = dict(GOOD, subject="Someone Not Here")
    [fact] = extractor([item]).extract(CONVERSATION, encounter_id="e1")
    assert fact.person_id is None
    assert fact.person_name is None
    assert fact.encounter_id == "e1"


def test_a_non_numeric_confidence_is_dropped():
    assert extractor([dict(GOOD, confidence="high")]).extract(
        CONVERSATION, encounter_id="e1"
    ) == []


def test_confidence_is_clamped():
    [fact] = extractor([dict(GOOD, confidence=4.0)]).extract(CONVERSATION, encounter_id="e1")
    assert fact.confidence == 1.0


def test_junk_items_are_skipped_without_killing_the_batch():
    got = extractor(["not a dict", GOOD]).extract(CONVERSATION, encounter_id="e1")
    assert len(got) == 1


# --- failures ------------------------------------------------------------


def test_no_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(F.GeminiError, match="GEMINI_API_KEY"):
        F.GeminiExtractor()


def test_a_blank_key_counts_as_unset(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    assert F.api_key() is None
    assert not F.has_api_key()


def test_a_non_json_body_is_a_gemini_error():
    bad = F.GeminiExtractor("k", post=lambda *_: b"<html>502</html>")
    with pytest.raises(F.GeminiError, match="not JSON"):
        bad.extract(CONVERSATION, encounter_id="e1")


def test_a_blocked_prompt_says_so():
    body = json.dumps({"promptFeedback": {"blockReason": "SAFETY"}}).encode()
    blocked = F.GeminiExtractor("k", post=lambda *_: body)
    with pytest.raises(F.GeminiError, match="SAFETY"):
        blocked.extract(CONVERSATION, encounter_id="e1")


def test_candidate_text_that_is_not_json_is_a_gemini_error():
    body = json.dumps(
        {"candidates": [{"content": {"parts": [{"text": "sorry, I cannot"}]}}]}
    ).encode()
    bad = F.GeminiExtractor("k", post=lambda *_: body)
    with pytest.raises(F.GeminiError, match="did not parse"):
        bad.extract(CONVERSATION, encounter_id="e1")


def _urlopen_raising(monkeypatch, error):
    """Make the real ``_post`` fail the way the network does.

    These go through ``_post`` rather than an injected ``post``: the
    mapping from a socket failure to GeminiError lives there, so a test
    that injects its own callable would pass with the bug still in.
    """
    import urllib.request

    def boom(*_a, **_k):
        raise error

    monkeypatch.setattr(urllib.request, "urlopen", boom)


def test_a_timeout_becomes_a_retryable_gemini_error(monkeypatch):
    """A read that times out mid-response raises the builtin TimeoutError,
    not a URLError. It escaped ``_post`` uncaught and took the whole
    command down, transcript included."""
    _urlopen_raising(monkeypatch, TimeoutError("timed out"))

    with pytest.raises(F.GeminiError, match="did not answer") as caught:
        F._post("https://example.invalid", {}, b"{}", timeout_s=1.0)
    assert caught.value.retryable


def test_a_socket_error_becomes_a_retryable_gemini_error(monkeypatch):
    _urlopen_raising(monkeypatch, OSError("connection reset"))

    with pytest.raises(F.GeminiError, match="could not reach") as caught:
        F._post("https://example.invalid", {}, b"{}")
    assert caught.value.retryable
