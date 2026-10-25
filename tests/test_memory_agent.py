"""Gemini recall sees selected memories, never the source conversation."""

from __future__ import annotations

import json

from glasses.memory_agent import answer
from glasses.profiles import find_profiles, save_encounter
from glasses.search.documents import PersonFact
from glasses.voice.transcribe import Utterance


def test_structured_memories_are_queryable_without_a_transcript(tmp_path):
    path = tmp_path / "encounters.json"
    spoken = [
        Utterance(
            text="This entire private conversation must not be written to disk.",
            start_s=0,
            end_s=4,
            person_id="p1",
            person_name="Test Person",
        ),
        Utterance("I live in Oakland", 5, 8, person_id="p2", person_name="Maya Chen"),
    ]
    facts = [
        PersonFact(
            encounter_id="e1",
            person_id="p1",
            kind="plan",
            topic="next summer",
            value="visit Japan",
            confidence=0.9,
        ),
        PersonFact(
            encounter_id="e1",
            person_id="p1",
            kind="favorite",
            topic="movie",
            value="Spirited Away",
            confidence=0.8,
        ),
        PersonFact(encounter_id="e1", person_id="p2", kind="location", value="Oakland"),
    ]
    save_encounter(path, spoken, facts, encounter_id="e1", occurred_at="2026-10-24T18:00:00+00:00")
    stored = path.read_text()
    assert "This entire private conversation" not in stored
    [profile] = find_profiles(path, "Test Person")
    assert profile.facts["plan"][0]["topic"] == "next summer"
    assert profile.facts["favorite"][0]["value"] == "Spirited Away"

    requests = []

    def post(url, headers, body):
        requests.append(json.loads(body))
        result = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": json.dumps({"answer": "Ask about the Japan trip."})}]
                    }
                }
            ]
        }
        return json.dumps(result).encode()

    response = answer(
        "What should I ask Test Person?", path, key="test-key", post=post, person="Test Person"
    )
    assert "Japan trip" in response
    prompt = requests[0]["contents"][0]["parts"][0]["text"]
    assert "visit Japan" in prompt
    assert "Spirited Away" in prompt
    assert "This entire private conversation" not in prompt
    assert "Maya Chen" not in prompt


def test_repeating_an_encounter_does_not_duplicate_a_fact(tmp_path):
    path = tmp_path / "encounters.json"
    utterances = [Utterance("hello", 0, 3, person_id="p1", person_name="Dan")]
    facts = [PersonFact(encounter_id="e1", person_id="p1", kind="school", value="Berkeley")]
    for _ in range(2):
        save_encounter(
            path, utterances, facts, encounter_id="e1", occurred_at="2026-10-24T18:00:00+00:00"
        )
    assert len(find_profiles(path, "p1")[0].facts["school"]) == 1
