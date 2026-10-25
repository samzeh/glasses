"""The active recall commands use local facts and Gemini."""

from __future__ import annotations

import json

import pytest

from glasses.cli import build_parser
from glasses.profiles import save_encounter
from glasses.search.documents import PersonFact
from glasses.search_cli import cmd_ask, cmd_person
from glasses.voice.transcribe import Utterance


def parse(*argv: str):
    return build_parser().parse_args(argv)


def test_old_transcript_index_command_is_unavailable():
    with pytest.raises(SystemExit):
        parse("search-index", "conversation.wav")


def test_gemini_answer_uses_the_selected_store(tmp_path, monkeypatch, capsys):
    seen = []

    def fake_answer(question, path, *, model, person):
        seen.append((question, path, model, person))
        return "Dan plans to visit Japan. Ask how the trip is coming along."

    monkeypatch.setattr("glasses.search_cli.answer", fake_answer)
    args = parse("--store", str(tmp_path), "ask", "What", "is", "Dan", "planning?")
    assert cmd_ask(args) == 0
    assert seen == [
        ("What is Dan planning?", tmp_path / "encounters.json", "gemini-3.6-flash", None)
    ]
    assert "visit Japan" in capsys.readouterr().out


def test_json_answer_contains_only_the_answer(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("glasses.search_cli.answer", lambda *a, **kw: "A short answer")
    assert cmd_ask(parse("--store", str(tmp_path), "ask", "--json", "Who is Dan?")) == 0
    assert json.loads(capsys.readouterr().out) == {"answer": "A short answer"}


def test_person_command_filters_organized_facts(tmp_path, capsys):
    save_encounter(
        tmp_path / "encounters.json",
        [Utterance("private transcript", 0, 3, person_id="p1", person_name="Test Person")],
        [
            PersonFact(encounter_id="e1", person_id="p1", kind="plan", value="visit Japan"),
            PersonFact(encounter_id="e1", person_id="p1", kind="school", value="Berkeley"),
        ],
        encounter_id="e1",
        occurred_at="2026-10-24T18:00:00+00:00",
    )
    args = parse("--store", str(tmp_path), "person", "Test Person", "--kind", "plan")
    assert cmd_person(args) == 0
    output = capsys.readouterr().out
    assert "visit Japan" in output
    assert "Berkeley" not in output
    assert "private transcript" not in output
