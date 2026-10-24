"""CLI wiring. The commands that need models are exercised by hand, not here —
these cover argument parsing and the paths that touch the store.
"""

from __future__ import annotations

import pytest

from glasses.cli import (
    build_parser,
    cmd_enroll,
    cmd_forget,
    cmd_people,
    expand_images,
    main,
    too_similar,
)
from glasses.store import LocalStore

from .fakes import unit

CONSENT = "agreed in person"


def parse(*argv: str):
    return build_parser().parse_args(argv)


def test_every_subcommand_parses():
    for argv in (
        ("check",),
        ("enroll", "--name", "Dan", "--consent", "yes", "a.jpg", "b.jpg"),
        ("identify", "a.jpg"),
        ("watch",),
        ("people",),
        ("forget", "Dan"),
        ("calibrate-pose", "dir/"),
        ("bench", "a.jpg"),
        ("voice-check",),
        ("enroll-voice", "--name", "Dan", "--consent", "yes", "--mic"),
        ("enroll-voice", "--name", "Dan", "--consent", "yes", "a.wav"),
        ("identify-voice", "--diarize", "--speakers", "2", "a.wav"),
        ("identify-voice", "--mic", "--seconds", "10"),
        ("forget-voice", "Dan"),
        ("transcribe", "a.wav"),
        ("transcribe", "--mic", "--model", "tiny.en"),
        ("transcribe", "--local", "a.wav"),
        ("transcribe", "--language", "es", "a.wav"),
        ("voice-compare", "a.wav", "b.wav"),
        ("voice-eval", "recordings/"),
    ):
        assert parse(*argv).command == argv[0]


def test_enroll_requires_a_consent_note():
    """Forgetting the flag has to be an error, not a default."""
    with pytest.raises(SystemExit):
        parse("enroll", "--name", "Dan", "a.jpg")


def test_enroll_defaults_to_camera_zero():
    assert parse("enroll", "--name", "D", "--consent", "y", "--camera").camera == 0
    assert parse("enroll", "--name", "D", "--consent", "y", "--camera", "2").camera == 2


def test_enroll_without_a_source_is_refused(capsys):
    """Neither photos nor a camera means there is nothing to enroll from."""
    assert cmd_enroll(parse("enroll", "--name", "D", "--consent", "y")) == 1
    assert "either --camera" in capsys.readouterr().err


def test_enroll_with_both_sources_is_refused(capsys):
    argv = parse("enroll", "--name", "D", "--consent", "y", "--camera", "0", "a.jpg")
    assert cmd_enroll(argv) == 1
    assert "not both" in capsys.readouterr().err


def test_enroll_voice_without_a_source_is_refused(capsys):
    from glasses.cli import build_parser as bp
    from glasses.voice_cli import cmd_enroll_voice

    argv = bp().parse_args(["enroll-voice", "--name", "D", "--consent", "y"])
    assert cmd_enroll_voice(argv) == 1
    assert "either --mic" in capsys.readouterr().err


def test_enroll_voice_with_both_sources_is_refused(capsys):
    from glasses.cli import build_parser as bp
    from glasses.voice_cli import cmd_enroll_voice

    argv = bp().parse_args(["enroll-voice", "--name", "D", "--consent", "y", "--mic", "a.wav"])
    assert cmd_enroll_voice(argv) == 1
    assert "not both" in capsys.readouterr().err


def test_transcribe_defaults_to_cartesia_and_english():
    """The default path is the hosted one; --local and --model are the escape
    hatches, not the normal case."""
    argv = parse("transcribe", "a.wav")

    assert argv.local is False
    assert argv.language == "en"
    assert argv.model == "base.en"


def test_too_similar_rejects_a_near_duplicate_view():
    """Ten copies of one angle is worse than three different ones — the top-3
    matcher averages near-duplicates into the number one template would give."""
    first = unit([1.0, 0.0, 0.0])
    near = unit([0.999, 0.01, 0.0])
    different = unit([0.6, 0.8, 0.0])

    assert too_similar(near, [first], 0.97)
    assert not too_similar(different, [first], 0.97)


def test_too_similar_checks_against_every_accepted_view():
    accepted = [unit([1.0, 0.0, 0.0]), unit([0.0, 1.0, 0.0])]

    assert too_similar(unit([0.01, 0.999, 0.0]), accepted, 0.97)
    assert not too_similar(unit([0.0, 0.0, 1.0]), accepted, 0.97)


def test_nothing_is_too_similar_to_an_empty_set():
    assert not too_similar(unit([1.0, 0.0]), [], 0.97)


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        parse()


def test_the_store_path_is_overridable(tmp_path):
    assert parse("--store", str(tmp_path), "people").store == tmp_path


def test_expand_images_walks_a_directory(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.jpg").touch()
    (tmp_path / "nested" / "b.PNG").touch()
    (tmp_path / "notes.txt").touch()

    found = expand_images([str(tmp_path)])

    assert [p.name for p in found] == ["a.jpg", "b.PNG"]


def test_expand_images_passes_explicit_files_through_unfiltered(tmp_path):
    path = tmp_path / "frame.ppm"
    path.touch()

    assert expand_images([str(path)]) == [path]


def test_people_lists_an_empty_store(tmp_path, capsys):
    assert cmd_people(parse("--store", str(tmp_path), "people")) == 0
    assert "nobody enrolled" in capsys.readouterr().out


def test_people_shows_the_consent_record(tmp_path, capsys):
    store = LocalStore.load(tmp_path)
    store.enroll("Dan Kim", [unit([1.0, 0.0])], consent_note=CONSENT)
    store.save()

    cmd_people(parse("--store", str(tmp_path), "people"))

    out = capsys.readouterr().out
    assert "Dan Kim" in out
    assert CONSENT in out


def test_forget_removes_the_person(tmp_path, capsys):
    store = LocalStore.load(tmp_path)
    store.enroll("Dan", [unit([1.0, 0.0])], consent_note=CONSENT)
    store.save()

    assert cmd_forget(parse("--store", str(tmp_path), "forget", "Dan")) == 0
    assert LocalStore.load(tmp_path).find_by_name("Dan") is None
    assert "removed Dan" in capsys.readouterr().out


def test_forgetting_a_stranger_exits_nonzero_without_raising(tmp_path, capsys):
    assert cmd_forget(parse("--store", str(tmp_path), "forget", "Nobody")) == 1
    assert "no such person" in capsys.readouterr().err


def test_main_dispatches(tmp_path, capsys):
    assert main(["--store", str(tmp_path), "people"]) == 0
    assert "nobody enrolled" in capsys.readouterr().out
