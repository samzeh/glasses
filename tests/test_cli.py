"""CLI wiring. The commands that need models are exercised by hand, not here —
these cover argument parsing and the paths that touch the store.
"""

from __future__ import annotations

import pytest

from glasses.cli import build_parser, cmd_forget, cmd_people, expand_images, main
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
    ):
        assert parse(*argv).command == argv[0]


def test_enroll_requires_a_consent_note():
    """Forgetting the flag has to be an error, not a default."""
    with pytest.raises(SystemExit):
        parse("enroll", "--name", "Dan", "a.jpg")


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
