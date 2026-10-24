"""The .env loader. Small, but it decides which recogniser runs."""

from __future__ import annotations

import os

import pytest

from glasses import env


def test_parses_plain_assignments():
    assert env.parse("A=1\nB=two") == {"A": "1", "B": "two"}


def test_skips_comments_and_blank_lines():
    assert env.parse("# a note\n\nA=1\n   \n# another\nB=2") == {"A": "1", "B": "2"}


def test_strips_a_leading_export():
    assert env.parse("export CARTESIA_API_KEY=sk-live") == {"CARTESIA_API_KEY": "sk-live"}


@pytest.mark.parametrize("quote", ['"', "'"])
def test_strips_one_layer_of_matching_quotes(quote):
    assert env.parse(f"A={quote}hello world{quote}") == {"A": "hello world"}


def test_leaves_mismatched_quotes_alone():
    assert env.parse("A=\"unclosed") == {"A": '"unclosed'}


def test_keeps_equals_signs_inside_a_value():
    # Base64-ish keys end in '=' padding, and losing it silently breaks auth.
    assert env.parse("K=abc==") == {"K": "abc=="}


def test_ignores_a_line_with_no_equals():
    assert env.parse("NOT_AN_ASSIGNMENT\nA=1") == {"A": "1"}


def test_finds_the_file_from_a_subdirectory(tmp_path):
    (tmp_path / ".env").write_text("A=1")
    nested = tmp_path / "one" / "two"
    nested.mkdir(parents=True)
    assert env.find_env_file(nested) == tmp_path / ".env"


def test_finds_nothing_when_there_is_no_file(tmp_path):
    assert env.find_env_file(tmp_path) is None


def test_load_sets_values_and_reports_them(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("GLASSES_TEST_KEY=from-file")
    monkeypatch.delenv("GLASSES_TEST_KEY", raising=False)

    assert env.load(tmp_path) == ["GLASSES_TEST_KEY"]
    assert os.environ["GLASSES_TEST_KEY"] == "from-file"


def test_load_never_overrides_a_real_environment_variable(tmp_path, monkeypatch):
    """A shell that exported a key wins over a stale .env.

    Someone switching keys does it in the shell and expects that to take.
    """
    (tmp_path / ".env").write_text("GLASSES_TEST_KEY=from-file")
    monkeypatch.setenv("GLASSES_TEST_KEY", "from-shell")

    assert env.load(tmp_path) == []
    assert os.environ["GLASSES_TEST_KEY"] == "from-shell"


def test_load_is_quiet_when_there_is_no_file(tmp_path):
    assert env.load(tmp_path) == []


def test_load_is_quiet_when_the_file_cannot_be_read(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("A=1")
    monkeypatch.setattr(
        env.Path, "read_text", lambda self, *a, **k: (_ for _ in ()).throw(OSError("nope"))
    )
    assert env.load(tmp_path) == []
