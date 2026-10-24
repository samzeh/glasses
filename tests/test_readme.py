"""The README is the first thing anyone runs. A command documented there that
does not parse is worse than no documentation, so the two are checked against
each other rather than kept in sync by hand.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from glasses.cli import build_parser

README = Path(__file__).resolve().parents[1] / "README.md"

#: Fenced ```sh blocks only. Other fenced blocks hold the source tree and
#: python, where the word "glasses" means the package, not the command.
SH_BLOCK = re.compile(r"```sh\n(.*?)```", re.DOTALL)
#: `glasses …` inside an inline code span, which is how the command table and
#: the prose refer to commands.
INLINE = re.compile(r"`(glasses [^`]+)`")


def documented_invocations() -> list[list[str]]:
    text = README.read_text()
    lines = [
        line.split("#")[0].strip()
        for block in SH_BLOCK.findall(text)
        for line in block.splitlines()
        if line.strip().startswith("glasses ")
    ]
    lines += INLINE.findall(text)

    found: list[list[str]] = []
    for line in lines:
        # Ellipsis stands for a value the reader supplies; it still has to
        # occupy the slot, or a flag that takes an argument looks optional.
        argv = shlex.split(line.replace("…", "PLACEHOLDER"))[1:]
        if argv and not argv[0].startswith("-"):
            found.append(argv)
    return found


def test_the_readme_actually_shows_commands():
    """Guard against the regex silently matching nothing."""
    assert len(documented_invocations()) >= 8


@pytest.mark.parametrize("argv", documented_invocations(), ids=lambda a: " ".join(a))
def test_every_documented_command_parses(argv):
    parser = build_parser()
    # Placeholder values stand in for real paths and names; what is under test
    # is the subcommand and its flags, not the arguments' content.
    parser.parse_args(argv)


def test_the_quick_start_is_the_minimum_real_path():
    """check, enroll and watch, in that order, early in the file — the three
    commands that take someone from nothing to a working recognition."""
    head = README.read_text().split("## What you are doing")[0]

    assert "glasses check" in head
    assert "--camera" in head
    assert "glasses watch" in head
    assert head.index("glasses check") < head.index("glasses watch")


def test_the_consent_flag_is_never_shown_without_a_value():
    """Documenting `enroll` without --consent would teach the wrong habit."""
    for argv in documented_invocations():
        if argv[0] == "enroll":
            assert "--consent" in argv
