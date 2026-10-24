"""Read a ``.env`` file into the environment at startup.

``build_transcriber`` picks Cartesia when it can see ``CARTESIA_API_KEY``
and the local Whisper when it cannot. Nothing was reading ``.env``, so a
fresh shell could not see a key that was sitting in the repo, chose the
local recogniser, and said nothing about it. The failure is quiet and the
transcript still arrives, which is what makes it worth fixing: you only
notice by reading the recogniser line.

A real environment variable always wins. ``.env`` is the fallback for a
shell nobody exported into, never an override of one that was.
"""

from __future__ import annotations

import os
from pathlib import Path

FILENAME = ".env"

#: Quote characters a value may be wrapped in, as a shell would write it.
_QUOTES = "\"'"


def find_env_file(start: Path | str | None = None) -> Path | None:
    """The nearest ``.env`` at or above ``start``, or None.

    Walks upwards so the command works from a subdirectory of the repo,
    which is where it is usually run from.
    """
    here = Path(start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / FILENAME
        if candidate.is_file():
            return candidate
    return None


def parse(text: str) -> dict[str, str]:
    """``KEY=value`` lines to a mapping.

    Deliberately small: comments, blank lines, a leading ``export``, and
    one layer of matching quotes. Anything fancier belongs to a real
    dotenv library, and this file exists precisely so there is no new
    dependency to install before a demo.
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in _QUOTES:
            value = value[1:-1]
        values[key] = value
    return values


def load(start: Path | str | None = None) -> list[str]:
    """Put a ``.env``'s values into ``os.environ``, returning what it set.

    Never overwrites a variable that is already set, and never raises: a
    missing or unreadable ``.env`` is the normal case, not an error.
    """
    path = find_env_file(start)
    if path is None:
        return []
    try:
        text = path.read_text()
    except OSError:
        return []

    loaded: list[str] = []
    for key, value in parse(text).items():
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


__all__ = ["FILENAME", "find_env_file", "load", "parse"]
