"""Select the CLI gallery backend, with explicit local storage for offline use."""

from __future__ import annotations

import atexit
from pathlib import Path

from .db.connect import database_url
from .store import LocalStore


def open_store(path: Path | str | None = None):
    """Use Tiger when configured; never fall back after a connection failure."""
    if path is not None or not database_url():
        return LocalStore.load(path or "glasses-store")
    from .db.cli_store import PgCliStore

    store = PgCliStore.open()
    atexit.register(store.close)
    return store


def store_label(path: Path | str | None = None) -> str:
    return str(path) if path is not None else ("TigerData" if database_url() else "glasses-store")
