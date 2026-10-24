from unittest.mock import Mock

import pytest

from glasses.storage import open_store
from glasses.store import LocalStore


def test_explicit_local_store_overrides_cloud(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    assert isinstance(open_store(tmp_path), LocalStore)


def test_unconfigured_store_remains_offline(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert isinstance(open_store(), LocalStore)


def test_configured_store_uses_cloud_and_never_falls_back(monkeypatch):
    from glasses.db.cli_store import PgCliStore

    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    sentinel = Mock()
    monkeypatch.setattr(PgCliStore, "open", lambda: sentinel)
    assert open_store() is sentinel
    monkeypatch.setattr(PgCliStore, "open", Mock(side_effect=RuntimeError("unavailable")))
    with pytest.raises(RuntimeError, match="unavailable"):
        open_store()
