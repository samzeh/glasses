"""Schema and storage. The schema itself lives in ``schema.sql``."""

from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def schema_sql() -> str:
    return SCHEMA_PATH.read_text()


__all__ = ["SCHEMA_PATH", "schema_sql"]
