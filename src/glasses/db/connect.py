"""Getting a connection, and applying ``schema.sql`` to it.

Everything here imports ``psycopg`` lazily. The default test suite runs on
numpy alone and must not need a driver, let alone a database, so nothing at
module scope may pull one in.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from . import schema_sql

#: Where the connection string comes from. One variable, the conventional name,
#: so the Postgres-backed tests can key off its presence and skip without it.
ENV_VAR = "DATABASE_URL"

#: What the README's docker command produces. Used only as the fallback in
#: error messages — never as a silent default, because connecting to a
#: different database than you meant is worse than failing to connect.
LOCAL_URL = "postgresql://postgres:glasses@127.0.0.1:55432/glasses"

REQUIRED_EXTENSIONS = ("timescaledb", "vector")

_DOLLAR_TAG = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*\$|\$\$")


class DatabaseUnavailable(RuntimeError):
    """No usable database. Carries the instructions, because the fix is always
    the same two commands and printing them beats describing them."""


def database_url(explicit: str | None = None) -> str | None:
    return explicit or os.environ.get(ENV_VAR) or None


def require_url(explicit: str | None = None) -> str:
    url = database_url(explicit)
    if not url:
        raise DatabaseUnavailable(
            f"No database. Set {ENV_VAR}, or start the local one:\n"
            "    docker run -d --name glasses-tiger -p 55432:5432 \\\n"
            "        -e POSTGRES_PASSWORD=glasses -e POSTGRES_DB=glasses \\\n"
            "        timescale/timescaledb-ha:pg17\n"
            f"    export {ENV_VAR}={LOCAL_URL}"
        )
    return url


def connect(url: str | None = None, *, autocommit: bool = False) -> Any:
    """A psycopg connection, with the import done here rather than at module
    scope."""
    try:
        import psycopg
    except ImportError as error:  # pragma: no cover - depends on the extra
        raise DatabaseUnavailable(
            "This command needs the Postgres driver. Install it with:\n"
            "    uv pip install -e '.[db]'"
        ) from error

    return psycopg.connect(require_url(url), autocommit=autocommit)


def split_statements(sql: str) -> list[str]:
    """Split a SQL script into statements, ignoring semicolons in comments,
    string literals and dollar-quoted blocks.

    Needed because the schema cannot be sent as one string. psycopg hands a
    multi-statement string to the simple query protocol, which wraps it in an
    implicit transaction, and ``CREATE MATERIALIZED VIEW ... WITH
    (timescaledb.continuous)`` refuses to run inside a transaction block. So
    the statements go one at a time, on an autocommit connection.
    """
    statements: list[str] = []
    current: list[str] = []
    index, end = 0, len(sql)

    while index < end:
        char = sql[index]

        if sql.startswith("--", index):
            newline = sql.find("\n", index)
            index = end if newline == -1 else newline
            continue

        if char == "'":
            cursor = index + 1
            while cursor < end:
                if sql[cursor] == "'":
                    if sql.startswith("''", cursor):  # an escaped quote
                        cursor += 2
                        continue
                    break
                cursor += 1
            current.append(sql[index : cursor + 1])
            index = cursor + 1
            continue

        if char == "$":
            tag = _DOLLAR_TAG.match(sql, index)
            if tag:
                closing = sql.find(tag.group(0), tag.end())
                stop = end if closing == -1 else closing + len(tag.group(0))
                current.append(sql[index:stop])
                index = stop
                continue

        if char == ";":
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            index += 1
            continue

        current.append(char)
        index += 1

    trailing = "".join(current).strip()
    if trailing:
        statements.append(trailing)
    return statements


def installed_extensions(connection: Any) -> dict[str, str]:
    with connection.cursor() as cursor:
        cursor.execute("SELECT extname, extversion FROM pg_extension")
        return dict(cursor.fetchall())


def available_extensions(connection: Any) -> set[str]:
    with connection.cursor() as cursor:
        cursor.execute("SELECT name FROM pg_available_extensions")
        return {row[0] for row in cursor.fetchall()}


def is_initialised(connection: Any) -> bool:
    with connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass('face_template') IS NOT NULL")
        return bool(cursor.fetchone()[0])


def apply_schema(connection: Any, *, sql: str | None = None) -> int:
    """Run ``schema.sql``, one statement at a time. Returns the count.

    The connection must be in autocommit: see ``split_statements``.
    """
    if not connection.autocommit:
        raise ValueError(
            "apply_schema needs an autocommit connection: a continuous "
            "aggregate cannot be created inside a transaction block"
        )

    available = available_extensions(connection)
    missing = [name for name in REQUIRED_EXTENSIONS if name not in available]
    if missing:
        raise DatabaseUnavailable(
            f"This Postgres has no {', '.join(missing)}. The schema needs "
            "TimescaleDB and pgvector in the same instance — that is the "
            "point of it. Use the documented image:\n"
            "    docker run -d --name glasses-tiger -p 55432:5432 \\\n"
            "        -e POSTGRES_PASSWORD=glasses -e POSTGRES_DB=glasses \\\n"
            "        timescale/timescaledb-ha:pg17"
        )

    statements = split_statements(sql if sql is not None else schema_sql())
    with connection.cursor() as cursor:
        for statement in statements:
            cursor.execute(statement)  # type: ignore[arg-type]
    return len(statements)


def apply_cloud_migration(connection: Any) -> int:
    """Upgrade an existing gallery without dropping tables or saved people."""
    statements = split_statements(Path(__file__).with_name("cloud_migration.sql").read_text())
    with connection.transaction(), connection.cursor() as cursor:
        for statement in statements:
            cursor.execute(statement)
    return len(statements)


def drop_schema(connection: Any, schema: str = "public") -> None:
    """Throw the whole thing away. ``db init --reset`` and the test fixture.

    Anything installed *into* that schema goes with it, extensions included,
    which is why the caller should reconnect before applying the schema again:
    TimescaleDB does not tolerate being dropped and recreated in one session.
    ``schema.sql`` recreates the extensions it needs, so one fresh connection
    is all the recovery required.
    """
    with connection.cursor() as cursor:
        cursor.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        cursor.execute(f'CREATE SCHEMA "{schema}"')


__all__ = [
    "ENV_VAR",
    "LOCAL_URL",
    "REQUIRED_EXTENSIONS",
    "DatabaseUnavailable",
    "apply_schema",
    "apply_cloud_migration",
    "available_extensions",
    "connect",
    "database_url",
    "drop_schema",
    "installed_extensions",
    "is_initialised",
    "require_url",
    "split_statements",
]
