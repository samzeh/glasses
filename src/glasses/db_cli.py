"""``glasses db`` — apply the schema, and look at what Tiger is holding.

Separate from the rest of the CLI in the same way ``voice_cli`` and
``meet_cli`` are, and for a harder reason: nothing here may be imported at
module scope by anything the default path touches, because the driver is an
extra and the test suite runs without it.
"""

from __future__ import annotations

import argparse
import sys

from .db.connect import (
    ENV_VAR,
    LOCAL_URL,
    DatabaseUnavailable,
    apply_cloud_migration,
    apply_schema,
    connect,
    drop_schema,
    installed_extensions,
    is_initialised,
)

DOCKER_HINT = f"""Start one, from the README:

    docker run -d --name glasses-tiger -p 55432:5432 \\
        -e POSTGRES_PASSWORD=glasses -e POSTGRES_DB=glasses \\
        timescale/timescaledb-ha:pg17
    export {ENV_VAR}={LOCAL_URL}"""


def cmd_db_init(args: argparse.Namespace) -> int:
    """Apply ``schema.sql``: tables, hypertables, aggregates, policies."""
    try:
        connection = connect(args.url, autocommit=True)
    except DatabaseUnavailable as error:
        print(f"{error}\n", file=sys.stderr)
        return 1

    with connection:
        if args.reset:
            print("dropping the public schema")
            drop_schema(connection)

    # A fresh connection, because dropping the schema took the extensions
    # with it and TimescaleDB will not be recreated in the session that
    # dropped it.
    try:
        connection = connect(args.url, autocommit=True)
    except DatabaseUnavailable as error:  # pragma: no cover - unreachable above
        print(f"{error}\n", file=sys.stderr)
        return 1

    with connection:
        if is_initialised(connection) and not args.reset:
            apply_cloud_migration(connection)
            print(
                "Already initialised — applied cloud enrollment upgrades without resetting data.\n"
                "`glasses db init --reset` drops everything and starts over."
            )
            return 0
        try:
            count = apply_schema(connection)
        except DatabaseUnavailable as error:
            print(f"{error}\n", file=sys.stderr)
            return 1

        versions = installed_extensions(connection)
        print(f"applied {count} statements")
        for name in ("timescaledb", "vector", "pgcrypto"):
            print(f"  {name:14s} {versions.get(name, 'MISSING')}")
        print(_summary(connection))
    return 0


def _summary(connection) -> str:  # type: ignore[no-untyped-def]
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT hypertable_name, num_chunks, compression_enabled
            FROM timescaledb_information.hypertables ORDER BY hypertable_name
            """
        )
        hypertables = cursor.fetchall()
        cursor.execute(
            """
            SELECT view_name, materialized_only
            FROM timescaledb_information.continuous_aggregates ORDER BY view_name
            """
        )
        aggregates = cursor.fetchall()

    lines = ["\nhypertables"]
    lines += [
        f"  {name:20s} {chunks} chunk(s)  compression {'on' if compressed else 'off'}"
        for name, chunks, compressed in hypertables
    ]
    lines.append("continuous aggregates")
    lines += [f"  {name:28s} real-time {'off' if only else 'on'}" for name, only in aggregates]
    return "\n".join(lines)


def cmd_db_stats(args: argparse.Namespace) -> int:
    """What is in there, what it costs, and what the live view says.

    Doubles as the demo: the same database answers "who is enrolled" from the
    relational tables and "what did the sensors see this minute" from a
    continuous aggregate over the hypertables.
    """
    try:
        connection = connect(args.url, autocommit=True)
    except DatabaseUnavailable as error:
        print(f"{error}\n{DOCKER_HINT}", file=sys.stderr)
        return 1

    with connection, connection.cursor() as cursor:
        if not is_initialised(connection):
            print("Schema not applied yet — run `glasses db init`.", file=sys.stderr)
            return 1

        cursor.execute(
            """
            SELECT (SELECT count(*) FROM person WHERE deleted_at IS NULL),
                   (SELECT count(*) FROM face_template WHERE deleted_at IS NULL),
                   (SELECT count(*) FROM voice_template WHERE deleted_at IS NULL),
                   (SELECT count(*) FROM encounter),
                   (SELECT count(*) FROM face_observation),
                   (SELECT count(*) FROM speech_observation)
            """
        )
        people, faces, voices, encounters, face_obs, speech_obs = cursor.fetchone()
        print("profile (relational)")
        print(f"  people            {people}")
        print(f"  face templates    {faces}")
        print(f"  voice templates   {voices}")
        print(f"  encounters        {encounters}")
        print("\nstreams (hypertables)")
        print(f"  face_observation    {face_obs} rows")
        print(f"  speech_observation  {speech_obs} rows")

        # The stats function reports the hypertable it was asked about only
        # as the argument, so the name is carried alongside.
        rows = []
        for table in ("face_observation", "speech_observation"):
            cursor.execute(
                """
                SELECT %s,
                       pg_size_pretty(before_compression_total_bytes),
                       pg_size_pretty(after_compression_total_bytes),
                       number_compressed_chunks
                FROM hypertable_compression_stats(%s)
                """,
                (table, table),
            )
            rows.extend(cursor.fetchall())
        print("\ncompression")
        if not any(row[3] for row in rows):
            print("  no compressed chunks yet — nothing is 7 days old. Force it with:")
            print("    SELECT compress_chunk(c) FROM show_chunks('face_observation') c;")
        for name, before, after, chunks in rows:
            if chunks:
                print(f"  {name:20s} {chunks} chunk(s)  {before} -> {after}")

        cursor.execute(
            """
            SELECT bucket, encounter_id::text, face_observations, distinct_faces,
                   round(face_visible_s::numeric, 1), speech_segments, distinct_voices,
                   round(partner_speech_s::numeric, 1), round(wearer_speech_s::numeric, 1)
            FROM encounter_minute
            ORDER BY bucket DESC
            LIMIT %s
            """,
            (args.minutes,),
        )
        minutes = cursor.fetchall()
        print(f"\nencounter_minute — the live view, last {args.minutes}")
        if not minutes:
            print("  nothing observed yet")
        else:
            print(
                f"  {'minute':17s} {'encounter':10s} {'faces':>6s} {'distinct':>9s} "
                f"{'seen s':>7s} {'speech':>7s} {'voices':>7s} {'partner s':>10s} "
                f"{'wearer s':>9s}"
            )
        for bucket, encounter, obs, distinct, seen, segments, voices_, partner, wearer in minutes:
            # Both values are sliced before the f-string: a slice inside a
            # replacement field is read as the start of the format spec.
            when = bucket.strftime("%Y-%m-%d %H:%M")
            short = encounter[:8]
            # The sums come back as Decimal, which has no integer
            # presentation, so the widths are applied to plain ints.
            print(
                f"  {when:17s} {short:10s} {int(obs):6d} {int(distinct):9d} "
                f"{float(seen):7.1f} {int(segments):7d} {int(voices_):7d} "
                f"{float(partner):10.1f} {float(wearer):9.1f}"
            )
    return 0


def register(subparsers) -> None:  # type: ignore[no-untyped-def]
    db = subparsers.add_parser("db", help="apply the schema, and inspect Tiger")
    actions = db.add_subparsers(dest="db_command", required=True)
    migrate = actions.add_parser("migrate-local", help="verify and move local data to TigerData")
    migrate.add_argument("--local-store", default="glasses-store")
    migrate.add_argument("--media-dir", default="data")
    migrate.add_argument("--remove-local", action="store_true")
    migrate.set_defaults(func=cmd_migrate_local)
    url_help = f"connection string (default: ${ENV_VAR})"

    init = actions.add_parser("init", help="apply schema.sql to the database")
    init.add_argument("--url", default=None, help=url_help)
    init.add_argument(
        "--reset",
        action="store_true",
        help="drop the public schema first — deletes every row",
    )
    init.set_defaults(func=cmd_db_init)

    stats = actions.add_parser("stats", help="rows, compression, and the live per-minute view")
    stats.add_argument("--url", default=None, help=url_help)
    stats.add_argument("--minutes", type=int, default=10, help="how many buckets to show")
    stats.set_defaults(func=cmd_db_stats)


def cmd_migrate_local(args: argparse.Namespace) -> int:
    from .db.migrate_local import migrate_local

    with connect(autocommit=True) as connection:
        if not is_initialised(connection):
            raise DatabaseUnavailable("Initialize the database first with `glasses db init`.")
        apply_cloud_migration(connection)
        result = migrate_local(
            connection, args.local_store, args.media_dir, remove_local=args.remove_local
        )
    print(
        f"Verified in TigerData: {result['people']} people, "
        f"{result['templates']} templates, {result['files']} archived files"
    )
    if args.remove_local:
        print("Removed the verified local gallery and media files.")
    return 0
