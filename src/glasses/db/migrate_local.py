"""Move a local gallery and captured media to Tiger, verifying before cleanup."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from ..store import LocalStore


def migrate_local(connection, gallery_path, media_path=None, *, remove_local=False):
    if not connection.autocommit or connection.info.transaction_status != 0:
        raise ValueError("Migration requires an idle autocommit connection before local cleanup")
    gallery_path = Path(gallery_path)
    media_path = Path(media_path) if media_path else None
    roots = [gallery_path] + ([media_path] if media_path else [])
    if any(root.is_symlink() or any(p.is_symlink() for p in root.parents) for root in roots):
        raise ValueError("Refusing to migrate symlinked data directories")
    store = LocalStore.load(gallery_path)
    paths = [gallery_path / name for name in (LocalStore.INDEX, LocalStore.VECTORS)]
    if media_path and media_path.exists():
        paths += sorted(p for p in media_path.rglob("*") if p.is_file())
    paths = [p for p in paths if p.is_file()]
    if any(p.is_symlink() for p in paths):
        raise ValueError("Refusing to migrate symlinked data files")
    files = {str(p.resolve()): p.read_bytes() for p in paths}
    if set(store._vectors) != {t.id for t in store.templates}:
        raise ValueError("Gallery metadata and embedding IDs do not match")
    for t in store.templates:
        vector = store._vectors[t.id]
        if t.person_id not in store.people or t.kind not in {"face", "voice"}:
            raise ValueError("Invalid template owner or kind")
        if vector.shape != (512,) or not np.isfinite(vector).all() or not np.any(vector):
            raise ValueError("Invalid embedding: expected 512 finite numbers")

    def uid(value):
        return str(uuid.UUID(value))

    with connection.transaction():
        has_wearer = any(t.kind == "voice" and t.is_wearer for t in store.templates)
        if has_wearer:
            connection.execute("SELECT pg_advisory_xact_lock(674521903)")
        # Preserve original IDs, consent, timestamps, model metadata and vectors.
        for person in store.people.values():
            values = (
                uid(person.id),
                person.display_name,
                person.consented_at,
                person.consent_note,
                person.created_at,
            )
            connection.execute(
                "INSERT INTO person (id, display_name, consented_at, consent_note, created_at) "
                "VALUES (%s,%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING",
                values,
            )
            row = connection.execute(
                "SELECT display_name, consented_at, consent_note, created_at, deleted_at "
                "FROM person WHERE id=%s",
                (uid(person.id),),
            ).fetchone()
            expected = (
                person.display_name,
                datetime.fromisoformat(person.consented_at),
                person.consent_note,
                datetime.fromisoformat(person.created_at),
                None,
            )
            if row != expected:
                raise ValueError("Person ID collision or metadata mismatch; local data kept")
        for t in store.templates:
            table = "face_template" if t.kind == "face" else "voice_template"
            vector = "[" + ",".join(repr(float(v)) for v in store._vectors[t.id]) + "]"
            columns = (
                "id, person_id, embedding, model_name, model_version, "
                "quality_score, source, created_at"
            )
            values = [
                uid(t.id),
                uid(t.person_id),
                vector,
                t.model_name,
                t.model_version,
                t.quality_score,
                t.source,
                t.created_at,
            ]
            if t.kind == "voice":
                columns += ", is_wearer"
                values.append(t.is_wearer)
            placeholders = ",".join(["%s"] * len(values))
            connection.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) "
                "ON CONFLICT (id) DO NOTHING",
                values,
            )
            row = connection.execute(
                f"SELECT person_id::text, embedding::text, model_name, model_version, "
                f"quality_score, source, created_at, deleted_at FROM {table} WHERE id=%s",
                (uid(t.id),),
            ).fetchone()
            restored = np.array(row[1].strip("[]").split(","), dtype=np.float32)
            if (
                row[0] != uid(t.person_id)
                or not np.array_equal(restored, store._vectors[t.id])
                or row[2:4] != (t.model_name, t.model_version)
                or not np.isclose(row[4], t.quality_score)
                or row[5] != t.source
                or row[6] != datetime.fromisoformat(t.created_at)
                or row[7] is not None
            ):
                raise ValueError("Template verification failed; local data kept")
            if t.kind == "voice":
                flag = connection.execute(
                    "SELECT is_wearer FROM voice_template WHERE id=%s", (uid(t.id),)
                ).fetchone()[0]
                if flag != t.is_wearer:
                    raise ValueError("Wearer metadata mismatch; local data kept")
        if has_wearer:
            wearers = connection.execute(
                "SELECT count(DISTINCT person_id) FROM voice_template "
                "WHERE is_wearer AND deleted_at IS NULL"
            ).fetchone()[0]
            if wearers > 1:
                raise ValueError("A different wearer is already enrolled; local data kept")
        person_ids = sorted(uid(person_id) for person_id in store.people)
        for path, content in files.items():
            digest = hashlib.sha256(content).hexdigest()
            connection.execute(
                "INSERT INTO local_import_file (source_path, sha256, content, person_ids) "
                "VALUES (%s,%s,%s,%s::uuid[]) "
                "ON CONFLICT (source_path) DO NOTHING",
                (path, digest, content, person_ids),
            )
            row = connection.execute(
                "SELECT sha256, content, person_ids FROM local_import_file WHERE source_path=%s",
                (path,),
            ).fetchone()
            if (
                row[0] != digest
                or bytes(row[1]) != content
                or sorted(str(value) for value in row[2]) != person_ids
            ):
                raise ValueError("Archive verification failed; local data kept")

    # The transaction above has committed. Check all disk files again before deleting any.
    if remove_local:
        for path, content in files.items():
            if Path(path).read_bytes() != content:
                raise ValueError("Local data changed during migration; local files kept")
        for path in files:
            Path(path).unlink()
        directories = {p.parent for p in paths}
        for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
            while directory in (gallery_path, media_path) or (
                media_path and media_path in directory.parents
            ):
                try:
                    directory.rmdir()
                except OSError:
                    break
                directory = directory.parent
    return {"people": len(store.people), "templates": len(store.templates), "files": len(files)}
