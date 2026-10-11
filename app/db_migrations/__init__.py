from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime

from .v001_initial_schema import migrate as _create_initial_schema
from .v002_meeting_bot_state import migrate as _add_meeting_bot_state
from .v003_task_evidence_columns import migrate as _add_task_evidence_columns
from .v004_backfill_finalization_jobs import migrate as _backfill_finalization_jobs
from .v005_task_extraction_cache import migrate as _create_task_extraction_cache


def _now() -> str:
    return datetime.now(UTC).isoformat()


Migration = tuple[int, Callable[[sqlite3.Connection], None]]
MIGRATIONS: tuple[Migration, ...] = (
    (1, _create_initial_schema),
    (2, _add_meeting_bot_state),
    (3, _add_task_evidence_columns),
    (4, _backfill_finalization_jobs),
    (5, _create_task_extraction_cache),
)


def apply_migrations(connection: sqlite3.Connection) -> None:
    connection.execute("BEGIN IMMEDIATE")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )"""
    )
    applied_versions = {
        row["version"]
        for row in connection.execute("SELECT version FROM schema_migrations")
    }
    supported_versions = {version for version, _ in MIGRATIONS}
    unknown_versions = applied_versions - supported_versions
    if unknown_versions:
        raise RuntimeError(
            "Database contains unsupported migration versions: "
            + ", ".join(str(version) for version in sorted(unknown_versions))
        )

    for version, migrate in sorted(MIGRATIONS):
        if version in applied_versions:
            continue
        migrate(connection)
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
            (version, _now()),
        )
