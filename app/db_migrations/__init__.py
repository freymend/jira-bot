from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime

from .v001_initial_schema import migrate as _create_initial_schema
from .v002_meeting_bot_state import migrate as _add_meeting_bot_state
from .v003_task_evidence_columns import migrate as _add_task_evidence_columns
from .v004_backfill_finalization_jobs import migrate as _backfill_finalization_jobs
from .v005_task_extraction_cache import migrate as _create_task_extraction_cache
from .v006_calendar_auto_join import migrate as _create_calendar_auto_join_requests
from .v007_screenshare_evidence import migrate as _create_screenshare_evidence
from .v008_calendar_push_sync import migrate as _create_calendar_push_sync
from .v009_meeting_auto_leave import migrate as _add_meeting_auto_leave
from .v010_meeting_participant_presence import (
    migrate as _add_meeting_participant_presence,
)
from .v011_meeting_leave_retries import migrate as _add_meeting_leave_retries
from .v012_leave_watchdog import migrate as _add_leave_watchdog


def _now() -> str:
    return datetime.now(UTC).isoformat()


Migration = tuple[int, Callable[[sqlite3.Connection], None]]
MIGRATIONS: tuple[Migration, ...] = (
    (1, _create_initial_schema),
    (2, _add_meeting_bot_state),
    (3, _add_task_evidence_columns),
    (4, _backfill_finalization_jobs),
    (5, _create_task_extraction_cache),
    (6, _create_calendar_auto_join_requests),
    (7, _create_screenshare_evidence),
    (8, _create_calendar_push_sync),
    (9, _add_meeting_auto_leave),
    (10, _add_meeting_participant_presence),
    (11, _add_meeting_leave_retries),
    (12, _add_leave_watchdog),
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
