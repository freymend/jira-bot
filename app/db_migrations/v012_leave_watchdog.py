from __future__ import annotations

import sqlite3

from .helpers import add_column_if_missing


def migrate(connection: sqlite3.Connection) -> None:
    add_column_if_missing(connection, "meetings", "leave_watchdog_checked_at", "TEXT")
    add_column_if_missing(connection, "meetings", "leave_stuck_at", "TEXT")
    connection.execute(
        """CREATE INDEX IF NOT EXISTS idx_meetings_leave_watchdog
           ON meetings(leave_status, leave_requested_at, leave_watchdog_checked_at)"""
    )
