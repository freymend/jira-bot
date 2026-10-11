from __future__ import annotations

import sqlite3

from .helpers import add_column_if_missing


def migrate(connection: sqlite3.Connection) -> None:
    add_column_if_missing(connection, "meetings", "scheduled_end_at", "TEXT")
    add_column_if_missing(connection, "meetings", "last_transcript_received_at", "TEXT")
    add_column_if_missing(connection, "meetings", "leave_evaluated_activity_at", "TEXT")
    add_column_if_missing(connection, "meetings", "leave_evaluation_started_at", "TEXT")
    add_column_if_missing(
        connection, "meetings", "leave_status", "TEXT NOT NULL DEFAULT 'not_requested'"
    )
    add_column_if_missing(connection, "meetings", "leave_requested_at", "TEXT")
    add_column_if_missing(connection, "meetings", "leave_reason", "TEXT")
    add_column_if_missing(connection, "calendar_scheduled_events", "event_end", "TEXT")
    connection.execute(
        """CREATE INDEX IF NOT EXISTS idx_meetings_leave_monitor
           ON meetings(bot_state, leave_status, scheduled_end_at)"""
    )
