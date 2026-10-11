from __future__ import annotations

import sqlite3


def migrate(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE IF NOT EXISTS calendar_auto_join_requests (
            event_key TEXT PRIMARY KEY,
            calendar_event_id TEXT NOT NULL,
            event_start TEXT NOT NULL,
            status TEXT NOT NULL,
            bot_id TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )"""
    )
    connection.execute(
        """CREATE INDEX IF NOT EXISTS idx_calendar_auto_join_created
           ON calendar_auto_join_requests(created_at DESC)"""
    )
