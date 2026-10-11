from __future__ import annotations

import sqlite3

from .helpers import add_column_if_missing


def migrate(connection: sqlite3.Connection) -> None:
    add_column_if_missing(
        connection,
        "meetings",
        "participant_tracking_enabled",
        "INTEGER NOT NULL DEFAULT 0",
    )
    add_column_if_missing(
        connection,
        "meetings",
        "participant_presence_initialized",
        "INTEGER NOT NULL DEFAULT 0",
    )
    add_column_if_missing(connection, "meetings", "last_participant_event_at", "TEXT")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS meeting_participant_events (
            bot_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (bot_id, event_id)
        )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS meeting_participants (
            bot_id TEXT NOT NULL,
            participant_uuid TEXT NOT NULL,
            participant_name TEXT,
            is_host INTEGER,
            is_present INTEGER NOT NULL,
            last_event_type TEXT NOT NULL,
            last_event_timestamp_ms INTEGER,
            last_event_received_at TEXT NOT NULL,
            PRIMARY KEY (bot_id, participant_uuid)
        )"""
    )
    connection.execute(
        """CREATE INDEX IF NOT EXISTS idx_meeting_participants_presence
           ON meeting_participants(bot_id, is_present)"""
    )
