from __future__ import annotations

import sqlite3

from .helpers import add_column_if_missing


def migrate(connection: sqlite3.Connection) -> None:
    add_column_if_missing(connection, "meetings", "recording_started_at_ms", "INTEGER")
    add_column_if_missing(connection, "meetings", "screen_frame_origin_ms", "INTEGER")
    add_column_if_missing(connection, "meetings", "screen_frame_origin_source", "TEXT")
    add_column_if_missing(
        connection,
        "meetings",
        "screenshare_capture_enabled",
        "INTEGER NOT NULL DEFAULT 0",
    )
    add_column_if_missing(
        connection,
        "tasks",
        "visual_evidence_json",
        "TEXT NOT NULL DEFAULT '[]'",
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS screen_frames (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id TEXT NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            received_at_ms INTEGER NOT NULL,
            time_source TEXT NOT NULL,
            participant_uuid TEXT,
            frame_data BLOB NOT NULL,
            summary TEXT,
            created_at REAL NOT NULL
        )"""
    )
    connection.execute(
        """CREATE INDEX IF NOT EXISTS idx_screen_frames_meeting_time
           ON screen_frames(bot_id, timestamp_ms)"""
    )
    connection.execute(
        """CREATE INDEX IF NOT EXISTS idx_screen_frames_retention
           ON screen_frames(created_at)"""
    )
