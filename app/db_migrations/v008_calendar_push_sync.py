from __future__ import annotations

import sqlite3


def migrate(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE IF NOT EXISTS calendar_sync_state (
            calendar_id TEXT PRIMARY KEY,
            sync_token TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS calendar_watch_channels (
            channel_id TEXT PRIMARY KEY,
            calendar_id TEXT NOT NULL,
            channel_token TEXT NOT NULL,
            resource_id TEXT,
            expiration_ms INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )"""
    )
    connection.execute(
        """CREATE INDEX IF NOT EXISTS idx_calendar_watch_expiration
           ON calendar_watch_channels(calendar_id, status, expiration_ms)"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS calendar_scheduled_events (
            calendar_id TEXT NOT NULL,
            event_key TEXT NOT NULL,
            calendar_event_id TEXT NOT NULL,
            event_start TEXT NOT NULL,
            meeting_url TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (calendar_id, event_key)
        )"""
    )
    connection.execute(
        """CREATE INDEX IF NOT EXISTS idx_calendar_scheduled_start
           ON calendar_scheduled_events(calendar_id, event_start)"""
    )
