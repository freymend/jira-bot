from __future__ import annotations

import sqlite3


def migrate(connection: sqlite3.Connection) -> None:
    statements = (
        """CREATE TABLE IF NOT EXISTS meetings (
            bot_id TEXT PRIMARY KEY,
            meeting_url TEXT,
            created_at TEXT NOT NULL,
            transcription_state TEXT NOT NULL DEFAULT 'not_started',
            status TEXT NOT NULL DEFAULT 'waiting',
            transcript_json TEXT,
            error TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS webhook_events (
            event_id TEXT PRIMARY KEY,
            bot_id TEXT NOT NULL,
            trigger TEXT NOT NULL,
            received_at TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS transcript_segments (
            bot_id TEXT NOT NULL,
            segment_key TEXT NOT NULL,
            timestamp_ms INTEGER,
            speaker_name TEXT,
            speaker_uuid TEXT,
            text TEXT NOT NULL,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (bot_id, segment_key)
        )""",
        """CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_id TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            summary TEXT NOT NULL,
            description TEXT NOT NULL,
            owner_speaker TEXT,
            due_date TEXT,
            priority TEXT,
            evidence_quote TEXT NOT NULL,
            timestamp_ms INTEGER,
            confidence REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            jira_key TEXT,
            error TEXT,
            UNIQUE (bot_id, fingerprint)
        )""",
        """CREATE TABLE IF NOT EXISTS background_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            bot_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            attempts INTEGER NOT NULL DEFAULT 0,
            available_at REAL NOT NULL,
            lease_until REAL,
            locked_by TEXT,
            last_error TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE (kind, bot_id)
        )""",
        """CREATE INDEX IF NOT EXISTS idx_background_jobs_ready
            ON background_jobs(status, available_at, lease_until)""",
        """CREATE TABLE IF NOT EXISTS worker_heartbeats (
            worker_id TEXT PRIMARY KEY,
            heartbeat_at REAL NOT NULL
        )""",
        "CREATE INDEX IF NOT EXISTS idx_tasks_bot_id ON tasks(bot_id)",
    )
    for statement in statements:
        connection.execute(statement)
