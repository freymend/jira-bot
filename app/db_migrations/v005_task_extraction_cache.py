from __future__ import annotations

import sqlite3


def migrate(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE IF NOT EXISTS task_extraction_cache (
            bot_id TEXT NOT NULL,
            transcript_hash TEXT NOT NULL,
            stage TEXT NOT NULL,
            item_key TEXT NOT NULL,
            result_json TEXT NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (bot_id, transcript_hash, stage, item_key)
        )"""
    )
