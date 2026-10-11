from __future__ import annotations

import sqlite3

from .helpers import add_column_if_missing


def migrate(connection: sqlite3.Connection) -> None:
    add_column_if_missing(connection, "tasks", "owner_evidence_quote", "TEXT")
    add_column_if_missing(connection, "tasks", "owner_timestamp_ms", "INTEGER")
    add_column_if_missing(
        connection,
        "tasks",
        "context_evidence_json",
        "TEXT NOT NULL DEFAULT '[]'",
    )
