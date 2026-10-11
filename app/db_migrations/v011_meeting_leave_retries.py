from __future__ import annotations

import sqlite3

from .helpers import add_column_if_missing


def migrate(connection: sqlite3.Connection) -> None:
    add_column_if_missing(
        connection, "meetings", "leave_attempt_count", "INTEGER NOT NULL DEFAULT 0"
    )
