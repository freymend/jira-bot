from __future__ import annotations

import sqlite3


def add_column_if_missing(
    connection: sqlite3.Connection, table: str, column: str, declaration: str
) -> None:
    columns = {
        row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
    }
    if column not in columns:
        connection.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
        )
