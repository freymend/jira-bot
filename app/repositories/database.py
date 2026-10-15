from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from app.config import settings
from app.db_migrations import apply_migrations

type SQLiteParameter = str | int | float | None


def _now() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def _connect() -> Generator[sqlite3.Connection, None, None]:
    path = Path(settings.database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA secure_delete=ON")
        with connection:
            yield connection
    finally:
        connection.close()


def init_db() -> None:
    with _connect() as connection:
        apply_migrations(connection)
