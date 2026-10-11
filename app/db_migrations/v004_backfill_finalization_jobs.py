from __future__ import annotations

import sqlite3
import time


def migrate(connection: sqlite3.Connection) -> None:
    now = time.time()
    connection.execute(
        """INSERT OR IGNORE INTO background_jobs
           (kind, bot_id, status, attempts, available_at, created_at, updated_at)
           SELECT 'finalize_meeting', bot_id, 'queued', 0, ?, ?, ?
           FROM meetings WHERE status='processing'""",
        (now, now, now),
    )
