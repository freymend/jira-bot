from __future__ import annotations

import json
import time

from app.repositories.database import _connect, _now
from app.schemas import JSONObject, StatusRow


def record_webhook(
    event_id: str,
    bot_id: str,
    trigger: str,
    payload: JSONObject,
    *,
    enqueue_finalization: bool = False,
) -> bool:
    with _connect() as connection:
        result = connection.execute(
            "INSERT OR IGNORE INTO webhook_events VALUES (?, ?, ?, ?, ?)",
            (
                event_id,
                bot_id,
                trigger,
                _now(),
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        connection.execute(
            "INSERT OR IGNORE INTO meetings(bot_id, created_at) VALUES (?, ?)",
            (bot_id, _now()),
        )
        if result.rowcount == 1 and enqueue_finalization:
            meeting = connection.execute(
                "SELECT status FROM meetings WHERE bot_id=?", (bot_id,)
            ).fetchone()
            meeting_status: StatusRow | None = (
                {"status": meeting["status"]} if meeting is not None else None
            )
            if meeting_status is not None and meeting_status["status"] != "completed":
                now = time.time()
                connection.execute(
                    """INSERT OR IGNORE INTO background_jobs
                       (kind, bot_id, status, attempts, available_at, created_at, updated_at)
                       VALUES ('finalize_meeting', ?, 'queued', 0, ?, ?, ?)""",
                    (bot_id, now, now, now),
                )
        return result.rowcount == 1
