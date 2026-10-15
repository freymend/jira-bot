from __future__ import annotations

import sqlite3
import time

from app.repositories import database
from app.schemas import ScreenFrameRecord


def store_screen_frame(
    bot_id: str,
    frame_data: bytes,
    *,
    received_at_ms: int,
    participant_uuid: str | None,
) -> int | None:
    now = time.time()
    cutoff = now - database.settings.screenshare_retention_hours * 3600
    with database._connect() as connection:
        connection.execute("DELETE FROM screen_frames WHERE created_at < ?", (cutoff,))
        meeting = connection.execute(
            """SELECT recording_started_at_ms, screen_frame_origin_ms,
                      screenshare_capture_enabled
               FROM meetings WHERE bot_id=?""",
            (bot_id,),
        ).fetchone()
        if meeting is None or not meeting["screenshare_capture_enabled"]:
            return None
        last_frame = connection.execute(
            """SELECT received_at_ms FROM screen_frames
               WHERE bot_id=? ORDER BY received_at_ms DESC LIMIT 1""",
            (bot_id,),
        ).fetchone()
        if last_frame is not None and (
            received_at_ms - last_frame["received_at_ms"]
            < database.settings.screenshare_sample_interval_seconds * 1000
        ):
            return None
        frame_count = connection.execute(
            "SELECT COUNT(*) FROM screen_frames WHERE bot_id=?", (bot_id,)
        ).fetchone()[0]
        if frame_count >= database.settings.screenshare_max_frames_per_meeting:
            return None

        origin = meeting["recording_started_at_ms"] or meeting["screen_frame_origin_ms"]
        time_source = (
            "recording_state" if meeting["recording_started_at_ms"] else "first_frame"
        )
        if origin is None:
            origin = received_at_ms
            connection.execute(
                """UPDATE meetings
                   SET screen_frame_origin_ms=?, screen_frame_origin_source='first_frame'
                   WHERE bot_id=? AND screen_frame_origin_ms IS NULL""",
                (origin, bot_id),
            )
        timestamp_ms = max(0, received_at_ms - int(origin))
        cursor = connection.execute(
            """INSERT INTO screen_frames
               (bot_id, timestamp_ms, received_at_ms, time_source,
                participant_uuid, frame_data, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                bot_id,
                timestamp_ms,
                received_at_ms,
                time_source,
                participant_uuid,
                sqlite3.Binary(frame_data),
                now,
            ),
        )
        return cursor.lastrowid


def get_screen_frames_near(
    bot_id: str, timestamp_ms: int, *, window_ms: int, limit: int = 2
) -> list[ScreenFrameRecord]:
    with database._connect() as connection:
        rows = connection.execute(
            """SELECT id, bot_id, timestamp_ms, time_source, participant_uuid,
                      frame_data, summary
               FROM screen_frames
               WHERE bot_id=? AND timestamp_ms BETWEEN ? AND ?
               ORDER BY ABS(timestamp_ms - ?), timestamp_ms
               LIMIT ?""",
            (
                bot_id,
                max(0, timestamp_ms - window_ms),
                timestamp_ms + window_ms,
                timestamp_ms,
                limit,
            ),
        ).fetchall()
        result: list[ScreenFrameRecord] = []
        for row in rows:
            result.append(
                {
                    "id": row["id"],
                    "bot_id": row["bot_id"],
                    "timestamp_ms": row["timestamp_ms"],
                    "time_source": row["time_source"],
                    "participant_uuid": row["participant_uuid"],
                    "frame_data": row["frame_data"],
                    "summary": row["summary"],
                }
            )
        return result


def get_screen_frame(bot_id: str, frame_id: int) -> ScreenFrameRecord | None:
    with database._connect() as connection:
        row = connection.execute(
            """SELECT id, bot_id, timestamp_ms, time_source, participant_uuid,
                      frame_data, summary
               FROM screen_frames WHERE bot_id=? AND id=?""",
            (bot_id, frame_id),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "bot_id": row["bot_id"],
            "timestamp_ms": row["timestamp_ms"],
            "time_source": row["time_source"],
            "participant_uuid": row["participant_uuid"],
            "frame_data": row["frame_data"],
            "summary": row["summary"],
        }


def save_screen_frame_summary(frame_id: int, summary: str) -> None:
    with database._connect() as connection:
        connection.execute(
            "UPDATE screen_frames SET summary=? WHERE id=?",
            (summary[:2000], frame_id),
        )


def delete_expired_screen_frames() -> int:
    cutoff = time.time() - database.settings.screenshare_retention_hours * 3600
    with database._connect() as connection:
        result = connection.execute(
            "DELETE FROM screen_frames WHERE created_at < ?", (cutoff,)
        )
        return result.rowcount


def is_screenshare_capture_enabled(bot_id: str) -> bool:
    with database._connect() as connection:
        row = connection.execute(
            "SELECT screenshare_capture_enabled FROM meetings WHERE bot_id=?",
            (bot_id,),
        ).fetchone()
        return row is not None and bool(row["screenshare_capture_enabled"])
