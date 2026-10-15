from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.repositories.database import _connect, _now
from app.schemas import (
    CalendarAutoJoin,
    CalendarScheduledEvent,
    CalendarSyncChange,
    CalendarWatchChannel,
)


def claim_calendar_auto_join(
    event_key: str, calendar_event_id: str, event_start: str
) -> bool:
    now = _now()
    with _connect() as connection:
        result = connection.execute(
            """INSERT OR IGNORE INTO calendar_auto_join_requests
               (event_key, calendar_event_id, event_start, status, created_at, updated_at)
               VALUES (?, ?, ?, 'requesting', ?, ?)""",
            (event_key, calendar_event_id, event_start, now, now),
        )
        return result.rowcount == 1


def finish_calendar_auto_join(
    event_key: str,
    status: str,
    *,
    bot_id: str | None = None,
    error: str | None = None,
) -> None:
    if status not in {"requested", "uncertain"}:
        raise ValueError("Invalid calendar auto-join status")
    with _connect() as connection:
        connection.execute(
            """UPDATE calendar_auto_join_requests
               SET status=?, bot_id=COALESCE(?, bot_id), error=?, updated_at=?
               WHERE event_key=?""",
            (status, bot_id, error[:1000] if error else None, _now(), event_key),
        )


def mark_stale_calendar_auto_joins_uncertain(*, stale_after_seconds: int = 300) -> int:
    cutoff = (datetime.now(UTC) - timedelta(seconds=stale_after_seconds)).isoformat()
    with _connect() as connection:
        result = connection.execute(
            """UPDATE calendar_auto_join_requests
               SET status='uncertain', error='Request interrupted; verify Attendee before retrying.',
                   updated_at=?
               WHERE status='requesting' AND updated_at < ?""",
            (_now(), cutoff),
        )
        return result.rowcount


def list_calendar_auto_joins(limit: int = 20) -> list[CalendarAutoJoin]:
    bounded_limit = min(max(limit, 1), 100)
    with _connect() as connection:
        rows = connection.execute(
            """SELECT calendar_event_id, event_start, status, bot_id, error,
                      created_at, updated_at
               FROM calendar_auto_join_requests
               ORDER BY event_start DESC LIMIT ?""",
            (bounded_limit,),
        ).fetchall()
        result: list[CalendarAutoJoin] = []
        for row in rows:
            result.append(
                {
                    "calendar_event_id": row["calendar_event_id"],
                    "event_start": row["event_start"],
                    "status": row["status"],
                    "bot_id": row["bot_id"],
                    "error": row["error"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )
        return result


def get_calendar_sync_token(calendar_id: str) -> str | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT sync_token FROM calendar_sync_state WHERE calendar_id=?",
            (calendar_id,),
        ).fetchone()
        return str(row["sync_token"]) if row is not None else None


def clear_calendar_sync_token(calendar_id: str) -> None:
    with _connect() as connection:
        connection.execute(
            "DELETE FROM calendar_sync_state WHERE calendar_id=?", (calendar_id,)
        )


def apply_calendar_sync(
    calendar_id: str,
    changes: list[CalendarSyncChange],
    sync_token: str,
    *,
    replace_all: bool = False,
) -> None:
    now = _now()
    with _connect() as connection:
        if replace_all:
            connection.execute(
                "DELETE FROM calendar_scheduled_events WHERE calendar_id=?",
                (calendar_id,),
            )
        for change in changes:
            event_key = change["event_key"]
            if change["action"] == "delete":
                connection.execute(
                    """DELETE FROM calendar_scheduled_events
                       WHERE calendar_id=? AND event_key=?""",
                    (calendar_id, event_key),
                )
            else:
                connection.execute(
                    """INSERT INTO calendar_scheduled_events
                       (calendar_id, event_key, calendar_event_id, event_start,
                        event_end, meeting_url, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(calendar_id, event_key) DO UPDATE SET
                         calendar_event_id=excluded.calendar_event_id,
                         event_start=excluded.event_start,
                         event_end=excluded.event_end,
                         meeting_url=excluded.meeting_url,
                         updated_at=excluded.updated_at""",
                    (
                        calendar_id,
                        event_key,
                        change["calendar_event_id"],
                        change["event_start"],
                        change.get("event_end"),
                        change["meeting_url"],
                        now,
                    ),
                )
        connection.execute(
            """INSERT INTO calendar_sync_state(calendar_id, sync_token, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(calendar_id) DO UPDATE SET
                 sync_token=excluded.sync_token, updated_at=excluded.updated_at""",
            (calendar_id, sync_token, now),
        )


def list_due_calendar_events(
    calendar_id: str, time_min: str, time_max: str
) -> list[CalendarScheduledEvent]:
    with _connect() as connection:
        rows = connection.execute(
            """SELECT event_key, calendar_event_id, event_start, event_end, meeting_url
               FROM calendar_scheduled_events
               WHERE calendar_id=? AND event_start>=? AND event_start<=?
               ORDER BY event_start LIMIT 100""",
            (calendar_id, time_min, time_max),
        ).fetchall()
        result: list[CalendarScheduledEvent] = []
        for row in rows:
            result.append(
                {
                    "event_key": row["event_key"],
                    "calendar_event_id": row["calendar_event_id"],
                    "event_start": row["event_start"],
                    "event_end": row["event_end"],
                    "meeting_url": row["meeting_url"],
                }
            )
        return result


def create_calendar_watch_channel(
    channel_id: str, calendar_id: str, channel_token: str, expiration_ms: int
) -> None:
    now = _now()
    with _connect() as connection:
        connection.execute(
            """INSERT INTO calendar_watch_channels
               (channel_id, calendar_id, channel_token, expiration_ms, status,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, 'pending', ?, ?)""",
            (channel_id, calendar_id, channel_token, expiration_ms, now, now),
        )


def activate_calendar_watch_channel(
    channel_id: str, resource_id: str, expiration_ms: int
) -> None:
    with _connect() as connection:
        connection.execute(
            """UPDATE calendar_watch_channels SET resource_id=?, expiration_ms=?,
                 status='active', updated_at=? WHERE channel_id=?""",
            (resource_id, expiration_ms, _now(), channel_id),
        )


def get_calendar_watch_channel(channel_id: str) -> CalendarWatchChannel | None:
    with _connect() as connection:
        row = connection.execute(
            """SELECT channel_id, calendar_id, channel_token, resource_id,
                      expiration_ms, status
               FROM calendar_watch_channels WHERE channel_id=?""",
            (channel_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "channel_id": row["channel_id"],
            "calendar_id": row["calendar_id"],
            "channel_token": row["channel_token"],
            "resource_id": row["resource_id"],
            "expiration_ms": row["expiration_ms"],
            "status": row["status"],
        }


def get_current_calendar_watch(calendar_id: str) -> CalendarWatchChannel | None:
    with _connect() as connection:
        row = connection.execute(
            """SELECT channel_id, calendar_id, channel_token, resource_id,
                      expiration_ms, status
               FROM calendar_watch_channels
               WHERE calendar_id=? AND status='active'
               ORDER BY created_at DESC LIMIT 1""",
            (calendar_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "channel_id": row["channel_id"],
            "calendar_id": row["calendar_id"],
            "channel_token": row["channel_token"],
            "resource_id": row["resource_id"],
            "expiration_ms": row["expiration_ms"],
            "status": row["status"],
        }


def delete_calendar_watch_channel(channel_id: str) -> None:
    with _connect() as connection:
        connection.execute(
            "DELETE FROM calendar_watch_channels WHERE channel_id=?", (channel_id,)
        )
