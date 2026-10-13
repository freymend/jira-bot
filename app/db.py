from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from app.config import settings
from app.schemas import (
    BackgroundJob,
    CalendarAutoJoin,
    CalendarScheduledEvent,
    CalendarSyncChange,
    CalendarWatchChannel,
    JSONValue,
    JSONObject,
    MeetingLeaveCandidate,
    MeetingLeaveRetry,
    MeetingRecord,
    MeetingTaskRecord,
    PersistedTaskPayload,
    RecentMeeting,
    RecentTranscriptSegment,
    ScreenFrameRecord,
    TranscriptSegment,
)
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


def create_meeting(
    bot_id: str,
    meeting_url: str | None,
    bot_state: str | None = None,
    *,
    screenshare_capture_enabled: bool = False,
    scheduled_end_at: str | None = None,
    participant_tracking_enabled: bool = False,
) -> None:
    with _connect() as connection:
        connection.execute(
            """INSERT INTO meetings
               (bot_id, meeting_url, bot_state, created_at, screenshare_capture_enabled,
                scheduled_end_at, participant_tracking_enabled)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(bot_id) DO UPDATE SET
                 meeting_url=COALESCE(excluded.meeting_url, meetings.meeting_url),
                 bot_state=COALESCE(excluded.bot_state, meetings.bot_state),
                 screenshare_capture_enabled=excluded.screenshare_capture_enabled,
                 scheduled_end_at=COALESCE(excluded.scheduled_end_at, meetings.scheduled_end_at),
                 participant_tracking_enabled=MAX(
                     meetings.participant_tracking_enabled,
                     excluded.participant_tracking_enabled
                 )""",
            (
                bot_id,
                meeting_url,
                bot_state,
                _now(),
                screenshare_capture_enabled,
                scheduled_end_at,
                participant_tracking_enabled,
            ),
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


def update_bot_state(bot_id: str, bot_state: str) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE meetings SET bot_state=? WHERE bot_id=?", (bot_state, bot_id)
        )


def list_meetings_for_leave_check() -> list[MeetingLeaveCandidate]:
    with _connect() as connection:
        rows = connection.execute(
            """SELECT m.bot_id, m.created_at, m.scheduled_end_at,
                      m.last_transcript_received_at, m.leave_evaluated_activity_at,
                      m.leave_evaluation_started_at, m.participant_tracking_enabled,
                      m.participant_presence_initialized, m.last_participant_event_at,
                      (SELECT COUNT(*) FROM meeting_participants AS p
                       WHERE p.bot_id=m.bot_id AND p.is_present=1)
                          AS active_participant_count,
                      (SELECT GROUP_CONCAT(p.participant_name, char(10))
                       FROM meeting_participants AS p
                       WHERE p.bot_id=m.bot_id AND p.is_host=1
                         AND p.participant_name IS NOT NULL)
                          AS meeting_host_names
               FROM meetings AS m
               WHERE m.bot_state='joined_recording' AND m.leave_status='not_requested'
               ORDER BY m.created_at"""
        ).fetchall()
        result: list[MeetingLeaveCandidate] = []
        for row in rows:
            result.append(
                {
                    "bot_id": row["bot_id"],
                    "created_at": row["created_at"],
                    "scheduled_end_at": row["scheduled_end_at"],
                    "last_transcript_received_at": row["last_transcript_received_at"],
                    "leave_evaluated_activity_at": row["leave_evaluated_activity_at"],
                    "leave_evaluation_started_at": row["leave_evaluation_started_at"],
                    "participant_tracking_enabled": row["participant_tracking_enabled"],
                    "participant_presence_initialized": row[
                        "participant_presence_initialized"
                    ],
                    "last_participant_event_at": row["last_participant_event_at"],
                    "active_participant_count": row["active_participant_count"],
                    "meeting_host_names": row["meeting_host_names"],
                }
            )
        return result


def record_participant_event(
    bot_id: str,
    event_id: str,
    participant_uuid: str,
    participant_name: str | None,
    is_host: bool | None,
    event_type: str,
    timestamp_ms: int,
) -> bool:
    if event_type not in {"join", "leave"}:
        return False
    now = _now()
    with _connect() as connection:
        inserted = connection.execute(
            """INSERT OR IGNORE INTO meeting_participant_events
               (bot_id, event_id, created_at) VALUES (?, ?, ?)""",
            (bot_id, event_id, now),
        )
        if inserted.rowcount != 1:
            return False
        current = connection.execute(
            """SELECT last_event_timestamp_ms FROM meeting_participants
               WHERE bot_id=? AND participant_uuid=?""",
            (bot_id, participant_uuid),
        ).fetchone()
        if (
            current is not None
            and current["last_event_timestamp_ms"] is not None
            and timestamp_ms < current["last_event_timestamp_ms"]
        ):
            return False
        is_present = event_type == "join"
        connection.execute(
            """INSERT INTO meeting_participants
               (bot_id, participant_uuid, participant_name, is_host, is_present,
                last_event_type, last_event_timestamp_ms, last_event_received_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(bot_id, participant_uuid) DO UPDATE SET
                 participant_name=COALESCE(excluded.participant_name,
                                           meeting_participants.participant_name),
                 is_host=COALESCE(excluded.is_host, meeting_participants.is_host),
                 is_present=excluded.is_present,
                 last_event_type=excluded.last_event_type,
                 last_event_timestamp_ms=excluded.last_event_timestamp_ms,
                 last_event_received_at=excluded.last_event_received_at""",
            (
                bot_id,
                participant_uuid,
                participant_name,
                is_host,
                is_present,
                event_type,
                timestamp_ms,
                now,
            ),
        )
        if event_type == "join":
            connection.execute(
                """UPDATE meetings SET participant_presence_initialized=1,
                   last_participant_event_at=? WHERE bot_id=?""",
                (now, bot_id),
            )
        else:
            connection.execute(
                "UPDATE meetings SET last_participant_event_at=? WHERE bot_id=?",
                (now, bot_id),
            )
        return True


def list_uncertain_meeting_leave_retries(
    *, retry_after_seconds: int = 60, now: datetime | None = None
) -> list[MeetingLeaveRetry]:
    cutoff = (
        (now or datetime.now(UTC)) - timedelta(seconds=retry_after_seconds)
    ).isoformat()
    with _connect() as connection:
        rows = connection.execute(
            """SELECT bot_id, leave_requested_at
               FROM meetings
               WHERE bot_state='joined_recording' AND leave_status='uncertain'
                 AND leave_requested_at IS NOT NULL AND leave_requested_at<=?
               ORDER BY leave_requested_at""",
            (cutoff,),
        ).fetchall()
        result: list[MeetingLeaveRetry] = []
        for row in rows:
            result.append(
                {
                    "bot_id": row["bot_id"],
                    "leave_requested_at": row["leave_requested_at"],
                }
            )
        return result


def claim_uncertain_meeting_leave_retry(
    bot_id: str,
    expected_requested_at: str,
    *,
    retry_after_seconds: int = 60,
    now: datetime | None = None,
) -> bool:
    current = now or datetime.now(UTC)
    cutoff = (current - timedelta(seconds=retry_after_seconds)).isoformat()
    with _connect() as connection:
        result = connection.execute(
            """UPDATE meetings
               SET leave_status='requesting', leave_requested_at=?,
                   leave_attempt_count=leave_attempt_count+1
               WHERE bot_id=? AND bot_state='joined_recording'
                 AND leave_status='uncertain' AND leave_requested_at=?
                 AND leave_requested_at<=?""",
            (current.isoformat(), bot_id, expected_requested_at, cutoff),
        )
        return result.rowcount == 1


def get_recent_transcript_segments(
    bot_id: str, limit: int = 40
) -> list[RecentTranscriptSegment]:
    bounded_limit = min(max(limit, 1), 100)
    with _connect() as connection:
        rows = connection.execute(
            """SELECT timestamp_ms, speaker_name, text
               FROM transcript_segments WHERE bot_id=?
               ORDER BY rowid DESC LIMIT ?""",
            (bot_id, bounded_limit),
        ).fetchall()
        result: list[RecentTranscriptSegment] = []
        for row in reversed(rows):
            result.append(
                {
                    "timestamp_ms": row["timestamp_ms"],
                    "speaker_name": row["speaker_name"],
                    "text": row["text"],
                }
            )
        return result


def claim_meeting_leave_evaluation(
    bot_id: str,
    transcript_activity_at: str | None,
    participant_activity_at: str | None,
    activity_token: str,
    *,
    stale_after_seconds: int = 300,
) -> bool:
    now = datetime.now(UTC)
    stale_before = (now - timedelta(seconds=stale_after_seconds)).isoformat()
    with _connect() as connection:
        result = connection.execute(
            """UPDATE meetings SET leave_evaluation_started_at=?
               WHERE bot_id=? AND bot_state='joined_recording'
                 AND leave_status='not_requested'
                 AND last_transcript_received_at IS ?
                 AND last_participant_event_at IS ?
                 AND (leave_evaluated_activity_at IS NULL
                      OR leave_evaluated_activity_at != ?)
                 AND (leave_evaluation_started_at IS NULL
                      OR leave_evaluation_started_at < ?)""",
            (
                now.isoformat(),
                bot_id,
                transcript_activity_at,
                participant_activity_at,
                activity_token,
                stale_before,
            ),
        )
        return result.rowcount == 1


def finish_meeting_leave_evaluation(
    bot_id: str,
    transcript_activity_at: str | None,
    participant_activity_at: str | None,
    activity_token: str,
) -> None:
    with _connect() as connection:
        connection.execute(
            """UPDATE meetings
               SET leave_evaluated_activity_at=?, leave_evaluation_started_at=NULL
               WHERE bot_id=? AND leave_evaluation_started_at IS NOT NULL
                 AND last_transcript_received_at IS ?
                 AND last_participant_event_at IS ?""",
            (activity_token, bot_id, transcript_activity_at, participant_activity_at),
        )


def claim_meeting_leave(
    bot_id: str,
    reason: str,
    *,
    expected_activity_at: str | None = None,
    expected_participant_event_at: str | None = None,
    check_activity: bool = False,
    require_recording: bool = True,
) -> bool:
    if reason not in {
        "wrap_up",
        "all_participants_left",
        "maximum_runtime",
        "manual",
    }:
        raise ValueError("Invalid meeting leave reason")
    now = _now()
    state_condition = " AND bot_state='joined_recording'" if require_recording else ""
    activity_condition = (
        " AND (?=0 OR last_transcript_received_at IS ?)"
        " AND (?=0 OR last_participant_event_at IS ?)"
    )
    parameters: tuple[SQLiteParameter, ...] = (
        reason,
        now,
        bot_id,
        int(check_activity),
        expected_activity_at,
        int(check_activity),
        expected_participant_event_at,
    )
    with _connect() as connection:
        result = connection.execute(
            """UPDATE meetings
               SET leave_status='requesting', leave_reason=?, leave_requested_at=?,
                   leave_attempt_count=leave_attempt_count+1
               WHERE bot_id=? AND leave_status='not_requested'"""
            + state_condition
            + activity_condition,
            parameters,
        )
        return result.rowcount == 1


def finish_meeting_leave(bot_id: str, status: str) -> None:
    if status not in {"requested", "uncertain"}:
        raise ValueError("Invalid meeting leave status")
    with _connect() as connection:
        connection.execute(
            "UPDATE meetings SET leave_status=? WHERE bot_id=? AND leave_status='requesting'",
            (status, bot_id),
        )


def mark_stale_meeting_leave_requests_uncertain(
    *, stale_after_seconds: int = 300
) -> int:
    cutoff = (datetime.now(UTC) - timedelta(seconds=stale_after_seconds)).isoformat()
    with _connect() as connection:
        result = connection.execute(
            """UPDATE meetings SET leave_status='uncertain'
               WHERE leave_status='requesting' AND leave_requested_at < ?""",
            (cutoff,),
        )
        return result.rowcount


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
            if meeting is not None and meeting["status"] != "completed":
                now = time.time()
                connection.execute(
                    """INSERT OR IGNORE INTO background_jobs
                       (kind, bot_id, status, attempts, available_at, created_at, updated_at)
                       VALUES ('finalize_meeting', ?, 'queued', 0, ?, ?, ?)""",
                    (bot_id, now, now, now),
                )
        return result.rowcount == 1


def save_segments(bot_id: str, segments: list[TranscriptSegment]) -> None:
    with _connect() as connection:
        changed = False
        for segment in segments:
            timestamp = segment.get("timestamp_ms")
            speaker_uuid = segment.get("speaker_uuid")
            speaker_name = segment.get("speaker_name")
            identity = segment.get("id") or segment.get("segment_id")
            if identity is None and timestamp is not None:
                identity = f"{speaker_uuid or speaker_name or ''}:{timestamp}"
            if identity is None:
                identity = json.dumps(segment, sort_keys=True, ensure_ascii=False)
            key = hashlib.sha256(str(identity).encode("utf-8")).hexdigest()
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            result = connection.execute(
                """INSERT INTO transcript_segments
                   (bot_id, segment_key, timestamp_ms, speaker_name, speaker_uuid, text, raw_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(bot_id, segment_key) DO UPDATE SET
                     timestamp_ms=excluded.timestamp_ms,
                     speaker_name=excluded.speaker_name,
                     speaker_uuid=excluded.speaker_uuid,
                     text=excluded.text,
                     raw_json=excluded.raw_json
                   WHERE transcript_segments.timestamp_ms IS NOT excluded.timestamp_ms
                      OR transcript_segments.speaker_name IS NOT excluded.speaker_name
                      OR transcript_segments.speaker_uuid IS NOT excluded.speaker_uuid
                      OR transcript_segments.text IS NOT excluded.text
                      OR transcript_segments.raw_json IS NOT excluded.raw_json""",
                (
                    bot_id,
                    key,
                    timestamp,
                    speaker_name,
                    speaker_uuid,
                    text,
                    json.dumps(segment.get("raw", segment), ensure_ascii=False),
                ),
            )
            changed = changed or result.rowcount == 1
        connection.execute(
            """UPDATE meetings SET transcription_state='in_progress'
               WHERE bot_id=? AND transcription_state!='complete'""",
            (bot_id,),
        )
        if changed:
            connection.execute(
                "UPDATE meetings SET last_transcript_received_at=? WHERE bot_id=?",
                (_now(), bot_id),
            )


def save_final_transcript(bot_id: str, segments: list[TranscriptSegment]) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE meetings SET transcript_json=?, transcription_state='complete' WHERE bot_id=?",
            (json.dumps(segments, ensure_ascii=False), bot_id),
        )
    save_segments(bot_id, segments)


def set_recording_started_at(bot_id: str, started_at_ms: int) -> None:
    with _connect() as connection:
        meeting = connection.execute(
            "SELECT recording_started_at_ms FROM meetings WHERE bot_id=?",
            (bot_id,),
        ).fetchone()
        if meeting is None:
            return
        origin = meeting["recording_started_at_ms"] or started_at_ms
        connection.execute(
            """UPDATE meetings
               SET recording_started_at_ms=?, screen_frame_origin_ms=?,
                   screen_frame_origin_source='recording_state'
               WHERE bot_id=?""",
            (origin, origin, bot_id),
        )
        connection.execute(
            """UPDATE screen_frames
               SET timestamp_ms=MAX(received_at_ms - ?, 0),
                   time_source='recording_state'
               WHERE bot_id=?""",
            (origin, bot_id),
        )


def store_screen_frame(
    bot_id: str,
    frame_data: bytes,
    *,
    received_at_ms: int,
    participant_uuid: str | None,
) -> int | None:
    now = time.time()
    cutoff = now - settings.screenshare_retention_hours * 3600
    with _connect() as connection:
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
            < settings.screenshare_sample_interval_seconds * 1000
        ):
            return None
        frame_count = connection.execute(
            "SELECT COUNT(*) FROM screen_frames WHERE bot_id=?", (bot_id,)
        ).fetchone()[0]
        if frame_count >= settings.screenshare_max_frames_per_meeting:
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
    with _connect() as connection:
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
    with _connect() as connection:
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
    with _connect() as connection:
        connection.execute(
            "UPDATE screen_frames SET summary=? WHERE id=?",
            (summary[:2000], frame_id),
        )


def delete_expired_screen_frames() -> int:
    cutoff = time.time() - settings.screenshare_retention_hours * 3600
    with _connect() as connection:
        result = connection.execute(
            "DELETE FROM screen_frames WHERE created_at < ?", (cutoff,)
        )
        return result.rowcount


def clear_stale_extraction_checkpoints(bot_id: str, transcript_hash: str) -> None:
    with _connect() as connection:
        connection.execute(
            "DELETE FROM task_extraction_cache WHERE bot_id=? AND transcript_hash!=?",
            (bot_id, transcript_hash),
        )


def get_extraction_checkpoint(
    bot_id: str, transcript_hash: str, stage: str, item_key: str
) -> JSONValue | None:
    with _connect() as connection:
        row = connection.execute(
            """SELECT result_json FROM task_extraction_cache
               WHERE bot_id=? AND transcript_hash=? AND stage=? AND item_key=?""",
            (bot_id, transcript_hash, stage, item_key),
        ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["result_json"])
        except json.JSONDecodeError:
            connection.execute(
                """DELETE FROM task_extraction_cache
                   WHERE bot_id=? AND transcript_hash=? AND stage=? AND item_key=?""",
                (bot_id, transcript_hash, stage, item_key),
            )
            return None


def delete_extraction_checkpoint(
    bot_id: str, transcript_hash: str, stage: str, item_key: str
) -> None:
    with _connect() as connection:
        connection.execute(
            """DELETE FROM task_extraction_cache
               WHERE bot_id=? AND transcript_hash=? AND stage=? AND item_key=?""",
            (bot_id, transcript_hash, stage, item_key),
        )


def save_extraction_checkpoint(
    bot_id: str,
    transcript_hash: str,
    stage: str,
    item_key: str,
    result: JSONValue,
) -> None:
    with _connect() as connection:
        connection.execute(
            """INSERT INTO task_extraction_cache
               (bot_id, transcript_hash, stage, item_key, result_json, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(bot_id, transcript_hash, stage, item_key)
               DO UPDATE SET result_json=excluded.result_json,
                             updated_at=excluded.updated_at""",
            (
                bot_id,
                transcript_hash,
                stage,
                item_key,
                json.dumps(result, ensure_ascii=False),
                time.time(),
            ),
        )


def enqueue_finalization(bot_id: str, *, allow_reprocess: bool = False) -> str:
    now = time.time()
    with _connect() as connection:
        meeting = connection.execute(
            "SELECT status FROM meetings WHERE bot_id=?", (bot_id,)
        ).fetchone()
        if meeting is None:
            return "not_found"
        if meeting["status"] == "completed":
            return "completed"

        connection.execute(
            """INSERT OR IGNORE INTO background_jobs
               (kind, bot_id, status, attempts, available_at, created_at, updated_at)
               VALUES ('finalize_meeting', ?, 'queued', 0, ?, ?, ?)""",
            (bot_id, now, now, now),
        )
        job = connection.execute(
            "SELECT status FROM background_jobs WHERE kind='finalize_meeting' AND bot_id=?",
            (bot_id,),
        ).fetchone()
        if job is None:
            return "queued"

        status = str(job["status"])
        if allow_reprocess and status in {"completed", "failed"}:
            connection.execute(
                """UPDATE background_jobs SET status='queued', attempts=0,
                     available_at=?, lease_until=NULL, locked_by=NULL, last_error=NULL,
                     updated_at=? WHERE kind='finalize_meeting' AND bot_id=?""",
                (now, now, bot_id),
            )
            return "queued"
        return "processing" if status == "running" else status


def claim_next_job(worker_id: str, *, lease_seconds: int = 60) -> BackgroundJob | None:
    now = time.time()
    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """SELECT id, bot_id, status FROM background_jobs
               WHERE available_at <= ?
                 AND (status='queued' OR
                      (status='running' AND lease_until <= ?))
               ORDER BY available_at, id LIMIT 1""",
            (now, now),
        ).fetchone()
        if row is None:
            return None
        job_id = int(row["id"])
        if row["status"] == "running":
            connection.execute(
                """UPDATE meetings SET status='retrying',
                     error='Worker lease expired; recovering the queued job.'
                   WHERE bot_id=?""",
                (row["bot_id"],),
            )
        connection.execute(
            """UPDATE background_jobs SET status='running', attempts=attempts+1,
                 lease_until=?, locked_by=?, updated_at=? WHERE id=?""",
            (now + lease_seconds, worker_id, now, job_id),
        )
        claimed = connection.execute(
            "SELECT * FROM background_jobs WHERE id=?", (job_id,)
        ).fetchone()
        if claimed is None:
            return None
        return {
            "id": claimed["id"],
            "kind": claimed["kind"],
            "bot_id": claimed["bot_id"],
            "status": claimed["status"],
            "attempts": claimed["attempts"],
            "available_at": claimed["available_at"],
            "lease_until": claimed["lease_until"],
            "locked_by": claimed["locked_by"],
            "last_error": claimed["last_error"],
            "created_at": claimed["created_at"],
            "updated_at": claimed["updated_at"],
        }


def renew_job_lease(job_id: int, worker_id: str, *, lease_seconds: int = 60) -> bool:
    now = time.time()
    with _connect() as connection:
        result = connection.execute(
            """UPDATE background_jobs SET lease_until=?, updated_at=?
               WHERE id=? AND status='running' AND locked_by=?""",
            (now + lease_seconds, now, job_id, worker_id),
        )
        connection.execute(
            """INSERT INTO worker_heartbeats(worker_id, heartbeat_at) VALUES (?, ?)
               ON CONFLICT(worker_id) DO UPDATE SET heartbeat_at=excluded.heartbeat_at""",
            (worker_id, now),
        )
        return result.rowcount == 1


def record_worker_heartbeat(worker_id: str) -> None:
    with _connect() as connection:
        connection.execute(
            """INSERT INTO worker_heartbeats(worker_id, heartbeat_at) VALUES (?, ?)
               ON CONFLICT(worker_id) DO UPDATE SET heartbeat_at=excluded.heartbeat_at""",
            (worker_id, time.time()),
        )


def health_checks(*, worker_stale_after_seconds: int = 30) -> dict[str, bool]:
    try:
        with _connect() as connection:
            connection.execute("SELECT 1").fetchone()
            connection.execute("SELECT COUNT(*) FROM background_jobs").fetchone()
            heartbeat = connection.execute(
                "SELECT MAX(heartbeat_at) AS latest FROM worker_heartbeats"
            ).fetchone()
    except OSError, sqlite3.Error:
        return {"database": False, "worker": False}

    latest = heartbeat["latest"] if heartbeat is not None else None
    return {
        "database": True,
        "worker": latest is not None
        and latest >= time.time() - worker_stale_after_seconds,
    }


def finish_job(job_id: int, worker_id: str) -> bool:
    with _connect() as connection:
        result = connection.execute(
            """UPDATE background_jobs SET status='completed', lease_until=NULL,
                 locked_by=NULL, last_error=NULL, updated_at=?
               WHERE id=? AND status='running' AND locked_by=?""",
            (time.time(), job_id, worker_id),
        )
        return result.rowcount == 1


def fail_job(
    job_id: int,
    worker_id: str,
    error: str,
    *,
    attempts: int,
    retryable: bool,
    retry_after_seconds: float | None = None,
) -> str:
    now = time.time()
    exponent = min(max(attempts - 1, 0), 6)
    backoff = min(5 * (2**exponent), 300)
    delay = max(random.uniform(backoff / 2, backoff), retry_after_seconds or 0)
    status = "queued" if retryable else "failed"
    with _connect() as connection:
        connection.execute(
            """UPDATE background_jobs SET status=?, available_at=?, lease_until=NULL,
                 locked_by=NULL, last_error=?, updated_at=?
               WHERE id=? AND status='running' AND locked_by=?""",
            (
                status,
                now + delay if retryable else now,
                error[:1000],
                now,
                job_id,
                worker_id,
            ),
        )
    return status


def set_meeting_status(bot_id: str, status: str, error: str | None = None) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE meetings SET status=?, error=? WHERE bot_id=?",
            (status, error, bot_id),
        )


def add_task(bot_id: str, task: PersistedTaskPayload) -> tuple[sqlite3.Row, bool]:
    fingerprint_input = "\n".join(
        [
            (task.get("owner_speaker") or "").strip().casefold(),
            task["evidence_quote"].strip().casefold(),
            str(task.get("timestamp_ms") or ""),
        ]
    )
    fingerprint = hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest()
    with _connect() as connection:
        cursor = connection.execute(
            """INSERT OR IGNORE INTO tasks
               (bot_id, fingerprint, summary, description, owner_speaker,
                owner_evidence_quote, owner_timestamp_ms, due_date, priority,
                evidence_quote, timestamp_ms, context_evidence_json,
                visual_evidence_json, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                bot_id,
                fingerprint,
                task["summary"],
                task.get("description", ""),
                task.get("owner_speaker"),
                task.get("owner_evidence_quote"),
                task.get("owner_timestamp_ms"),
                task.get("due_date"),
                task.get("priority"),
                task["evidence_quote"],
                task.get("timestamp_ms"),
                json.dumps(task.get("context_evidence", []), ensure_ascii=False),
                json.dumps(task.get("visual_evidence", []), ensure_ascii=False),
                task["confidence"],
            ),
        )
        inserted = cursor.rowcount == 1
        if not inserted:
            connection.execute(
                """UPDATE tasks SET summary=?, description=?, owner_speaker=?,
                   owner_evidence_quote=?, owner_timestamp_ms=?, due_date=?, priority=?,
                   evidence_quote=?, timestamp_ms=?, context_evidence_json=?,
                   visual_evidence_json=?, confidence=?
                   WHERE bot_id=? AND fingerprint=? AND jira_key IS NULL
                     AND status IN ('pending', 'dry_run')""",
                (
                    task["summary"],
                    task.get("description", ""),
                    task.get("owner_speaker"),
                    task.get("owner_evidence_quote"),
                    task.get("owner_timestamp_ms"),
                    task.get("due_date"),
                    task.get("priority"),
                    task["evidence_quote"],
                    task.get("timestamp_ms"),
                    json.dumps(task.get("context_evidence", []), ensure_ascii=False),
                    json.dumps(task.get("visual_evidence", []), ensure_ascii=False),
                    task["confidence"],
                    bot_id,
                    fingerprint,
                ),
            )
        row = connection.execute(
            "SELECT * FROM tasks WHERE bot_id=? AND fingerprint=?",
            (bot_id, fingerprint),
        ).fetchone()
        return row, inserted


def set_task_status(
    task_id: int, status: str, jira_key: str | None = None, error: str | None = None
) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE tasks SET status=?, jira_key=COALESCE(?, jira_key), error=? WHERE id=?",
            (status, jira_key, error, task_id),
        )


def is_screenshare_capture_enabled(bot_id: str) -> bool:
    with _connect() as connection:
        row = connection.execute(
            "SELECT screenshare_capture_enabled FROM meetings WHERE bot_id=?",
            (bot_id,),
        ).fetchone()
        return row is not None and bool(row["screenshare_capture_enabled"])


def list_recent_meetings(limit: int = 25) -> list[RecentMeeting]:
    bounded_limit = min(max(limit, 1), 100)
    with _connect() as connection:
        rows = connection.execute(
            """SELECT m.bot_id, m.meeting_url, m.created_at, m.bot_state,
                      m.transcription_state, m.status, m.error, COUNT(t.id) AS task_count
               FROM meetings AS m
               LEFT JOIN tasks AS t ON t.bot_id=m.bot_id
               GROUP BY m.bot_id
               ORDER BY m.created_at DESC
               LIMIT ?""",
            (bounded_limit,),
        ).fetchall()
        result: list[RecentMeeting] = []
        for row in rows:
            result.append(
                {
                    "bot_id": row["bot_id"],
                    "meeting_url": row["meeting_url"],
                    "created_at": row["created_at"],
                    "bot_state": row["bot_state"],
                    "transcription_state": row["transcription_state"],
                    "status": row["status"],
                    "error": row["error"],
                    "task_count": row["task_count"],
                }
            )
        return result


def get_meeting(bot_id: str) -> MeetingRecord | None:
    with _connect() as connection:
        meeting = connection.execute(
            "SELECT * FROM meetings WHERE bot_id=?", (bot_id,)
        ).fetchone()
        if meeting is None:
            return None

        tasks: list[MeetingTaskRecord] = []
        for row in connection.execute(
            "SELECT * FROM tasks WHERE bot_id=? ORDER BY id", (bot_id,)
        ):
            tasks.append(
                {
                    "id": row["id"],
                    "bot_id": row["bot_id"],
                    "fingerprint": row["fingerprint"],
                    "summary": row["summary"],
                    "description": row["description"],
                    "owner_speaker": row["owner_speaker"],
                    "owner_evidence_quote": row["owner_evidence_quote"],
                    "owner_timestamp_ms": row["owner_timestamp_ms"],
                    "due_date": row["due_date"],
                    "priority": row["priority"],
                    "evidence_quote": row["evidence_quote"],
                    "timestamp_ms": row["timestamp_ms"],
                    "context_evidence": json.loads(
                        row["context_evidence_json"] or "[]"
                    ),
                    "visual_evidence": json.loads(row["visual_evidence_json"] or "[]"),
                    "confidence": row["confidence"],
                    "status": row["status"],
                    "jira_key": row["jira_key"],
                    "error": row["error"],
                }
            )

        return {
            "bot_id": meeting["bot_id"],
            "meeting_url": meeting["meeting_url"],
            "created_at": meeting["created_at"],
            "transcription_state": meeting["transcription_state"],
            "status": meeting["status"],
            "error": meeting["error"],
            "bot_state": meeting["bot_state"],
            "transcript": json.loads(meeting["transcript_json"] or "null"),
            "recording_started_at_ms": meeting["recording_started_at_ms"],
            "screen_frame_origin_ms": meeting["screen_frame_origin_ms"],
            "screen_frame_origin_source": meeting["screen_frame_origin_source"],
            "screenshare_capture_enabled": meeting["screenshare_capture_enabled"],
            "scheduled_end_at": meeting["scheduled_end_at"],
            "last_transcript_received_at": meeting["last_transcript_received_at"],
            "leave_evaluated_activity_at": meeting["leave_evaluated_activity_at"],
            "leave_evaluation_started_at": meeting["leave_evaluation_started_at"],
            "leave_status": meeting["leave_status"],
            "leave_requested_at": meeting["leave_requested_at"],
            "leave_reason": meeting["leave_reason"],
            "participant_tracking_enabled": meeting["participant_tracking_enabled"],
            "participant_presence_initialized": meeting[
                "participant_presence_initialized"
            ],
            "last_participant_event_at": meeting["last_participant_event_at"],
            "leave_attempt_count": meeting["leave_attempt_count"],
            "tasks": tasks,
        }
