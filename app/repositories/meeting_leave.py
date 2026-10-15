from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.repositories.database import SQLiteParameter, _connect, _now
from app.schemas import (
    MeetingLeaveCandidate,
    MeetingLeaveRetry,
    MeetingLeaveWatchdogCandidate,
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
                   leave_attempt_count=leave_attempt_count+1,
                   leave_watchdog_checked_at=NULL, leave_stuck_at=NULL
               WHERE bot_id=? AND bot_state='joined_recording'
                 AND leave_status='uncertain' AND leave_requested_at=?
                 AND leave_requested_at<=?""",
            (current.isoformat(), bot_id, expected_requested_at, cutoff),
        )
        return result.rowcount == 1


def list_stale_meeting_leave_watchdog_candidates(
    *,
    timeout_seconds: int = 300,
    poll_interval_seconds: int = 60,
    now: datetime | None = None,
) -> list[MeetingLeaveWatchdogCandidate]:
    current = now or datetime.now(UTC)
    stale_before = (current - timedelta(seconds=timeout_seconds)).isoformat()
    checked_before = (current - timedelta(seconds=poll_interval_seconds)).isoformat()
    with _connect() as connection:
        rows = connection.execute(
            """SELECT bot_id, leave_requested_at
               FROM meetings
               WHERE leave_status IN ('requested', 'stuck', 'processing')
                 AND leave_requested_at IS NOT NULL AND leave_requested_at<=?
                 AND (leave_watchdog_checked_at IS NULL
                      OR leave_watchdog_checked_at<=?)
               ORDER BY leave_requested_at""",
            (stale_before, checked_before),
        ).fetchall()
        return [
            {
                "bot_id": row["bot_id"],
                "leave_requested_at": row["leave_requested_at"],
            }
            for row in rows
        ]


def claim_meeting_leave_watchdog_poll(
    bot_id: str,
    expected_requested_at: str,
    *,
    timeout_seconds: int = 300,
    poll_interval_seconds: int = 60,
    now: datetime | None = None,
) -> bool:
    current = now or datetime.now(UTC)
    stale_before = (current - timedelta(seconds=timeout_seconds)).isoformat()
    checked_before = (current - timedelta(seconds=poll_interval_seconds)).isoformat()
    with _connect() as connection:
        result = connection.execute(
            """UPDATE meetings SET leave_watchdog_checked_at=?
               WHERE bot_id=? AND leave_requested_at=?
                 AND leave_requested_at<=?
                 AND leave_status IN ('requested', 'stuck', 'processing')
                 AND (leave_watchdog_checked_at IS NULL
                      OR leave_watchdog_checked_at<=?)""",
            (
                current.isoformat(),
                bot_id,
                expected_requested_at,
                stale_before,
                checked_before,
            ),
        )
        return result.rowcount == 1


def record_meeting_leave_watchdog_state(
    bot_id: str,
    expected_requested_at: str,
    bot_state: str | None,
    leave_status: str,
    *,
    now: datetime | None = None,
) -> bool:
    if leave_status not in {"stuck", "processing", "failed"}:
        raise ValueError("Invalid leave watchdog status")
    current = now or datetime.now(UTC)
    checked_at = current.isoformat()
    with _connect() as connection:
        meeting = connection.execute(
            """SELECT leave_status, bot_state FROM meetings
               WHERE bot_id=? AND leave_requested_at=?
                 AND leave_status IN ('requested', 'stuck', 'processing')""",
            (bot_id, expected_requested_at),
        ).fetchone()
        if meeting is None or meeting["bot_state"] == "post_processing_completed":
            return False
        result = connection.execute(
            """UPDATE meetings SET bot_state=COALESCE(?, bot_state),
                   leave_status=?, leave_watchdog_checked_at=?,
                   leave_stuck_at=CASE WHEN ?='stuck'
                       THEN COALESCE(leave_stuck_at, ?) ELSE NULL END
               WHERE bot_id=? AND leave_requested_at=?
                 AND leave_status IN ('requested', 'stuck', 'processing')
                 AND bot_state IS NOT 'post_processing_completed'""",
            (
                bot_state,
                leave_status,
                checked_at,
                leave_status,
                checked_at,
                bot_id,
                expected_requested_at,
            ),
        )
        return result.rowcount == 1 and meeting["leave_status"] != leave_status


def finish_meeting_leave_watchdog(
    bot_id: str,
    status: str,
    *,
    expected_requested_at: str | None = None,
) -> bool:
    if status not in {"completed", "failed"}:
        raise ValueError("Invalid leave watchdog result")
    request_condition = (
        " AND leave_requested_at=?" if expected_requested_at is not None else ""
    )
    parameters: tuple[SQLiteParameter, ...] = (
        status,
        _now(),
        bot_id,
        *((expected_requested_at,) if expected_requested_at is not None else ()),
    )
    with _connect() as connection:
        result = connection.execute(
            """UPDATE meetings SET leave_status=?, leave_stuck_at=NULL,
                   leave_watchdog_checked_at=?
               WHERE bot_id=? AND leave_status IN (
                   'requesting', 'requested', 'uncertain', 'stuck', 'processing', 'failed'
               )"""
            + request_condition,
            parameters,
        )
        return result.rowcount == 1


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
                   leave_attempt_count=leave_attempt_count+1,
                   leave_watchdog_checked_at=NULL, leave_stuck_at=NULL
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
