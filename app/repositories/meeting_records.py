from __future__ import annotations

import json

from app.repositories.database import _connect, _now
from app.schemas import (
    MeetingRecordingOriginRow,
    MeetingRecord,
    MeetingTaskRecord,
    RecentMeeting,
)


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


def update_bot_state(bot_id: str, bot_state: str) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE meetings SET bot_state=? WHERE bot_id=?", (bot_state, bot_id)
        )


def set_recording_started_at(bot_id: str, started_at_ms: int) -> None:
    with _connect() as connection:
        meeting = connection.execute(
            "SELECT recording_started_at_ms FROM meetings WHERE bot_id=?",
            (bot_id,),
        ).fetchone()
        if meeting is None:
            return
        recording_origin: MeetingRecordingOriginRow = {
            "recording_started_at_ms": meeting["recording_started_at_ms"],
        }
        origin = recording_origin["recording_started_at_ms"] or started_at_ms
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


def set_meeting_status(bot_id: str, status: str, error: str | None = None) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE meetings SET status=?, error=? WHERE bot_id=?",
            (status, error, bot_id),
        )


def list_recent_meetings(limit: int = 25) -> list[RecentMeeting]:
    bounded_limit = min(max(limit, 1), 100)
    with _connect() as connection:
        rows = connection.execute(
            """SELECT m.bot_id, m.meeting_url, m.created_at, m.bot_state,
                      m.transcription_state, m.status, m.error, m.leave_status,
                      m.leave_stuck_at, COUNT(t.id) AS task_count
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
                    "leave_status": row["leave_status"],
                    "leave_stuck_at": row["leave_stuck_at"],
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
            "leave_watchdog_checked_at": meeting["leave_watchdog_checked_at"],
            "leave_stuck_at": meeting["leave_stuck_at"],
            "participant_tracking_enabled": meeting["participant_tracking_enabled"],
            "participant_presence_initialized": meeting[
                "participant_presence_initialized"
            ],
            "last_participant_event_at": meeting["last_participant_event_at"],
            "leave_attempt_count": meeting["leave_attempt_count"],
            "tasks": tasks,
        }
