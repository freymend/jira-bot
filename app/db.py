from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import settings


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
        with connection:
            yield connection
    finally:
        connection.close()


def init_db() -> None:
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meetings (
                bot_id TEXT PRIMARY KEY,
                meeting_url TEXT,
                bot_state TEXT,
                created_at TEXT NOT NULL,
                transcription_state TEXT NOT NULL DEFAULT 'not_started',
                status TEXT NOT NULL DEFAULT 'waiting',
                transcript_json TEXT,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS webhook_events (
                event_id TEXT PRIMARY KEY,
                bot_id TEXT NOT NULL,
                trigger TEXT NOT NULL,
                received_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS transcript_segments (
                bot_id TEXT NOT NULL,
                segment_key TEXT NOT NULL,
                timestamp_ms INTEGER,
                speaker_name TEXT,
                speaker_uuid TEXT,
                text TEXT NOT NULL,
                raw_json TEXT NOT NULL,
                PRIMARY KEY (bot_id, segment_key)
            );
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                summary TEXT NOT NULL,
                description TEXT NOT NULL,
                owner_speaker TEXT,
                due_date TEXT,
                priority TEXT,
                evidence_quote TEXT NOT NULL,
                timestamp_ms INTEGER,
                confidence REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                jira_key TEXT,
                error TEXT,
                UNIQUE (bot_id, fingerprint)
            );
            CREATE TABLE IF NOT EXISTS background_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                bot_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                attempts INTEGER NOT NULL DEFAULT 0,
                available_at REAL NOT NULL,
                lease_until REAL,
                locked_by TEXT,
                last_error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE (kind, bot_id)
            );
            CREATE INDEX IF NOT EXISTS idx_background_jobs_ready
                ON background_jobs(status, available_at, lease_until);
            CREATE TABLE IF NOT EXISTS worker_heartbeats (
                worker_id TEXT PRIMARY KEY,
                heartbeat_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tasks_bot_id ON tasks(bot_id);
            """
        )
        meeting_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(meetings)")
        }
        if "bot_state" not in meeting_columns:
            connection.execute("ALTER TABLE meetings ADD COLUMN bot_state TEXT")
        now = time.time()
        connection.execute(
            """INSERT OR IGNORE INTO background_jobs
               (kind, bot_id, status, attempts, available_at, created_at, updated_at)
               SELECT 'finalize_meeting', bot_id, 'queued', 0, ?, ?, ?
               FROM meetings WHERE status='processing'""",
            (now, now, now),
        )


def create_meeting(
    bot_id: str, meeting_url: str | None, bot_state: str | None = None
) -> None:
    with _connect() as connection:
        connection.execute(
            """INSERT INTO meetings(bot_id, meeting_url, bot_state, created_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(bot_id) DO UPDATE SET
                 meeting_url=COALESCE(excluded.meeting_url, meetings.meeting_url),
                 bot_state=COALESCE(excluded.bot_state, meetings.bot_state)""",
            (bot_id, meeting_url, bot_state, _now()),
        )


def update_bot_state(bot_id: str, bot_state: str) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE meetings SET bot_state=? WHERE bot_id=?", (bot_state, bot_id)
        )


def record_webhook(
    event_id: str,
    bot_id: str,
    trigger: str,
    payload: dict[str, Any],
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


def save_segments(bot_id: str, segments: list[dict[str, Any]]) -> None:
    with _connect() as connection:
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
            connection.execute(
                """INSERT INTO transcript_segments
                   (bot_id, segment_key, timestamp_ms, speaker_name, speaker_uuid, text, raw_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(bot_id, segment_key) DO UPDATE SET
                     timestamp_ms=excluded.timestamp_ms,
                     speaker_name=excluded.speaker_name,
                     speaker_uuid=excluded.speaker_uuid,
                     text=excluded.text,
                     raw_json=excluded.raw_json""",
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
        connection.execute(
            """UPDATE meetings SET transcription_state='in_progress'
               WHERE bot_id=? AND transcription_state!='complete'""",
            (bot_id,),
        )


def save_final_transcript(bot_id: str, segments: list[dict[str, Any]]) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE meetings SET transcript_json=?, transcription_state='complete' WHERE bot_id=?",
            (json.dumps(segments, ensure_ascii=False), bot_id),
        )
    save_segments(bot_id, segments)


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


def claim_next_job(
    worker_id: str, *, lease_seconds: int = 60, max_attempts: int = 5
) -> dict[str, Any] | None:
    now = time.time()
    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        expired = connection.execute(
            """SELECT id, bot_id FROM background_jobs
               WHERE status='running' AND lease_until <= ? AND attempts >= ?""",
            (now, max_attempts),
        ).fetchall()
        for row in expired:
            error = "Worker lease expired after the maximum number of attempts."
            updated = connection.execute(
                """UPDATE background_jobs SET status='failed', lease_until=NULL,
                     locked_by=NULL, last_error=?, updated_at=?
                   WHERE id=? AND status='running' AND lease_until <= ? AND attempts >= ?""",
                (error, now, row["id"], now, max_attempts),
            )
            if updated.rowcount == 1:
                connection.execute(
                    "UPDATE meetings SET status='error', error=? WHERE bot_id=?",
                    (error, row["bot_id"]),
                )
        row = connection.execute(
            """SELECT id FROM background_jobs
               WHERE available_at <= ?
                 AND (status='queued' OR
                      (status='running' AND lease_until <= ? AND attempts < ?))
               ORDER BY available_at, id LIMIT 1""",
            (now, now, max_attempts),
        ).fetchone()
        if row is None:
            return None
        job_id = int(row["id"])
        connection.execute(
            """UPDATE background_jobs SET status='running', attempts=attempts+1,
                 lease_until=?, locked_by=?, updated_at=? WHERE id=?""",
            (now + lease_seconds, worker_id, now, job_id),
        )
        claimed = connection.execute(
            "SELECT * FROM background_jobs WHERE id=?", (job_id,)
        ).fetchone()
        return dict(claimed) if claimed is not None else None


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
    max_attempts: int = 5,
) -> str:
    now = time.time()
    is_terminal = attempts >= max_attempts
    delay = min(5 * (2 ** max(attempts - 1, 0)), 300)
    status = "failed" if is_terminal else "queued"
    with _connect() as connection:
        connection.execute(
            """UPDATE background_jobs SET status=?, available_at=?, lease_until=NULL,
                 locked_by=NULL, last_error=?, updated_at=?
               WHERE id=? AND status='running' AND locked_by=?""",
            (
                status,
                now if is_terminal else now + delay,
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


def add_task(bot_id: str, task: dict[str, Any]) -> tuple[sqlite3.Row, bool]:
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
               (bot_id, fingerprint, summary, description, owner_speaker, due_date,
                priority, evidence_quote, timestamp_ms, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                bot_id,
                fingerprint,
                task["summary"],
                task.get("description", ""),
                task.get("owner_speaker"),
                task.get("due_date"),
                task.get("priority"),
                task["evidence_quote"],
                task.get("timestamp_ms"),
                task["confidence"],
            ),
        )
        inserted = cursor.rowcount == 1
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


def get_meeting(bot_id: str) -> dict[str, Any] | None:
    with _connect() as connection:
        meeting = connection.execute(
            "SELECT * FROM meetings WHERE bot_id=?", (bot_id,)
        ).fetchone()
        if meeting is None:
            return None
        result = dict(meeting)
        result["transcript"] = json.loads(result.pop("transcript_json") or "null")
        result["tasks"] = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM tasks WHERE bot_id=? ORDER BY id", (bot_id,)
            )
        ]
        return result
