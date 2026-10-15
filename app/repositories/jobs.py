from __future__ import annotations

import random
import sqlite3
import time

from app.repositories.database import _connect
from app.schemas import BackgroundJob


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
