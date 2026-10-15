from __future__ import annotations

import hashlib
import json

from app.repositories.database import _connect
from app.schemas import PersistedTaskPayload, TaskPersistenceResult


def add_task(
    bot_id: str, task: PersistedTaskPayload
) -> tuple[TaskPersistenceResult, bool]:
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
            "SELECT id, status, jira_key FROM tasks WHERE bot_id=? AND fingerprint=?",
            (bot_id, fingerprint),
        ).fetchone()
        if row is None:
            raise RuntimeError("Task was not persisted")
        result: TaskPersistenceResult = {
            "id": row["id"],
            "status": row["status"],
            "jira_key": row["jira_key"],
        }
        return result, inserted


def set_task_status(
    task_id: int, status: str, jira_key: str | None = None, error: str | None = None
) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE tasks SET status=?, jira_key=COALESCE(?, jira_key), error=? WHERE id=?",
            (status, jira_key, error, task_id),
        )
