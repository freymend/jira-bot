from __future__ import annotations

import sqlite3
import tempfile
import time
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import db
from app.config import settings
from app.schemas import JSONObject
from app.main import app


class DurableJobTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "test.sqlite3"
        self.settings_patch = patch(
            "app.db.settings", replace(settings, database_path=str(database_path))
        )
        self.settings_patch.start()
        db.init_db()

    def tearDown(self) -> None:
        self.settings_patch.stop()
        self.temp_dir.cleanup()

    def test_migrations_upgrade_legacy_database_and_run_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "legacy.sqlite3"
            with closing(sqlite3.connect(database_path)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE meetings (
                        bot_id TEXT PRIMARY KEY,
                        meeting_url TEXT,
                        created_at TEXT NOT NULL,
                        transcription_state TEXT NOT NULL DEFAULT 'not_started',
                        status TEXT NOT NULL DEFAULT 'waiting',
                        transcript_json TEXT,
                        error TEXT
                    );
                    CREATE TABLE tasks (
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
                    INSERT INTO meetings(bot_id, created_at, status)
                    VALUES ('legacy-bot', '2025-01-01T00:00:00+00:00', 'processing');
                    """
                )

            with patch(
                "app.db.settings",
                replace(settings, database_path=str(database_path)),
            ):
                db.init_db()
                db.init_db()
                with db._connect() as connection:
                    meeting_columns = {
                        row["name"]
                        for row in connection.execute("PRAGMA table_info(meetings)")
                    }
                    task_columns = {
                        row["name"]
                        for row in connection.execute("PRAGMA table_info(tasks)")
                    }
                    versions = [
                        row["version"]
                        for row in connection.execute(
                            "SELECT version FROM schema_migrations ORDER BY version"
                        )
                    ]
                    jobs = connection.execute(
                        "SELECT COUNT(*) AS count FROM background_jobs "
                        "WHERE bot_id='legacy-bot'"
                    ).fetchone()["count"]
                    cache_table = connection.execute(
                        """SELECT name FROM sqlite_master
                           WHERE type='table' AND name='task_extraction_cache'"""
                    ).fetchone()
                    calendar_join_table = connection.execute(
                        """SELECT name FROM sqlite_master
                           WHERE type='table' AND name='calendar_auto_join_requests'"""
                    ).fetchone()
                    calendar_sync_tables = {
                        row["name"]
                        for row in connection.execute(
                            """SELECT name FROM sqlite_master WHERE type='table'
                               AND name IN ('calendar_sync_state',
                                            'calendar_watch_channels',
                                            'calendar_scheduled_events')"""
                        )
                    }
                    participant_tables = {
                        row["name"]
                        for row in connection.execute(
                            """SELECT name FROM sqlite_master WHERE type='table'
                               AND name IN ('meeting_participants',
                                            'meeting_participant_events')"""
                        )
                    }

            self.assertIn("bot_state", meeting_columns)
            self.assertTrue(
                {
                    "scheduled_end_at",
                    "last_transcript_received_at",
                    "leave_status",
                    "leave_requested_at",
                    "leave_reason",
                    "participant_tracking_enabled",
                    "participant_presence_initialized",
                    "last_participant_event_at",
                    "leave_attempt_count",
                    "leave_watchdog_checked_at",
                    "leave_stuck_at",
                }.issubset(meeting_columns)
            )
            self.assertTrue(
                {
                    "owner_evidence_quote",
                    "owner_timestamp_ms",
                    "context_evidence_json",
                }.issubset(task_columns)
            )
            self.assertEqual(versions, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            self.assertEqual(jobs, 1)
            self.assertIsNotNone(cache_table)
            self.assertIsNotNone(calendar_join_table)
            self.assertEqual(
                calendar_sync_tables,
                {
                    "calendar_sync_state",
                    "calendar_watch_channels",
                    "calendar_scheduled_events",
                },
            )
            self.assertEqual(
                participant_tables,
                {"meeting_participants", "meeting_participant_events"},
            )

    def test_final_webhook_queues_once_and_expired_lease_is_reclaimed(self) -> None:
        payload: JSONObject = {"trigger": "bot.state_change", "bot_id": "bot-1"}
        self.assertTrue(
            db.record_webhook(
                "event-1",
                "bot-1",
                "bot.state_change",
                payload,
                enqueue_finalization=True,
            )
        )
        self.assertFalse(
            db.record_webhook(
                "event-1",
                "bot-1",
                "bot.state_change",
                payload,
                enqueue_finalization=True,
            )
        )

        first_claim = db.claim_next_job("worker-1", lease_seconds=0)
        self.assertIsNotNone(first_claim)
        assert first_claim is not None
        self.assertEqual(first_claim["attempts"], 1)

        reclaimed = db.claim_next_job("worker-2", lease_seconds=60)
        self.assertIsNotNone(reclaimed)
        assert reclaimed is not None
        self.assertEqual(reclaimed["id"], first_claim["id"])
        self.assertEqual(reclaimed["attempts"], 2)
        self.assertFalse(db.finish_job(int(first_claim["id"]), "worker-1"))
        self.assertTrue(db.finish_job(int(reclaimed["id"]), "worker-2"))

    def test_transient_job_retries_beyond_the_previous_attempt_limit(self) -> None:
        db.create_meeting("bot-2", "https://meet.google.com/abc-defg-hij")
        self.assertEqual(db.enqueue_finalization("bot-2"), "queued")

        for attempt in range(1, 8):
            with db._connect() as connection:
                connection.execute(
                    "UPDATE background_jobs SET available_at=0 WHERE bot_id='bot-2'"
                )
            job = db.claim_next_job(f"worker-{attempt}")
            self.assertIsNotNone(job)
            assert job is not None
            self.assertEqual(job["attempts"], attempt)
            self.assertEqual(
                db.fail_job(
                    int(job["id"]),
                    f"worker-{attempt}",
                    "temporary network failure",
                    attempts=int(job["attempts"]),
                    retryable=True,
                ),
                "queued",
            )

        with db._connect() as connection:
            connection.execute(
                "UPDATE background_jobs SET available_at=0 WHERE bot_id='bot-2'"
            )
        recovered = db.claim_next_job("worker-recovered")
        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual(recovered["attempts"], 8)
        self.assertTrue(db.finish_job(int(recovered["id"]), "worker-recovered"))

    def test_permanent_job_failure_is_terminal(self) -> None:
        db.create_meeting("bot-permanent", "https://meet.google.com/abc-defg-hij")
        db.enqueue_finalization("bot-permanent")
        job = db.claim_next_job("worker-1")
        self.assertIsNotNone(job)
        assert job is not None

        self.assertEqual(
            db.fail_job(
                int(job["id"]),
                "worker-1",
                "invalid configuration",
                attempts=int(job["attempts"]),
                retryable=False,
            ),
            "failed",
        )
        self.assertIsNone(db.claim_next_job("worker-2"))

    def test_retry_after_sets_the_minimum_next_attempt_time(self) -> None:
        db.create_meeting("bot-retry-after", "https://meet.google.com/abc-defg-hij")
        db.enqueue_finalization("bot-retry-after")
        job = db.claim_next_job("worker-1")
        self.assertIsNotNone(job)
        assert job is not None

        now = time.time()
        with patch("app.db.random.uniform", return_value=1.0):
            db.fail_job(
                int(job["id"]),
                "worker-1",
                "rate limited",
                attempts=int(job["attempts"]),
                retryable=True,
                retry_after_seconds=120,
            )
        with db._connect() as connection:
            available_at = connection.execute(
                "SELECT available_at FROM background_jobs WHERE id=?", (job["id"],)
            ).fetchone()["available_at"]

        self.assertGreaterEqual(available_at, now + 120)
        self.assertLess(available_at, now + 121)

    def test_repeated_worker_crashes_remain_reclaimable(self) -> None:
        db.create_meeting("bot-lease", "https://meet.google.com/abc-defg-hij")
        db.enqueue_finalization("bot-lease")

        for attempt in range(1, 8):
            job = db.claim_next_job(f"worker-{attempt}", lease_seconds=0)
            self.assertIsNotNone(job)
            assert job is not None
            self.assertEqual(job["attempts"], attempt)

        recovered = db.claim_next_job("worker-recovered", lease_seconds=60)
        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual(recovered["attempts"], 8)
        self.assertTrue(db.finish_job(int(recovered["id"]), "worker-recovered"))
        with db._connect() as connection:
            meeting_status = connection.execute(
                "SELECT status FROM meetings WHERE bot_id='bot-lease'"
            ).fetchone()["status"]
        self.assertEqual(meeting_status, "retrying")

    def test_readiness_requires_a_recent_worker_heartbeat(self) -> None:
        self.assertEqual(db.health_checks(), {"database": True, "worker": False})
        db.record_worker_heartbeat("worker-1")
        self.assertEqual(db.health_checks(), {"database": True, "worker": True})

    def test_http_readiness_returns_service_unavailable_without_worker(self) -> None:
        client = TestClient(app)
        try:
            response = client.get("/health")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["checks"]["worker"], "unavailable")

            db.record_worker_heartbeat("worker-1")
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "ok")
        finally:
            client.close()

    def test_dry_run_job_can_be_requeued_for_reprocessing(self) -> None:
        db.create_meeting("bot-3", "https://meet.google.com/abc-defg-hij")
        self.assertEqual(db.enqueue_finalization("bot-3"), "queued")
        job = db.claim_next_job("worker-1")
        self.assertIsNotNone(job)
        assert job is not None
        self.assertTrue(db.finish_job(int(job["id"]), "worker-1"))
        db.set_meeting_status("bot-3", "dry_run")

        self.assertEqual(
            db.enqueue_finalization("bot-3", allow_reprocess=True), "queued"
        )


if __name__ == "__main__":
    unittest.main()
