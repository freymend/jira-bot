from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import db
from app.config import settings
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

    def test_final_webhook_queues_once_and_expired_lease_is_reclaimed(self) -> None:
        payload = {"trigger": "bot.state_change", "bot_id": "bot-1"}
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

    def test_failed_job_retries_then_becomes_terminal(self) -> None:
        db.create_meeting("bot-2", "https://meet.google.com/abc-defg-hij")
        self.assertEqual(db.enqueue_finalization("bot-2"), "queued")

        first = db.claim_next_job("worker-1")
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(
            db.fail_job(
                int(first["id"]),
                "worker-1",
                "temporary failure",
                attempts=int(first["attempts"]),
                max_attempts=2,
            ),
            "queued",
        )
        with db._connect() as connection:
            connection.execute(
                "UPDATE background_jobs SET available_at=0 WHERE id=?",
                (first["id"],),
            )

        second = db.claim_next_job("worker-2")
        self.assertIsNotNone(second)
        assert second is not None
        self.assertEqual(
            db.fail_job(
                int(second["id"]),
                "worker-2",
                "permanent failure",
                attempts=int(second["attempts"]),
                max_attempts=2,
            ),
            "failed",
        )

    def test_repeated_worker_crashes_exhaust_the_lease_retry_limit(self) -> None:
        db.create_meeting("bot-lease", "https://meet.google.com/abc-defg-hij")
        db.enqueue_finalization("bot-lease")

        for attempt in range(1, 6):
            job = db.claim_next_job(
                f"worker-{attempt}", lease_seconds=0, max_attempts=5
            )
            self.assertIsNotNone(job)
            assert job is not None
            self.assertEqual(job["attempts"], attempt)

        self.assertIsNone(
            db.claim_next_job("worker-final", lease_seconds=60, max_attempts=5)
        )
        with db._connect() as connection:
            job_status = connection.execute(
                "SELECT status FROM background_jobs WHERE bot_id='bot-lease'"
            ).fetchone()["status"]
            meeting_status = connection.execute(
                "SELECT status FROM meetings WHERE bot_id='bot-lease'"
            ).fetchone()["status"]
        self.assertEqual(job_status, "failed")
        self.assertEqual(meeting_status, "error")

    def test_readiness_requires_a_recent_worker_heartbeat(self) -> None:
        self.assertEqual(
            db.health_checks(), {"database": True, "worker": False}
        )
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