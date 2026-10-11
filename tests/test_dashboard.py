from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from app import db
from app.config import settings
from app.dashboard import server as dashboard
from app.main import dashboard_overview


class DashboardConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env_file = Path(self.temp_dir.name) / ".env"
        self.env_file.write_text(
            "APP_API_TOKEN=dashboard-secret\n"
            "JIRA_API_TOKEN=jira-secret\n"
            "LLM_MODEL=local-model\n",
            encoding="utf-8",
        )
        self.env_file.chmod(0o600)
        self.env_patch = patch("app.dashboard.server.ENV_FILE", self.env_file)
        self.env_patch.start()

    def tearDown(self) -> None:
        self.env_patch.stop()
        self.temp_dir.cleanup()

    def test_config_response_redacts_secrets(self) -> None:
        response = dashboard._current_config_response()
        self.assertNotIn("APP_API_TOKEN", response["values"])
        self.assertNotIn("JIRA_API_TOKEN", response["values"])
        self.assertTrue(response["secret_configured"]["APP_API_TOKEN"])
        self.assertTrue(response["secret_configured"]["JIRA_API_TOKEN"])
        serialized = repr(response)
        self.assertNotIn("dashboard-secret", serialized)
        self.assertNotIn("jira-secret", serialized)

    def test_config_update_preserves_unspecified_secrets_and_writes_private_file(
        self,
    ) -> None:
        result = dashboard._write_env_file({"JIRA_PROJECT_KEY": "OPS"}, [])
        self.assertIsNone(result)
        values = dashboard._env_values()
        self.assertEqual(values["APP_API_TOKEN"], "dashboard-secret")
        self.assertEqual(values["JIRA_API_TOKEN"], "jira-secret")
        self.assertEqual(values["JIRA_PROJECT_KEY"], "OPS")
        self.assertEqual(self.env_file.stat().st_mode & 0o777, 0o600)

    def test_config_api_authenticates_redacts_and_saves_allowlisted_values(
        self,
    ) -> None:
        with TestClient(dashboard.app, client=("127.0.0.1", 12345)) as client:
            headers = {"Authorization": "Bearer dashboard-secret"}
            read_response = client.get("/api/config", headers=headers)
            self.assertEqual(read_response.status_code, 200)
            self.assertNotIn("dashboard-secret", read_response.text)
            self.assertNotIn("jira-secret", read_response.text)

            update_response = client.put(
                "/api/config",
                headers=headers,
                json={"values": {"JIRA_PROJECT_KEY": "OPS"}, "clear": []},
            )
            self.assertEqual(update_response.status_code, 200)
            self.assertTrue(update_response.json()["restart_required"])

            invalid_response = client.put(
                "/api/config",
                headers=headers,
                json={"values": {"UNSUPPORTED_KEY": "value"}, "clear": []},
            )
            self.assertEqual(invalid_response.status_code, 400)

        self.assertEqual(dashboard._env_values()["JIRA_PROJECT_KEY"], "OPS")

    def test_dashboard_requires_loopback_and_valid_token(self) -> None:
        request = Request(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": "/api/config",
                "raw_path": b"/api/config",
                "query_string": b"",
                "headers": [],
                "client": ("127.0.0.1", 12345),
                "server": ("127.0.0.1", 5006),
            }
        )
        self.assertEqual(
            dashboard.require_dashboard_auth(request, "Bearer dashboard-secret"),
            "dashboard-secret",
        )
        with self.assertRaises(HTTPException) as error:
            dashboard.require_dashboard_auth(request, "Bearer wrong")
        self.assertEqual(error.exception.status_code, 401)

    def test_invalid_config_value_is_rejected_before_writing(self) -> None:
        with self.assertRaises(ValueError):
            dashboard._validate_candidate(
                {
                    **dashboard._env_values(),
                    "LLM_CONTEXT_TOKENS": "2048",
                }
            )
        self.assertNotIn("LLM_CONTEXT_TOKENS", dashboard._env_values())


class DashboardOverviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_overview_includes_scheduled_calendar_joins(self) -> None:
        event = {
            "calendar_event_id": "event-1",
            "event_start": "2026-10-12T17:00:00+00:00",
            "event_end": "2026-10-12T18:00:00+00:00",
            "meeting_url": "https://meet.google.com/abc-defg-hij",
        }
        attempt = {
            "calendar_event_id": "event-1",
            "event_start": event["event_start"],
            "status": "requested",
            "bot_id": "bot-1",
            "error": None,
        }
        configured = replace(
            settings,
            google_calendar_auto_join_enabled=True,
            google_calendar_event_title="Daily Sync",
            google_calendar_timezone="America/Los_Angeles",
            google_calendar_id="primary",
        )
        with (
            patch("app.main.settings", configured),
            patch(
                "app.main.db.health_checks",
                return_value={"database": True, "worker": True},
            ),
            patch("app.main.db.list_recent_meetings", return_value=[]),
            patch("app.main.db.list_due_calendar_events", return_value=[event]),
            patch("app.main.db.list_calendar_auto_joins", return_value=[attempt]),
            patch("app.main._probe_service", new=AsyncMock(return_value="healthy")),
        ):
            result = await dashboard_overview()

        self.assertEqual(result["calendar"]["event_title_filter"], "Daily Sync")
        self.assertEqual(len(result["calendar"]["upcoming_events"]), 1)
        scheduled = result["calendar"]["upcoming_events"][0]
        self.assertEqual(scheduled["join_status"], "requested")
        self.assertEqual(scheduled["bot_id"], "bot-1")


class DashboardMeetingListTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_patch = patch(
            "app.db.settings",
            replace(
                settings, database_path=str(Path(self.temp_dir.name) / "db.sqlite3")
            ),
        )
        self.database_patch.start()
        db.init_db()

    def tearDown(self) -> None:
        self.database_patch.stop()
        self.temp_dir.cleanup()

    def test_recent_meetings_include_task_counts(self) -> None:
        db.create_meeting("bot-dashboard", "https://meet.google.com/abc-defg-hij")
        db.add_task(
            "bot-dashboard",
            {
                "summary": "Add dashboard tests",
                "evidence_quote": "I will add dashboard tests.",
                "confidence": 0.95,
            },
        )

        meetings = db.list_recent_meetings()

        self.assertEqual(len(meetings), 1)
        self.assertEqual(meetings[0]["bot_id"], "bot-dashboard")
        self.assertEqual(meetings[0]["task_count"], 1)


if __name__ == "__main__":
    unittest.main()
