from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, time as datetime_time, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from app import db
from app.calendar_scheduler import (
    request_due_meetings,
    request_stored_due_meetings,
)
from app.config import settings
from app.dashboard import server as dashboard
from app.main import _dashboard_calendar_schedule, dashboard_overview


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


class DashboardCalendarScheduleTests(unittest.IsolatedAsyncioTestCase):
    def _settings(self):
        return replace(
            settings,
            google_calendar_auto_join_enabled=True,
            google_calendar_event_title="Daily Sync",
            google_calendar_timezone="America/Los_Angeles",
            google_calendar_start_time="10:00",
            google_calendar_organizer_email="",
            google_calendar_id="primary",
        )

    async def test_polling_mode_reads_upcoming_events_from_calendar_api(self) -> None:
        timezone = ZoneInfo("America/Los_Angeles")
        now = datetime.now(UTC)
        start_day = now.astimezone(timezone).date() + timedelta(days=2)
        event_start = datetime.combine(start_day, datetime_time(10, 0), tzinfo=timezone)
        event_end = event_start + timedelta(minutes=30)
        raw_event = {
            "id": "event-1",
            "summary": "Daily Sync",
            "start": {"dateTime": event_start.isoformat()},
            "end": {"dateTime": event_end.isoformat()},
            "hangoutLink": "https://meet.google.com/abc-defg-hij",
        }
        calendar_client = AsyncMock()
        calendar_client.list_events = AsyncMock(return_value=[raw_event])
        calendar_client.close = AsyncMock()
        configured = self._settings()
        attendee_client = SimpleNamespace(
            create_bot=AsyncMock(return_value={"id": "bot-1"})
        )

        with (
            patch("app.main.settings", configured),
            patch("app.calendar_scheduler.settings", configured),
            patch("app.main.db.get_current_calendar_watch", return_value=None),
            patch("app.main.db.list_due_calendar_events", return_value=[]),
            patch("app.main.db.list_calendar_auto_joins", return_value=[]),
            patch("app.main.GoogleCalendarClient", return_value=calendar_client),
            patch("app.main._DASHBOARD_CALENDAR_CACHE", None),
            patch(
                "app.calendar_scheduler.db.claim_calendar_auto_join", return_value=True
            ),
            patch("app.calendar_scheduler.db.create_meeting"),
            patch("app.calendar_scheduler.db.finish_calendar_auto_join"),
        ):
            events, source, error = await _dashboard_calendar_schedule(now)
            due_now = datetime.fromisoformat(events[0]["event_start"]) - timedelta(
                minutes=configured.google_calendar_join_lead_minutes
            )
            requested = await request_due_meetings(
                [raw_event], attendee_client, due_now
            )

        self.assertEqual(source, "calendar_api")
        self.assertIsNone(error)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_id"], "event-1")
        self.assertEqual(events[0]["join_status"], "scheduled")
        self.assertEqual(requested, 1)
        attendee_client.create_bot.assert_awaited_once_with(
            "https://meet.google.com/abc-defg-hij"
        )
        calendar_client.list_events.assert_awaited_once_with(
            now, now + timedelta(days=14)
        )
        calendar_client.close.assert_awaited_once()

    async def test_push_mode_uses_synchronized_calendar_rows(self) -> None:
        now = datetime.now(UTC)
        row = {
            "event_key": "event-key-push",
            "calendar_event_id": "event-push",
            "event_start": (now + timedelta(days=1)).isoformat(),
            "event_end": (now + timedelta(days=1, minutes=30)).isoformat(),
            "meeting_url": "https://meet.google.com/abc-defg-hij",
        }
        watch = {"expiration_ms": int((now + timedelta(days=1)).timestamp() * 1000)}
        configured = self._settings()
        attendee_client = SimpleNamespace(
            create_bot=AsyncMock(return_value={"id": "bot-push"})
        )
        with (
            patch("app.main.settings", configured),
            patch("app.calendar_scheduler.settings", configured),
            patch("app.main.db.get_current_calendar_watch", return_value=watch),
            patch("app.main.db.list_due_calendar_events", return_value=[row]),
            patch("app.main.db.list_calendar_auto_joins", return_value=[]),
            patch("app.main.GoogleCalendarClient") as calendar_client,
            patch(
                "app.calendar_scheduler.db.claim_calendar_auto_join", return_value=True
            ),
            patch("app.calendar_scheduler.db.create_meeting"),
            patch("app.calendar_scheduler.db.finish_calendar_auto_join"),
        ):
            events, source, error = await _dashboard_calendar_schedule(now)
            event_start = datetime.fromisoformat(row["event_start"])
            due_now = event_start - timedelta(
                minutes=configured.google_calendar_join_lead_minutes
            )
            requested = await request_stored_due_meetings(attendee_client, due_now)

        self.assertEqual(source, "push_sync")
        self.assertIsNone(error)
        self.assertEqual(events[0]["event_id"], "event-push")
        self.assertEqual(requested, 1)
        attendee_client.create_bot.assert_awaited_once_with(row["meeting_url"])
        calendar_client.assert_not_called()

    async def test_overview_includes_calendar_events_and_source(self) -> None:
        event = {
            "event_id": "event-1",
            "event_title": "Daily Sync",
            "event_start": "2026-10-13T17:00:00+00:00",
            "event_end": "2026-10-13T17:30:00+00:00",
            "meeting_url": "https://meet.google.com/abc-defg-hij",
            "join_status": "requested",
            "bot_id": "bot-1",
            "error": None,
        }
        configured = self._settings()
        with (
            patch("app.main.settings", configured),
            patch(
                "app.main.db.health_checks",
                return_value={"database": True, "worker": True},
            ),
            patch("app.main.db.list_recent_meetings", return_value=[]),
            patch(
                "app.main._dashboard_calendar_schedule",
                new=AsyncMock(return_value=([event], "calendar_api", None)),
            ),
            patch("app.main._probe_service", new=AsyncMock(return_value="healthy")),
        ):
            result = await dashboard_overview()

        self.assertEqual(result["calendar"]["event_title_filter"], "Daily Sync")
        self.assertEqual(result["calendar"]["source"], "calendar_api")
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
