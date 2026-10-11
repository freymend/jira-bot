from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch
from zoneinfo import ZoneInfo

import httpx

from app import db
from app.calendar_scheduler import request_due_meetings, scheduled_meet
from app.config import settings
from app.integrations.google_calendar import GoogleCalendarClient


class FakeAttendee:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.fail = fail

    async def create_bot(self, meeting_url: str) -> dict[str, str]:
        self.calls.append(meeting_url)
        if self.fail:
            raise RuntimeError("request outcome unknown")
        return {"id": "bot-calendar-1", "state": "waiting"}


class CalendarAutoJoinTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings_patch = patch(
            "app.calendar_scheduler.settings",
            replace(
                settings,
                google_calendar_id="primary",
                google_calendar_event_title="Daily Product Sync",
                google_calendar_organizer_email="host@example.test",
                google_calendar_timezone="America/Los_Angeles",
                google_calendar_start_time="10:00",
                google_calendar_join_lead_minutes=5,
                google_calendar_join_grace_minutes=2,
            ),
        )
        self.settings_patch.start()
        self.db_settings_patch = patch(
            "app.db.settings",
            replace(
                settings,
                database_path=str(Path(self.temp_dir.name) / "calendar.sqlite3"),
            ),
        )
        self.db_settings_patch.start()
        db.init_db()

    def tearDown(self) -> None:
        self.db_settings_patch.stop()
        self.settings_patch.stop()
        self.temp_dir.cleanup()

    def _event(self, start: datetime | None = None) -> dict[str, Any]:
        local_start = start or datetime(
            2026, 10, 13, 10, 0, tzinfo=ZoneInfo("America/Los_Angeles")
        )
        start_text = local_start.isoformat()
        return {
            "id": "occurrence-1",
            "recurringEventId": "daily-series",
            "originalStartTime": {"dateTime": start_text},
            "status": "confirmed",
            "summary": "Daily Product Sync",
            "organizer": {"email": "host@example.test"},
            "start": {"dateTime": start_text},
            "hangoutLink": "https://meet.google.com/abc-defg-hij",
        }

    def test_only_exact_title_organizer_time_and_meet_events_match(self) -> None:
        start = datetime(2026, 10, 13, 10, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        now = start.astimezone(UTC) - timedelta(minutes=4)
        event = self._event(start)

        self.assertIsNotNone(scheduled_meet(event, now))
        self.assertIsNone(
            scheduled_meet({**event, "summary": "Another meeting"}, now)
        )
        self.assertIsNone(
            scheduled_meet(
                {**event, "organizer": {"email": "other@example.test"}}, now
            )
        )
        self.assertIsNone(scheduled_meet({**event, "status": "cancelled"}, now))
        self.assertIsNone(
            scheduled_meet(
                {**event, "hangoutLink": "https://example.com/meeting"}, now
            )
        )
        late_start = start + timedelta(hours=1)
        self.assertIsNone(scheduled_meet(self._event(late_start), now))

    async def test_requests_each_occurrence_once_and_persists_bot(self) -> None:
        start = datetime(2026, 10, 13, 10, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        now = start.astimezone(UTC) - timedelta(minutes=4)
        attendee = FakeAttendee()
        event = self._event(start)

        self.assertEqual(await request_due_meetings([event], attendee, now), 1)
        self.assertEqual(await request_due_meetings([event], attendee, now), 0)
        self.assertEqual(attendee.calls, ["https://meet.google.com/abc-defg-hij"])
        rows = db.list_calendar_auto_joins()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "requested")
        self.assertEqual(rows[0]["bot_id"], "bot-calendar-1")
        self.assertIsNotNone(db.get_meeting("bot-calendar-1"))

    async def test_uncertain_create_is_not_automatically_retried(self) -> None:
        start = datetime(2026, 10, 13, 10, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        now = start.astimezone(UTC) - timedelta(minutes=4)
        attendee = FakeAttendee(fail=True)
        event = self._event(start)

        self.assertEqual(await request_due_meetings([event], attendee, now), 0)
        self.assertEqual(await request_due_meetings([event], attendee, now), 0)
        self.assertEqual(len(attendee.calls), 1)
        self.assertEqual(db.list_calendar_auto_joins()[0]["status"], "uncertain")


class GoogleCalendarClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_refreshes_token_and_reads_event_pages(self) -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.host == "oauth2.googleapis.com":
                return httpx.Response(
                    200,
                    json={"access_token": "access-token", "expires_in": 3600},
                    request=request,
                )
            return httpx.Response(
                200,
                json={"items": [{"id": "event-1"}]},
                request=request,
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        calendar = GoogleCalendarClient(client)
        try:
            time_min = datetime(2026, 10, 13, 16, 55, tzinfo=UTC)
            time_max = datetime(2026, 10, 13, 17, 5, tzinfo=UTC)
            events = await calendar.list_events(time_min, time_max)
        finally:
            await client.aclose()

        self.assertEqual(events, [{"id": "event-1"}])
        self.assertEqual(len(requests), 2)
        self.assertEqual(
            requests[1].headers["Authorization"], "Bearer access-token"
        )
        self.assertEqual(requests[1].url.params["singleEvents"], "true")


if __name__ == "__main__":
    unittest.main()
