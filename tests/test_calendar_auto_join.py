from __future__ import annotations

import asyncio
import os
import tempfile
import time as system_time
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch
from zoneinfo import ZoneInfo

import httpx
from fastapi.testclient import TestClient

from app import db
from app.main import app as api_app
from app.calendar_scheduler import (
    _ensure_calendar_watch,
    accept_calendar_notification,
    request_due_meetings,
    scheduled_meet,
    set_calendar_sync_wakeup,
    sync_calendar_events,
)
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


class FakeWatchClient:
    def __init__(self) -> None:
        self.callback_url: str | None = None
        self.stopped: list[str] = []

    async def watch_events(
        self,
        channel_id: str,
        callback_url: str,
        channel_token: str,
        expiration_ms: int,
    ) -> dict[str, Any]:
        self.callback_url = callback_url
        return {"resourceId": "resource-1", "expiration": expiration_ms}

    async def stop_watching(self, channel_id: str, resource_id: str) -> None:
        self.stopped.append(channel_id)


class FakeSyncClient:
    def __init__(self, event_pages: list[list[dict[str, Any]]]) -> None:
        self.event_pages = event_pages
        self.tokens: list[str | None] = []

    async def sync_events(
        self, sync_token: str | None
    ) -> tuple[list[dict[str, Any]], str]:
        self.tokens.append(sync_token)
        return self.event_pages[len(self.tokens) - 1], f"sync-{len(self.tokens)}"


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
            "end": {"dateTime": (local_start + timedelta(hours=1)).isoformat()},
            "hangoutLink": "https://meet.google.com/abc-defg-hij",
        }

    def test_only_exact_title_organizer_time_and_meet_events_match(self) -> None:
        start = datetime(2026, 10, 13, 10, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        now = start.astimezone(UTC) - timedelta(minutes=4)
        event = self._event(start)

        candidate = scheduled_meet(event, now)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.event_end, "2026-10-13T18:00:00+00:00")
        self.assertIsNone(scheduled_meet({**event, "summary": "Another meeting"}, now))
        self.assertIsNone(
            scheduled_meet({**event, "organizer": {"email": "other@example.test"}}, now)
        )
        self.assertIsNone(scheduled_meet({**event, "status": "cancelled"}, now))
        self.assertIsNone(
            scheduled_meet({**event, "hangoutLink": "https://example.com/meeting"}, now)
        )
        late_start = start + timedelta(hours=1)
        self.assertIsNone(scheduled_meet(self._event(late_start), now))

    def test_schedule_uses_configured_meeting_timezone_not_host_timezone(self) -> None:
        if not hasattr(system_time, "tzset"):
            self.skipTest("Changing the host timezone requires time.tzset")

        event = self._event(datetime(2026, 10, 13, 17, 0, tzinfo=UTC))
        event["start"] = {
            "dateTime": "2026-10-13T17:00:00+00:00",
            "timeZone": "UTC",
        }
        now = datetime(2026, 10, 13, 16, 56, tzinfo=UTC)
        original_timezone = os.environ.get("TZ")
        try:
            for host_timezone in ("America/Los_Angeles", "Asia/Tokyo"):
                os.environ["TZ"] = host_timezone
                system_time.tzset()
                candidate = scheduled_meet(event, now)
                self.assertIsNotNone(candidate, host_timezone)
                assert candidate is not None
                self.assertEqual(candidate.event_start, "2026-10-13T17:00:00+00:00")
        finally:
            if original_timezone is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = original_timezone
            system_time.tzset()

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
        meeting = db.get_meeting("bot-calendar-1")
        self.assertIsNotNone(meeting)
        assert meeting is not None
        self.assertEqual(meeting["scheduled_end_at"], "2026-10-13T18:00:00+00:00")

    async def test_uncertain_create_is_not_automatically_retried(self) -> None:
        start = datetime(2026, 10, 13, 10, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        now = start.astimezone(UTC) - timedelta(minutes=4)
        attendee = FakeAttendee(fail=True)
        event = self._event(start)

        self.assertEqual(await request_due_meetings([event], attendee, now), 0)
        self.assertEqual(await request_due_meetings([event], attendee, now), 0)
        self.assertEqual(len(attendee.calls), 1)
        self.assertEqual(db.list_calendar_auto_joins()[0]["status"], "uncertain")

    def test_webhook_route_rejects_invalid_calendar_notifications(self) -> None:
        db.create_calendar_watch_channel(
            "channel-route", "primary", "route-secret", 2_000_000_000_000
        )
        db.activate_calendar_watch_channel(
            "channel-route", "resource-route", 2_000_000_000_000
        )
        app_settings = replace(settings, google_calendar_auto_join_enabled=False)
        with patch("app.main.settings", app_settings), TestClient(api_app) as client:
            headers = {
                "X-Goog-Resource-State": "exists",
                "X-Goog-Channel-ID": "channel-route",
                "X-Goog-Channel-Token": "wrong-secret",
                "X-Goog-Resource-ID": "resource-route",
            }
            self.assertEqual(
                client.post("/webhooks/google-calendar", headers=headers).status_code,
                401,
            )
            headers["X-Goog-Channel-Token"] = "route-secret"
            self.assertEqual(
                client.post("/webhooks/google-calendar", headers=headers).status_code,
                204,
            )

    async def test_calendar_push_validates_channel_and_wakes_sync(self) -> None:
        db.create_calendar_watch_channel(
            "channel-1", "primary", "unguessable-token", 2_000_000_000_000
        )
        db.activate_calendar_watch_channel("channel-1", "resource-1", 2_000_000_000_000)
        wakeup = asyncio.Event()
        set_calendar_sync_wakeup(wakeup)
        try:
            self.assertFalse(
                accept_calendar_notification("channel-1", "wrong-token", "resource-1")
            )
            self.assertFalse(wakeup.is_set())
            self.assertTrue(
                accept_calendar_notification(
                    "channel-1", "unguessable-token", "resource-1"
                )
            )
            self.assertTrue(wakeup.is_set())
        finally:
            set_calendar_sync_wakeup(None)

    async def test_watch_uses_public_webhook_and_persists_channel(self) -> None:
        public_settings = replace(
            settings,
            google_calendar_id="primary",
            app_public_url="https://calendar.example.test",
        )
        calendar = FakeWatchClient()
        with patch("app.calendar_scheduler.settings", public_settings):
            self.assertTrue(await _ensure_calendar_watch(calendar))

        self.assertEqual(
            calendar.callback_url,
            "https://calendar.example.test/webhooks/google-calendar",
        )
        channel = db.get_current_calendar_watch("primary")
        self.assertIsNotNone(channel)
        assert channel is not None
        self.assertEqual(channel["resource_id"], "resource-1")
        self.assertEqual(channel["status"], "active")

    async def test_expired_sync_token_triggers_a_safe_full_resync(self) -> None:
        db.apply_calendar_sync("primary", [], "expired-token", replace_all=True)
        event = self._event()

        class ExpiredSyncClient:
            def __init__(self) -> None:
                self.tokens: list[str | None] = []

            async def sync_events(
                self, sync_token: str | None
            ) -> tuple[list[dict[str, Any]], str]:
                self.tokens.append(sync_token)
                if sync_token:
                    request = httpx.Request("GET", "https://calendar.test/events")
                    response = httpx.Response(410, request=request)
                    raise httpx.HTTPStatusError(
                        "expired sync token", request=request, response=response
                    )
                return [event], "fresh-token"

        calendar = ExpiredSyncClient()
        self.assertEqual(await sync_calendar_events(calendar), 1)
        self.assertEqual(calendar.tokens, ["expired-token", None])
        self.assertEqual(db.get_calendar_sync_token("primary"), "fresh-token")

    async def test_push_sync_upserts_and_removes_occurrences(self) -> None:
        event = self._event()
        cancelled_event = {**event, "status": "cancelled"}
        calendar = FakeSyncClient([[event], [cancelled_event]])

        self.assertEqual(await sync_calendar_events(calendar), 1)
        self.assertEqual(db.get_calendar_sync_token("primary"), "sync-1")
        self.assertEqual(
            len(
                db.list_due_calendar_events(
                    "primary",
                    "2026-10-13T16:59:00+00:00",
                    "2026-10-13T17:01:00+00:00",
                )
            ),
            1,
        )
        self.assertEqual(await sync_calendar_events(calendar), 1)
        self.assertEqual(calendar.tokens, [None, "sync-1"])
        self.assertEqual(db.get_calendar_sync_token("primary"), "sync-2")
        self.assertEqual(
            db.list_due_calendar_events(
                "primary",
                "2026-10-13T16:59:00+00:00",
                "2026-10-13T17:01:00+00:00",
            ),
            [],
        )

    def test_incremental_sync_changes_and_token_commit_together(self) -> None:
        event = {
            "action": "upsert",
            "event_key": "occurrence-key",
            "calendar_event_id": "occurrence-1",
            "event_start": "2026-10-13T17:00:00+00:00",
            "event_end": "2026-10-13T18:00:00+00:00",
            "meeting_url": "https://meet.google.com/abc-defg-hij",
        }
        db.apply_calendar_sync("primary", [event], "sync-token-1", replace_all=True)
        self.assertEqual(db.get_calendar_sync_token("primary"), "sync-token-1")
        stored_events = db.list_due_calendar_events(
            "primary",
            "2026-10-13T16:59:00+00:00",
            "2026-10-13T17:01:00+00:00",
        )
        self.assertEqual(len(stored_events), 1)
        self.assertEqual(stored_events[0]["event_end"], "2026-10-13T18:00:00+00:00")

        db.apply_calendar_sync(
            "primary",
            [{"action": "delete", "event_key": "occurrence-key"}],
            "sync-token-2",
        )
        self.assertEqual(db.get_calendar_sync_token("primary"), "sync-token-2")
        self.assertEqual(
            db.list_due_calendar_events(
                "primary",
                "2026-10-13T16:59:00+00:00",
                "2026-10-13T17:01:00+00:00",
            ),
            [],
        )


class GoogleCalendarClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_forbidden_response_includes_google_api_reason(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "oauth2.googleapis.com":
                return httpx.Response(
                    200,
                    json={"access_token": "secret-access-token", "expires_in": 3600},
                    request=request,
                )
            return httpx.Response(
                403,
                json={
                    "error": {
                        "code": 403,
                        "message": "Calendar API is not enabled",
                        "errors": [{"reason": "accessNotConfigured"}],
                    }
                },
                request=request,
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        calendar = GoogleCalendarClient(client)
        try:
            with self.assertRaisesRegex(
                httpx.HTTPStatusError, "accessNotConfigured"
            ) as raised:
                await calendar.list_events(
                    datetime(2026, 10, 13, 16, 55, tzinfo=UTC),
                    datetime(2026, 10, 13, 17, 5, tzinfo=UTC),
                )
        finally:
            await client.aclose()
        self.assertNotIn("secret-access-token", str(raised.exception))

    async def test_full_and_incremental_sync_use_persisted_token_shape(self) -> None:
        event_requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "oauth2.googleapis.com":
                return httpx.Response(
                    200,
                    json={"access_token": "access-token", "expires_in": 3600},
                    request=request,
                )
            event_requests.append(request)
            if request.url.params.get("syncToken"):
                result = {"items": [{"id": "changed-event"}], "nextSyncToken": "next"}
            else:
                result = {
                    "items": [{"id": "initial-event"}],
                    "nextSyncToken": "initial",
                }
            return httpx.Response(200, json=result, request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        calendar = GoogleCalendarClient(client)
        try:
            initial_events, initial_token = await calendar.sync_events(None)
            changed_events, next_token = await calendar.sync_events(initial_token)
        finally:
            await client.aclose()

        self.assertEqual(initial_events, [{"id": "initial-event"}])
        self.assertEqual(changed_events, [{"id": "changed-event"}])
        self.assertEqual(next_token, "next")
        self.assertIn("timeMin", event_requests[0].url.params)
        self.assertNotIn("timeMax", event_requests[0].url.params)
        self.assertEqual(event_requests[1].url.params["syncToken"], "initial")
        self.assertNotIn("timeMin", event_requests[1].url.params)
        self.assertEqual(event_requests[1].url.params["singleEvents"], "true")

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
        self.assertEqual(requests[1].headers["Authorization"], "Bearer access-token")
        self.assertEqual(requests[1].url.params["singleEvents"], "true")


if __name__ == "__main__":
    unittest.main()
