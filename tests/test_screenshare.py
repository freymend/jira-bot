from __future__ import annotations

import base64
import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app import db
from app.config import settings
from app.integrations.attendee import AttendeeClient, AttendeeClientProtocol
from app.integrations.jira import JiraClient
from app.integrations.screenshare import add_screenshare_context, summarize_screen_frame
from app.main import app
from app.schemas import (
    AttendeeBotResponse,
    ChatMessage,
    CreatedAttendeeBot,
    JSONValue,
    TaskPayload,
    TaskProposal,
)


class FakeAttendee(AttendeeClientProtocol):
    def __init__(self) -> None:
        self.create_bot_calls: list[tuple[str, bool | None]] = []
        self.close_calls = 0

    async def create_bot(
        self,
        meeting_url: str,
        *,
        screen_share_capture: bool | None = None,
    ) -> CreatedAttendeeBot:
        self.create_bot_calls.append((meeting_url, screen_share_capture))
        return {"id": "bot-created", "state": "waiting"}

    async def leave_bot(self, bot_id: str) -> JSONValue:
        return {"ok": True}

    async def get_bot(self, bot_id: str) -> AttendeeBotResponse:
        return {"id": bot_id, "state": "joined_recording"}

    async def get_transcript(self, bot_id: str) -> JSONValue:
        return []

    async def get_chat_messages(self, bot_id: str) -> list[ChatMessage]:
        return []

    async def close(self) -> None:
        self.close_calls += 1


class ScreenShareStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_settings = replace(
            settings,
            database_path=str(Path(self.temp_dir.name) / "screenshare.sqlite3"),
            screenshare_sample_interval_seconds=10,
            screenshare_retention_hours=24,
            screenshare_max_frames_per_meeting=2,
        )
        self.settings_patch = patch("app.db.settings", self.db_settings)
        self.settings_patch.start()
        db.init_db()
        db.create_meeting(
            "bot-screenshare",
            "https://meet.google.com/abc-defg-hij",
            screenshare_capture_enabled=True,
        )
        self.recording_start_ms = 1_700_000_000_000
        db.set_recording_started_at("bot-screenshare", self.recording_start_ms)
        self.jpeg = b"\xff\xd8sample-image\xff\xd9"

    def tearDown(self) -> None:
        self.settings_patch.stop()
        self.temp_dir.cleanup()

    def test_frames_are_time_linked_sampled_and_bounded(self) -> None:
        first_id = db.store_screen_frame(
            "bot-screenshare",
            self.jpeg,
            received_at_ms=self.recording_start_ms + 1_000,
            participant_uuid="participant-1",
        )
        self.assertIsNotNone(first_id)
        self.assertIsNone(
            db.store_screen_frame(
                "bot-screenshare",
                self.jpeg,
                received_at_ms=self.recording_start_ms + 9_000,
                participant_uuid="participant-1",
            )
        )
        second_id = db.store_screen_frame(
            "bot-screenshare",
            self.jpeg,
            received_at_ms=self.recording_start_ms + 11_000,
            participant_uuid="participant-1",
        )
        self.assertIsNotNone(second_id)
        self.assertIsNone(
            db.store_screen_frame(
                "bot-screenshare",
                self.jpeg,
                received_at_ms=self.recording_start_ms + 21_000,
                participant_uuid="participant-1",
            )
        )
        frames = db.get_screen_frames_near(
            "bot-screenshare", 1_000, window_ms=0, limit=2
        )
        self.assertEqual([frame["timestamp_ms"] for frame in frames], [1_000])
        self.assertEqual(frames[0]["time_source"], "recording_state")
        self.assertEqual(frames[0]["frame_data"], self.jpeg)

    def test_expired_frames_are_removed(self) -> None:
        frame_id = db.store_screen_frame(
            "bot-screenshare",
            self.jpeg,
            received_at_ms=self.recording_start_ms + 1_000,
            participant_uuid=None,
        )
        assert frame_id is not None
        with db._connect() as connection:
            connection.execute(
                "UPDATE screen_frames SET created_at=? WHERE id=?",
                (time.time() - 7200, frame_id),
            )
        retention_settings = replace(self.db_settings, screenshare_retention_hours=1)
        with patch("app.db.settings", retention_settings):
            self.assertEqual(db.delete_expired_screen_frames(), 1)
        self.assertIsNone(db.get_screen_frame("bot-screenshare", frame_id))

    def test_recording_webhook_rebases_early_frames(self) -> None:
        db.create_meeting("bot-late-start", None, screenshare_capture_enabled=True)
        frame_id = db.store_screen_frame(
            "bot-late-start",
            self.jpeg,
            received_at_ms=self.recording_start_ms + 1_500,
            participant_uuid=None,
        )
        self.assertIsNotNone(frame_id)
        assert frame_id is not None
        db.set_recording_started_at("bot-late-start", self.recording_start_ms + 500)
        frame = db.get_screen_frame("bot-late-start", int(frame_id))
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame["timestamp_ms"], 1_000)
        self.assertEqual(frame["time_source"], "recording_state")

    def test_task_records_visual_source_metadata(self) -> None:
        frame_id = db.store_screen_frame(
            "bot-screenshare",
            self.jpeg,
            received_at_ms=self.recording_start_ms + 1_000,
            participant_uuid=None,
        )
        assert frame_id is not None
        db.add_task(
            "bot-screenshare",
            {
                "summary": "Fix the login form",
                "description": "",
                "evidence_quote": "I will fix the login form.",
                "timestamp_ms": 1_000,
                "confidence": 0.95,
                "visual_evidence": [
                    {
                        "frame_id": frame_id,
                        "timestamp_ms": 1_000,
                        "time_source": "recording_state",
                        "summary": "The login form shows a validation error.",
                    }
                ],
            },
        )
        meeting = db.get_meeting("bot-screenshare")
        assert meeting is not None
        self.assertEqual(meeting["screenshare_capture_enabled"], 1)
        self.assertEqual(
            meeting["tasks"][0]["visual_evidence"][0]["frame_id"], frame_id
        )


class ScreenShareContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_frame_summary_sends_image_as_local_model_input(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"summary":"The checkout page shows a tax toggle."}'
                    )
                )
            ]
        )
        create = AsyncMock(return_value=response)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        frame = {"id": 12, "frame_data": b"\xff\xd8test\xff\xd9"}
        with (
            patch(
                "app.integrations.screenshare.settings",
                replace(settings, screenshare_vision_enabled=True),
            ),
            patch("app.integrations.screenshare.db.save_screen_frame_summary") as save,
        ):
            summary = await summarize_screen_frame(client, frame)  # type: ignore[arg-type]
        self.assertEqual(summary, "The checkout page shows a tax toggle.")
        save.assert_called_once_with(12, summary)
        assert create.await_args is not None
        messages = create.await_args.kwargs["messages"]
        image_part = messages[1]["content"][1]
        self.assertTrue(
            image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
        )
        self.assertIn("untrusted data", messages[0]["content"])

    async def test_task_context_uses_nearby_frame_without_changing_audio_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_settings = replace(
                settings,
                database_path=str(Path(directory) / "context.sqlite3"),
                screenshare_sample_interval_seconds=1,
                screenshare_max_frames_per_meeting=5,
            )
            with patch("app.db.settings", db_settings):
                db.init_db()
                db.create_meeting("bot-context", None, screenshare_capture_enabled=True)
                start_ms = 1_700_000_000_000
                db.set_recording_started_at("bot-context", start_ms)
                frame_id = db.store_screen_frame(
                    "bot-context",
                    b"\xff\xd8image\xff\xd9",
                    received_at_ms=start_ms + 5_000,
                    participant_uuid=None,
                )
            assert frame_id is not None
            close = AsyncMock()
            client = MagicMock()
            client.close = close
            task = TaskProposal(
                summary="Update the checkout design",
                evidence_quote="I will update the checkout design.",
                timestamp_ms=5_000,
                confidence=0.95,
            )
            context_settings = replace(
                settings,
                screenshare_capture_enabled=True,
                screenshare_context_window_seconds=30,
            )
            with (
                patch("app.db.settings", db_settings),
                patch("app.integrations.screenshare.settings", context_settings),
                patch("app.integrations.screenshare.AsyncOpenAI", return_value=client),
                patch(
                    "app.integrations.screenshare.summarize_screen_frame",
                    new=AsyncMock(return_value="A checkout prototype is on screen."),
                ),
            ):
                enriched = await add_screenshare_context([task], "bot-context")
            self.assertEqual(enriched[0].evidence_quote, task.evidence_quote)
            self.assertIn("Screen-share context", enriched[0].description)
            self.assertEqual(enriched[0].visual_evidence[0].frame_id, frame_id)
            self.assertEqual(
                enriched[0].visual_evidence[0].summary,
                "A checkout prototype is on screen.",
            )
            close.assert_awaited_once()


class ScreenShareWebSocketTests(unittest.TestCase):
    def test_authenticated_websocket_persists_only_screenshare_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app_settings = replace(
                settings,
                app_api_token="test-api-token",
                database_path=str(Path(directory) / "websocket.sqlite3"),
                screenshare_capture_enabled=True,
                screenshare_websocket_token="s" * 40,
                screenshare_sample_interval_seconds=1,
                screenshare_retention_hours=24,
                screenshare_max_frames_per_meeting=5,
                google_calendar_auto_join_enabled=False,
                meeting_auto_leave_enabled=False,
            )
            with patch("app.db.settings", app_settings):
                db.init_db()
            attendee = FakeAttendee()
            with (
                patch("app.main.settings", app_settings),
                patch("app.db.settings", app_settings),
                patch("app.main.attendee", new=attendee),
                TestClient(app) as client,
            ):
                db.create_meeting("bot-ws", None, screenshare_capture_enabled=True)
                db.create_meeting("bot-audio-only", None)
                with client.websocket_connect(
                    "/webhooks/attendee/video?token="
                    + app_settings.screenshare_websocket_token
                ) as websocket:
                    websocket.send_json(
                        {
                            "bot_id": "bot-ws",
                            "trigger": "realtime_video.per_participant",
                            "data": {
                                "participant_uuid": "participant-1",
                                "frame": base64.b64encode(
                                    b"\xff\xd8frame\xff\xd9"
                                ).decode("ascii"),
                                "format": "jpeg",
                                "source": "screenshare",
                            },
                        }
                    )
                    websocket.send_json(
                        {
                            "bot_id": "bot-ws",
                            "trigger": "realtime_video.per_participant",
                            "data": {
                                "frame": base64.b64encode(
                                    b"\xff\xd8webcam\xff\xd9"
                                ).decode("ascii"),
                                "format": "jpeg",
                                "source": "webcam",
                            },
                        }
                    )
                    websocket.send_json(
                        {
                            "bot_id": "bot-audio-only",
                            "trigger": "realtime_video.per_participant",
                            "data": {
                                "frame": base64.b64encode(
                                    b"\xff\xd8disabled\xff\xd9"
                                ).decode("ascii"),
                                "format": "jpeg",
                                "source": "screenshare",
                            },
                        }
                    )
                frames = db.get_screen_frames_near("bot-ws", 0, window_ms=0)
                self.assertEqual(len(frames), 1)
                self.assertEqual(
                    db.get_screen_frames_near("bot-audio-only", 0, window_ms=0), []
                )
                response = client.get(
                    f"/meetings/bot-ws/screen-frames/{frames[0]['id']}",
                    headers={"Authorization": "Bearer test-api-token"},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, b"\xff\xd8frame\xff\xd9")

    def test_api_rejects_capture_request_when_globally_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app_settings = replace(
                settings,
                app_api_token="test-api-token",
                database_path=str(Path(directory) / "disabled.sqlite3"),
                screenshare_capture_enabled=False,
                google_calendar_auto_join_enabled=False,
                meeting_auto_leave_enabled=False,
            )
            with patch("app.db.settings", app_settings):
                db.init_db()
            attendee = FakeAttendee()
            with (
                patch("app.main.settings", app_settings),
                patch("app.db.settings", app_settings),
                patch("app.main.attendee", new=attendee),
                TestClient(app) as client,
            ):
                response = client.post(
                    "/meetings",
                    headers={"Authorization": "Bearer test-api-token"},
                    json={
                        "meeting_url": "https://meet.google.com/abc-defg-hij",
                        "screen_share_capture": True,
                    },
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(attendee.create_bot_calls, [])

    def test_websocket_rejects_wrong_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app_settings = replace(
                settings,
                database_path=str(Path(directory) / "rejected.sqlite3"),
                screenshare_capture_enabled=True,
                screenshare_websocket_token="s" * 40,
                google_calendar_auto_join_enabled=False,
                meeting_auto_leave_enabled=False,
            )
            with patch("app.db.settings", app_settings):
                db.init_db()
            attendee = FakeAttendee()
            with (
                patch("app.main.settings", app_settings),
                patch("app.db.settings", app_settings),
                patch("app.main.attendee", new=attendee),
                TestClient(app) as client,
            ):
                with self.assertRaises(WebSocketDisconnect):
                    with client.websocket_connect(
                        "/webhooks/attendee/video?token=wrong"
                    ):
                        pass


class ScreenShareBotConfigurationTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_bot_configures_screen_only_websocket_when_enabled(
        self,
    ) -> None:
        configuration = replace(
            settings,
            attendee_api_key="attendee-key",
            app_public_url="https://assistant.example",
            mlx_transcriber_api_key="transcriber-key",
            screenshare_capture_enabled=True,
            screenshare_websocket_token="t" * 40,
            screenshare_resolution="720p",
        )
        response = MagicMock()
        response.json.return_value = {"id": "bot-1"}
        client = AttendeeClient()
        post = AsyncMock(return_value=response)
        client.client.post = post  # type: ignore[method-assign]
        with patch("app.integrations.attendee.settings", configuration):
            await client.create_bot("https://meet.google.com/abc-defg-hij")
        assert post.await_args is not None
        request_body = post.await_args.kwargs["json"]
        self.assertIn(
            "participant_events.join_leave",
            request_body["webhooks"][0]["triggers"],
        )
        video = request_body["websocket_settings"]["per_participant_video"]
        self.assertEqual(video["webcam_resolution"], "none")
        self.assertEqual(video["screenshare_resolution"], "720p")
        self.assertTrue(video["url"].startswith("wss://assistant.example/"))
        self.assertIn(configuration.screenshare_websocket_token, video["url"])
        await client.close()

    async def test_meeting_can_disable_globally_enabled_capture(self) -> None:
        configuration = replace(
            settings,
            attendee_api_key="attendee-key",
            app_public_url="http://api:5005",
            mlx_transcriber_api_key="transcriber-key",
            screenshare_capture_enabled=True,
            screenshare_websocket_token="t" * 40,
        )
        response = MagicMock()
        response.json.return_value = {"id": "bot-audio-only"}
        client = AttendeeClient()
        post = AsyncMock(return_value=response)
        client.client.post = post  # type: ignore[method-assign]
        with patch("app.integrations.attendee.settings", configuration):
            await client.create_bot(
                "https://meet.google.com/abc-defg-hij", screen_share_capture=False
            )
        assert post.await_args is not None
        self.assertNotIn("websocket_settings", post.await_args.kwargs["json"])
        await client.close()

    async def test_capture_requires_public_https_endpoint(self) -> None:
        configuration = replace(
            settings,
            attendee_api_key="attendee-key",
            app_public_url="http://api:5005",
            screenshare_capture_enabled=True,
            screenshare_websocket_token="t" * 40,
        )
        client = AttendeeClient()
        post = AsyncMock()
        client.client.post = post  # type: ignore[method-assign]
        with (
            patch("app.integrations.attendee.settings", configuration),
            self.assertRaisesRegex(RuntimeError, "requires APP_PUBLIC_URL"),
        ):
            await client.create_bot("https://meet.google.com/abc-defg-hij")
        post.assert_not_awaited()
        await client.close()


class JiraScreenShareEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_visual_summary_and_opt_in_image_attachment_reach_jira(self) -> None:
        requests: list[httpx.Request] = []

        async def handle(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.path.endswith("/attachments"):
                return httpx.Response(200, json=[])
            return httpx.Response(201, json={"key": "TEST-42"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        configuration = replace(
            settings,
            jira_base_url="https://jira.example",
            jira_email="bot@example.test",
            jira_api_token="jira-token",
            jira_project_key="TEST",
            jira_attach_screenshare_evidence=True,
        )
        task: TaskPayload = {
            "summary": "Update checkout page",
            "evidence_quote": "I will update the checkout page.",
            "timestamp_ms": 15_000,
            "visual_evidence": [
                {
                    "frame_id": 9,
                    "timestamp_ms": 15_000,
                    "time_source": "recording_state",
                    "summary": "A tax toggle appears on the checkout page.",
                }
            ],
        }
        with (
            patch("app.integrations.jira.settings", configuration),
            patch("app.integrations.jira.httpx.AsyncClient", return_value=client),
        ):
            key = await JiraClient().create_issue(
                task,
                "https://meet.google.com/abc-defg-hij",
                screen_frames=[
                    {
                        "id": 9,
                        "timestamp_ms": 15_000,
                        "frame_data": b"\xff\xd8image\xff\xd9",
                    }
                ],
            )
        self.assertEqual(key, "TEST-42")
        self.assertEqual(len(requests), 2)
        issue_body = json.loads(requests[0].content)
        description = json.dumps(issue_body["fields"]["description"])
        self.assertIn("Screen-share context", description)
        self.assertIn("tax toggle", description)
        self.assertIn("frame 9", description)
        self.assertIn("X-Atlassian-Token", requests[1].headers)
        self.assertIn(b"screenshare-frame-9-15.jpg", requests[1].content)
        self.assertIn(b"\xff\xd8image\xff\xd9", requests[1].content)

    async def test_jira_rejection_logs_structured_error_details(self) -> None:
        async def handle(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={
                    "errorMessages": [],
                    "errors": {"duedate": "Invalid date format"},
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        configuration = replace(
            settings,
            jira_base_url="https://jira.example",
            jira_email="bot@example.test",
            jira_api_token="jira-token",
            jira_project_key="TEST",
        )
        task: TaskPayload = {
            "summary": "Update checkout page",
            "evidence_quote": "I will update the checkout page.",
        }
        with (
            patch("app.integrations.jira.settings", configuration),
            patch("app.integrations.jira.httpx.AsyncClient", return_value=client),
            self.assertLogs("jira-bot.jira", level="ERROR") as captured,
        ):
            with self.assertRaises(httpx.HTTPStatusError):
                await JiraClient().create_issue(task, None)

        self.assertIn("HTTP 400", captured.output[0])
        self.assertIn("duedate", captured.output[0])
        self.assertIn("Invalid date format", captured.output[0])


if __name__ == "__main__":
    unittest.main()
