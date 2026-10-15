from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from app import db
from app.config import settings
from app.integrations.attendee import AttendeeClientProtocol
from app.integrations.jira import JiraClientProtocol
from app.pipeline import finalize_meeting
from app.schemas import (
    AttendeeBotResponse,
    ChatMessage,
    CreatedAttendeeBot,
    JSONValue,
    JiraScreenFrame,
    MeetingTaskRecord,
    PersistedTaskPayload,
    TaskExtraction,
    TaskPayload,
    TaskProposal,
)


class FakeAttendee(AttendeeClientProtocol):
    def __init__(
        self,
        transcript: JSONValue,
        chat_messages: list[ChatMessage] | None = None,
    ) -> None:
        self.transcript = transcript
        self.chat_messages = chat_messages or []
        self.transcript_call_count = 0

    async def create_bot(
        self,
        meeting_url: str,
        *,
        screen_share_capture: bool | None = None,
    ) -> CreatedAttendeeBot:
        return {"id": "fake-bot", "state": "waiting"}

    async def leave_bot(self, bot_id: str) -> JSONValue:
        return {"ok": True}

    async def get_bot(self, bot_id: str) -> AttendeeBotResponse:
        return {"id": bot_id, "state": "joined_recording"}

    async def get_transcript(self, bot_id: str) -> JSONValue:
        self.transcript_call_count += 1
        return self.transcript

    async def get_chat_messages(self, bot_id: str) -> list[ChatMessage]:
        return self.chat_messages

    async def close(self) -> None:
        return None


class FakeJiraClient(JiraClientProtocol):
    def __init__(self, outcomes: list[str | Exception]) -> None:
        self.outcomes = outcomes
        self.call_count = 0

    async def create_issue(
        self,
        task: TaskPayload | PersistedTaskPayload,
        meeting_url: str | None,
        *,
        screen_frames: list[JiraScreenFrame] | None = None,
    ) -> str:
        self.call_count += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class PipelineNetworkResilienceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "test.sqlite3"
        self.db_settings_patch = patch(
            "app.repositories.database.settings",
            replace(settings, database_path=str(database_path)),
        )
        self.db_settings_patch.start()
        self.pipeline_settings_patch = patch(
            "app.pipeline.settings",
            replace(
                settings,
                jira_dry_run=False,
                jira_base_url="https://jira.example",
                jira_email="bot@example.test",
                jira_api_token="test-token",
                jira_project_key="TEST",
            ),
        )
        self.pipeline_settings_patch.start()
        self.logger_patch = patch("app.pipeline.logger.exception")
        self.logger_patch.start()
        db.init_db()
        db.create_meeting("bot-network", "https://meet.google.com/abc-defg-hij")
        self.transcript_text = "I will add a regression test for reconnects."
        self.transcript: JSONValue = [
            {
                "id": "segment-1",
                "timestamp_ms": 1000,
                "speaker_name": "Maya",
                "text": self.transcript_text,
            }
        ]
        self.proposals = TaskExtraction(
            tasks=[
                TaskProposal(
                    summary="Add a reconnect regression test",
                    evidence_quote="I will add a regression test for reconnects.",
                    timestamp_ms=1000,
                    confidence=0.95,
                )
            ]
        )

    def tearDown(self) -> None:
        self.logger_patch.stop()
        self.pipeline_settings_patch.stop()
        self.db_settings_patch.stop()
        self.temp_dir.cleanup()

    def _stored_task(self) -> MeetingTaskRecord:
        meeting = db.get_meeting("bot-network")
        self.assertIsNotNone(meeting)
        assert meeting is not None
        return meeting["tasks"][0]

    async def test_definitely_unsent_jira_create_is_retried_safely(self) -> None:
        connect_error = httpx.ConnectError(
            "offline",
            request=httpx.Request("POST", "https://jira.example/rest/api/3/issue"),
        )
        attendee = FakeAttendee(self.transcript)
        jira = FakeJiraClient([connect_error, "TEST-1"])
        with (
            patch("app.pipeline.attendee", new=attendee),
            patch("app.pipeline.jira", new=jira),
            patch(
                "app.pipeline.extract_tasks", new=AsyncMock(return_value=self.proposals)
            ),
        ):
            with self.assertRaises(httpx.ConnectError):
                await finalize_meeting("bot-network")
            self.assertEqual(self._stored_task()["status"], "pending")

            await finalize_meeting("bot-network")

        task = self._stored_task()
        self.assertEqual(task["status"], "created")
        self.assertEqual(task["jira_key"], "TEST-1")
        self.assertEqual(jira.call_count, 2)
        self.assertEqual(attendee.transcript_call_count, 1)

    async def test_ambiguous_jira_response_is_not_retried(self) -> None:
        read_timeout = httpx.ReadTimeout(
            "response lost",
            request=httpx.Request("POST", "https://jira.example/rest/api/3/issue"),
        )
        attendee = FakeAttendee(self.transcript)
        jira = FakeJiraClient([read_timeout])
        with (
            patch("app.pipeline.attendee", new=attendee),
            patch("app.pipeline.jira", new=jira),
            patch(
                "app.pipeline.extract_tasks", new=AsyncMock(return_value=self.proposals)
            ),
        ):
            await finalize_meeting("bot-network")
            self.assertEqual(self._stored_task()["status"], "uncertain")

            await finalize_meeting("bot-network")

        self.assertEqual(jira.call_count, 1)
        self.assertEqual(attendee.transcript_call_count, 1)
        self.assertEqual(self._stored_task()["status"], "uncertain")

    async def test_chat_messages_are_forwarded_for_context_enrichment(self) -> None:
        chat_messages: list[ChatMessage] = [
            {
                "id": "chat-1",
                "timestamp_ms": 1100,
                "speaker_name": "Leo",
                "text": "Cover HTTP 429 with Retry-After.",
                "source": "chat",
            }
        ]
        attendee = FakeAttendee(self.transcript, chat_messages)
        jira = FakeJiraClient(["TEST-2"])
        extract = AsyncMock(return_value=self.proposals)
        with (
            patch("app.pipeline.attendee", new=attendee),
            patch("app.pipeline.jira", new=jira),
            patch("app.pipeline.extract_tasks", new=extract),
        ):
            await finalize_meeting("bot-network")

        extract.assert_awaited_once()
        call = extract.await_args
        assert call is not None
        self.assertEqual(call.args[0][0]["text"], self.transcript_text)
        self.assertEqual(call.kwargs["bot_id"], "bot-network")
        self.assertEqual(call.kwargs["chat_messages"], chat_messages)
        self.assertEqual(self._stored_task()["status"], "created")


if __name__ == "__main__":
    unittest.main()
