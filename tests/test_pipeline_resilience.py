from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from app import db
from app.config import settings
from app.pipeline import finalize_meeting
from app.schemas import MeetingTaskRecord, TaskExtraction, TaskProposal


class PipelineNetworkResilienceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "test.sqlite3"
        self.db_settings_patch = patch(
            "app.db.settings", replace(settings, database_path=str(database_path))
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
        self.transcript = [
            {
                "id": "segment-1",
                "timestamp_ms": 1000,
                "speaker_name": "Maya",
                "text": "I will add a regression test for reconnects.",
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
        create_issue = AsyncMock(side_effect=[connect_error, "TEST-1"])
        get_transcript = AsyncMock(return_value=self.transcript)
        get_chat_messages = AsyncMock(return_value=[])
        with (
            patch("app.pipeline.attendee.get_transcript", new=get_transcript),
            patch("app.pipeline.attendee.get_chat_messages", new=get_chat_messages),
            patch(
                "app.pipeline.extract_tasks", new=AsyncMock(return_value=self.proposals)
            ),
            patch("app.pipeline.jira.create_issue", new=create_issue),
        ):
            with self.assertRaises(httpx.ConnectError):
                await finalize_meeting("bot-network")
            self.assertEqual(self._stored_task()["status"], "pending")

            await finalize_meeting("bot-network")

        task = self._stored_task()
        self.assertEqual(task["status"], "created")
        self.assertEqual(task["jira_key"], "TEST-1")
        self.assertEqual(create_issue.await_count, 2)
        self.assertEqual(get_transcript.await_count, 1)

    async def test_ambiguous_jira_response_is_not_retried(self) -> None:
        read_timeout = httpx.ReadTimeout(
            "response lost",
            request=httpx.Request("POST", "https://jira.example/rest/api/3/issue"),
        )
        create_issue = AsyncMock(side_effect=read_timeout)
        get_transcript = AsyncMock(return_value=self.transcript)
        get_chat_messages = AsyncMock(return_value=[])
        with (
            patch("app.pipeline.attendee.get_transcript", new=get_transcript),
            patch("app.pipeline.attendee.get_chat_messages", new=get_chat_messages),
            patch(
                "app.pipeline.extract_tasks", new=AsyncMock(return_value=self.proposals)
            ),
            patch("app.pipeline.jira.create_issue", new=create_issue),
        ):
            await finalize_meeting("bot-network")
            self.assertEqual(self._stored_task()["status"], "uncertain")

            await finalize_meeting("bot-network")

        self.assertEqual(create_issue.await_count, 1)
        self.assertEqual(get_transcript.await_count, 1)
        self.assertEqual(self._stored_task()["status"], "uncertain")

    async def test_chat_messages_are_forwarded_for_context_enrichment(self) -> None:
        chat_messages = [
            {
                "id": "chat-1",
                "timestamp_ms": 1100,
                "speaker_name": "Leo",
                "text": "Cover HTTP 429 with Retry-After.",
                "source": "chat",
            }
        ]
        get_transcript = AsyncMock(return_value=self.transcript)
        get_chat_messages = AsyncMock(return_value=chat_messages)
        extract = AsyncMock(return_value=self.proposals)
        create_issue = AsyncMock(return_value="TEST-2")
        with (
            patch("app.pipeline.attendee.get_transcript", new=get_transcript),
            patch("app.pipeline.attendee.get_chat_messages", new=get_chat_messages),
            patch("app.pipeline.extract_tasks", new=extract),
            patch("app.pipeline.jira.create_issue", new=create_issue),
        ):
            await finalize_meeting("bot-network")

        extract.assert_awaited_once()
        call = extract.await_args
        assert call is not None
        self.assertEqual(call.args[0][0]["text"], self.transcript[0]["text"])
        self.assertEqual(call.kwargs["bot_id"], "bot-network")
        self.assertEqual(call.kwargs["chat_messages"], chat_messages)
        self.assertEqual(self._stored_task()["status"], "created")


if __name__ == "__main__":
    unittest.main()
