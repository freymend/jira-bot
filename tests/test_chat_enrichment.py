from __future__ import annotations

import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx

from app.config import settings
from app.integrations.attendee import AttendeeClient
from app.integrations.jira import JiraClient
from app.integrations.task_extraction import extract_tasks


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class AttendeeChatRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_chat_messages_follows_attendee_cursor_pages(self) -> None:
        requests: list[httpx.Request] = []

        def handle(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.params.get("cursor") is None:
                return httpx.Response(
                    200,
                    json={
                        "next": (
                            "https://attendee.example/api/v1/bots/bot-1/"
                            "chat_messages?cursor=page-2"
                        ),
                        "results": [
                            {
                                "id": "chat-1",
                                "text": "I will post the retry plan.",
                                "timestamp_ms": 1000,
                                "sender_name": "Maya",
                            }
                        ],
                    },
                )
            return httpx.Response(
                200,
                json={
                    "next": None,
                    "results": [
                        {
                            "id": "chat-2",
                            "text": "Include the Retry-After header.",
                            "timestamp": 2,
                            "sender_name": "Leo",
                        }
                    ],
                },
            )

        client = httpx.AsyncClient(
            base_url="https://attendee.example/api/v1/",
            transport=httpx.MockTransport(handle),
        )
        with patch("app.integrations.attendee.httpx.AsyncClient", return_value=client):
            attendee = AttendeeClient()
        try:
            messages = await attendee.get_chat_messages("bot-1")
        finally:
            await attendee.close()

        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[1].url.params["cursor"], "page-2")
        self.assertEqual(messages[0]["source"], "chat")
        self.assertEqual(messages[0]["timestamp_ms"], 1000)
        self.assertEqual(messages[0]["speaker_name"], "Maya")
        self.assertEqual(messages[1]["timestamp_ms"], 2000)


class ChatTaskEnrichmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_enriches_context_without_becoming_task_evidence(self) -> None:
        transcript = [
            {
                "id": "spoken-1",
                "timestamp_ms": 1000,
                "speaker_name": "Maya",
                "text": "I will add a regression test for reconnects.",
            }
        ]
        chat_message = {
            "id": "chat-1",
            "timestamp_ms": 2000,
            "speaker_name": "Leo",
            "text": "The reconnect test should cover HTTP 429 and Retry-After.",
            "source": "chat",
        }
        create = AsyncMock(
            side_effect=[
                _response(
                    json.dumps(
                        {
                            "tasks": [
                                {
                                    "summary": "Add a reconnect regression test",
                                    "evidence_quote": transcript[0]["text"],
                                    "timestamp_ms": 1000,
                                    "context_query": "reconnect HTTP 429 Retry-After",
                                    "confidence": 0.95,
                                }
                            ]
                        }
                    )
                ),
                _response(
                    json.dumps(
                        {
                            "description": (
                                "Add reconnect coverage for HTTP 429 responses "
                                "and the Retry-After header."
                            ),
                            "evidence_segment_ids": ["seg-00001"],
                        }
                    )
                ),
            ]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=AsyncMock(),
        )

        with patch("app.integrations.task_extraction.AsyncOpenAI", return_value=client):
            result = await extract_tasks(transcript, chat_messages=[chat_message])

        extraction_prompt = create.await_args_list[0].kwargs["messages"][1]["content"]
        context_prompt = create.await_args_list[1].kwargs["messages"][1]["content"]
        self.assertNotIn(chat_message["text"], extraction_prompt)
        self.assertIn(chat_message["text"], context_prompt)
        self.assertEqual(len(result.tasks), 1)
        self.assertEqual(result.tasks[0].evidence_quote, transcript[0]["text"])
        self.assertEqual(result.tasks[0].context_evidence[0].source, "transcript")
        self.assertEqual(result.tasks[0].context_evidence[1].source, "chat")
        self.assertEqual(
            result.tasks[0].context_evidence[1].quote, chat_message["text"]
        )

    async def test_jira_marks_chat_citations_as_chat_messages(self) -> None:
        issue_fields: dict[str, Any] = {}

        def handle(request: httpx.Request) -> httpx.Response:
            issue_fields.update(json.loads(request.content)["fields"])
            return httpx.Response(201, json={"key": "TEST-1"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        configuration = replace(
            settings,
            jira_base_url="https://jira.example",
            jira_email="bot@example.test",
            jira_api_token="jira-token",
            jira_project_key="TEST",
            jira_issue_type="Task",
        )
        task = {
            "summary": "Add reconnect coverage",
            "evidence_quote": "I will add a regression test for reconnects.",
            "description": "Cover reconnect behavior and HTTP 429 responses.",
            "context_evidence": [
                {
                    "quote": "Include the Retry-After header.",
                    "timestamp_ms": 2000,
                    "speaker_name": "Leo",
                    "source": "chat",
                }
            ],
        }
        try:
            with (
                patch("app.integrations.jira.settings", configuration),
                patch("app.integrations.jira.httpx.AsyncClient", return_value=client),
            ):
                key = await JiraClient().create_issue(task, None)
        finally:
            await client.aclose()

        description = issue_fields["description"]
        rendered = " ".join(
            item["content"][0]["text"]
            for item in description["content"]
            if item.get("type") == "paragraph"
        )
        self.assertEqual(key, "TEST-1")
        self.assertIn("Chat message (00:02)", rendered)
        self.assertIn("Include the Retry-After header.", rendered)


if __name__ == "__main__":
    unittest.main()
