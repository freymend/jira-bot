from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

import httpx
from openai import AsyncOpenAI

from app import db
from app.config import settings
from app.integrations.task_extraction import _extract_chunk_checkpointed, extract_tasks
from app.schemas import TranscriptSegment


def _response(content: str, *, finish_reason: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason=finish_reason
            )
        ]
    )


def _fake_client(side_effects: list[object]) -> tuple[SimpleNamespace, AsyncMock]:
    create = AsyncMock(side_effect=side_effects)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=AsyncMock(),
    )
    return client, create


class TaskExtractionResilienceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "test.sqlite3"
        self.settings_patch = patch(
            "app.repositories.database.settings",
            replace(settings, database_path=str(database_path)),
        )
        self.settings_patch.start()
        db.init_db()

    def tearDown(self) -> None:
        self.settings_patch.stop()
        self.temp_dir.cleanup()

    async def test_truncated_json_response_is_split_and_retried(self) -> None:
        segments: list[TranscriptSegment] = [
            {
                "id": "s1",
                "timestamp_ms": 1000,
                "duration_ms": 0,
                "speaker_name": "Maya",
                "speaker_uuid": None,
                "text": "I'll add a regression test.",
                "raw": {},
            },
            {
                "id": "s2",
                "timestamp_ms": 2000,
                "duration_ms": 0,
                "speaker_name": "Leo",
                "speaker_uuid": None,
                "text": "I'll document recovery steps.",
                "raw": {},
            },
        ]

        def extracted_task(summary: str, quote: str, timestamp_ms: int) -> str:
            return json.dumps(
                {
                    "tasks": [
                        {
                            "summary": summary,
                            "evidence_quote": quote,
                            "timestamp_ms": timestamp_ms,
                            "context_query": summary,
                            "confidence": 0.95,
                        }
                    ]
                }
            )

        client, create = _fake_client(
            [
                _response('{"tasks":[{"summary":"truncated'),
                _response(
                    extracted_task("Add regression test", segments[0]["text"], 1000)
                ),
                _response(
                    extracted_task("Document recovery", segments[1]["text"], 2000)
                ),
            ]
        )

        result = await _extract_chunk_checkpointed(
            cast(AsyncOpenAI, client),
            segments,
            bot_id=None,
            transcript_hash=None,
            item_key="0",
        )

        self.assertEqual(create.await_count, 3)
        self.assertEqual(
            [task.summary for task in result.tasks],
            ["Add regression test", "Document recovery"],
        )

    async def test_completion_token_limit_splits_even_valid_json(self) -> None:
        segments: list[TranscriptSegment] = [
            {
                "id": "s1",
                "timestamp_ms": 1000,
                "duration_ms": 0,
                "speaker_name": "Maya",
                "speaker_uuid": None,
                "text": "I'll add a regression test.",
                "raw": {},
            },
            {
                "id": "s2",
                "timestamp_ms": 2000,
                "duration_ms": 0,
                "speaker_name": "Leo",
                "speaker_uuid": None,
                "text": "I'll document recovery steps.",
                "raw": {},
            },
        ]
        client, create = _fake_client(
            [
                _response('{"tasks":[]}', finish_reason="length"),
                _response('{"tasks":[]}'),
                _response('{"tasks":[]}'),
            ]
        )

        result = await _extract_chunk_checkpointed(
            cast(AsyncOpenAI, client),
            segments,
            bot_id=None,
            transcript_hash=None,
            item_key="0",
        )

        self.assertEqual(create.await_count, 3)
        self.assertEqual(result.tasks, [])

    async def test_completed_chunk_and_context_results_survive_interruption(
        self,
    ) -> None:
        segments: list[TranscriptSegment] = [
            {
                "id": "s1",
                "timestamp_ms": 1000,
                "duration_ms": 0,
                "speaker_name": "Maya",
                "speaker_uuid": None,
                "text": "I'll add a regression test for reconnects.",
                "raw": {},
            },
            {
                "id": "s2",
                "timestamp_ms": 2000,
                "duration_ms": 0,
                "speaker_name": "Leo",
                "speaker_uuid": None,
                "text": "I'll document the offline recovery steps.",
                "raw": {},
            },
        ]
        extracted = {
            "tasks": [
                {
                    "summary": "Add a retry regression test",
                    "evidence_quote": "I'll add a regression test for reconnects.",
                    "timestamp_ms": 1000,
                    "context_query": "retry regression test",
                    "confidence": 0.95,
                },
                {
                    "summary": "Document offline recovery",
                    "evidence_quote": "I'll document the offline recovery steps.",
                    "timestamp_ms": 2000,
                    "context_query": "offline recovery",
                    "confidence": 0.96,
                },
            ]
        }
        context = json.dumps(
            {
                "description": "Document the reconnect procedure.",
                "evidence_segment_ids": [],
            }
        )
        offline_error = httpx.ConnectError(
            "network unavailable",
            request=httpx.Request("POST", "https://model.example/v1/chat/completions"),
        )
        first_client, first_create = _fake_client(
            [_response(json.dumps(extracted)), _response(context), offline_error]
        )
        retry_client, retry_create = _fake_client([_response(context)])

        with patch(
            "app.integrations.task_extraction.AsyncOpenAI",
            side_effect=[first_client, retry_client],
        ):
            with self.assertRaises(httpx.ConnectError):
                await extract_tasks(segments, bot_id="bot-checkpoint")
            result = await extract_tasks(segments, bot_id="bot-checkpoint")

        self.assertEqual(first_create.await_count, 3)
        self.assertEqual(retry_create.await_count, 1)
        self.assertEqual(len(result.tasks), 2)
        self.assertIn(
            segments[0]["text"],
            [item.quote for item in result.tasks[0].context_evidence],
        )
        self.assertIn(
            segments[1]["text"],
            [item.quote for item in result.tasks[1].context_evidence],
        )


if __name__ == "__main__":
    unittest.main()
