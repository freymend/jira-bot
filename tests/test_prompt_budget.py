from __future__ import annotations

import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.config import settings
from app.integrations import task_extraction


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _client(side_effects: list[object]) -> tuple[SimpleNamespace, AsyncMock]:
    create = AsyncMock(side_effect=side_effects)
    return (
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=AsyncMock(),
        ),
        create,
    )


class PromptBudgetTests(unittest.IsolatedAsyncioTestCase):
    def test_multilingual_transcript_chunks_respect_the_byte_budget(self) -> None:
        segments = [
            {
                "timestamp_ms": index,
                "speaker_name": "Maya",
                "text": "中文内容🙂 " * 5000,
            }
            for index in range(3)
        ]
        chunks = task_extraction._transcript_chunks(segments)
        max_chunk_bytes = (
            settings.llm_context_tokens
            - settings.llm_max_completion_tokens
            - task_extraction._PROMPT_SAFETY_TOKENS
            - task_extraction._CHAT_TEMPLATE_OVERHEAD_TOKENS
            - task_extraction._EXTRACTION_PROMPT_RESERVE_TOKENS
        )

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            prompt_text = "\n".join(
                task_extraction._format_transcript_segment(segment)
                for segment in chunk
            )
            self.assertLessEqual(len(prompt_text.encode("utf-8")), max_chunk_bytes)

    def test_transcript_cache_key_includes_prompt_budget(self) -> None:
        segments = [
            {"timestamp_ms": 1000, "speaker_name": "Maya", "text": "hello"}
        ]
        original_hash = task_extraction._transcript_cache_hash(segments)
        changed_settings = replace(
            settings, llm_context_tokens=settings.llm_context_tokens - 1024
        )
        with patch("app.integrations.task_extraction.settings", changed_settings):
            changed_hash = task_extraction._transcript_cache_hash(segments)

        self.assertNotEqual(original_hash, changed_hash)

    def test_context_packet_compacts_multilingual_excerpts_to_budget(self) -> None:
        text = "前置背景。" + "重要任务内容🙂。" * 1500 + "后续说明。"
        segment = {
            "id": "s1",
            "timestamp_ms": 1000,
            "speaker_name": "Maya",
            "text": text,
        }
        task = task_extraction.TaskProposal(
            summary="Document the task",
            evidence_quote="重要任务内容🙂。",
            timestamp_ms=1000,
            context_query="重要任务内容",
            confidence=0.95,
        )
        packet, _, _ = task_extraction._task_context_packet(
            [segment], task, max_context_bytes=1024
        )

        self.assertTrue(packet)
        self.assertLessEqual(
            sum(len(line.encode("utf-8")) + 1 for _, line in packet), 1024
        )
        self.assertIn(task.evidence_quote, packet[0][1])

    async def test_context_limit_error_splits_transcript_and_retries(self) -> None:
        segments = [
            {"timestamp_ms": 1000, "speaker_name": "Maya", "text": "First item."},
            {"timestamp_ms": 2000, "speaker_name": "Leo", "text": "Second item."},
        ]
        client, create = _client(
            [
                RuntimeError("maximum context length exceeded"),
                _response('{"tasks":[]}'),
                _response('{"tasks":[]}'),
            ]
        )
        with patch(
            "app.integrations.task_extraction.AsyncOpenAI", return_value=client
        ):
            result = await task_extraction.extract_tasks(segments)

        self.assertEqual(result.tasks, [])
        self.assertEqual(create.await_count, 3)
        self.assertTrue(
            all(
                call.kwargs["max_tokens"] == settings.llm_max_completion_tokens
                for call in create.await_args_list
            )
        )

    async def test_context_enrichment_retries_with_a_smaller_packet(self) -> None:
        segments = [
            {
                "id": "s1",
                "timestamp_ms": 1000,
                "speaker_name": "Maya",
                "text": "I will document the offline recovery steps. " + "中🙂" * 1000,
            }
        ]
        extracted = {
            "tasks": [
                {
                    "summary": "Document offline recovery",
                    "evidence_quote": "I will document the offline recovery steps.",
                    "timestamp_ms": 1000,
                    "context_query": "offline recovery",
                    "confidence": 0.95,
                }
            ]
        }
        context = json.dumps(
            {"description": "Document the recovery procedure.", "evidence_segment_ids": []}
        )
        client, create = _client(
            [
                _response(json.dumps(extracted)),
                RuntimeError("prompt exceeds max_model_len 16384"),
                _response(context),
            ]
        )
        with patch(
            "app.integrations.task_extraction.AsyncOpenAI", return_value=client
        ):
            result = await task_extraction.extract_tasks(segments)

        self.assertEqual(len(result.tasks), 1)
        self.assertEqual(create.await_count, 3)
        self.assertEqual(
            create.await_args_list[1].kwargs["max_tokens"],
            task_extraction._CONTEXT_COMPLETION_TOKENS,
        )
        self.assertLess(
            len(create.await_args_list[2].kwargs["messages"][1]["content"]),
            len(create.await_args_list[1].kwargs["messages"][1]["content"]),
        )


if __name__ == "__main__":
    unittest.main()
