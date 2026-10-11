from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.integrations.task_extraction import _expand_owner_quote
from scripts.evaluate_task_extraction import _score

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_daily_sync.json"


class SyntheticTaskExtractionEvalTests(unittest.TestCase):
    def test_fixture_spans_an_hour_and_gold_quotes_exist_in_transcript(self) -> None:
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        segments = fixture["segments"]
        by_id = {segment["id"]: segment for segment in segments}
        duration_ms = max(item["timestamp_ms"] for item in segments) - min(
            item["timestamp_ms"] for item in segments
        )

        self.assertGreaterEqual(duration_ms, 55 * 60 * 1000)
        self.assertLessEqual(duration_ms, 65 * 60 * 1000)
        speakers = {segment["speaker_name"] for segment in segments}
        self.assertGreaterEqual(len(speakers), 5)
        self.assertNotIn("Zachary", speakers)
        self.assertNotIn("Zac", speakers)
        self.assertEqual(len(fixture["expected_tasks"]), 3)
        self.assertTrue(
            all(
                task["owner_speaker"] == "Zachary" for task in fixture["expected_tasks"]
            )
        )

        zachary_tasks = [
            task
            for task in fixture["expected_tasks"]
            if task["owner_speaker"] == "Zachary"
        ]
        self.assertEqual(len(zachary_tasks), 3)
        for task in zachary_tasks:
            owner_segment = by_id[task["owner_evidence_segment_id"]]
            self.assertNotIn(owner_segment["speaker_name"], {"Zachary", "Zac"})
            self.assertIn("Zachary", task["owner_evidence_quote"])

        for task in fixture["expected_tasks"]:
            self.assertIn(
                task["evidence_quote"],
                by_id[task["evidence_segment_id"]]["text"],
                task["id"],
            )
            if task["owner_evidence_quote"]:
                self.assertIn(
                    task["owner_evidence_quote"],
                    by_id[task["owner_evidence_segment_id"]]["text"],
                    task["id"],
                )

    def test_expands_same_segment_pronoun_assignment_but_not_ambiguous_language(self) -> None:
        assigned_segment = (
            "Zachary is out today, but let's assign him the parser tests for when he returns."
        )
        assignment_quote = "let's assign him the parser tests for when he returns."
        self.assertEqual(
            _expand_owner_quote("Zachary", assignment_quote, assigned_segment),
            assigned_segment,
        )

        ambiguous_segment = (
            "Zachary is out today; perhaps we should assign him the parser tests."
        )
        ambiguous_quote = "perhaps we should assign him the parser tests."
        self.assertEqual(
            _expand_owner_quote("Zachary", ambiguous_quote, ambiguous_segment),
            ambiguous_quote,
        )

    def test_scorer_limits_metrics_to_target_owner_and_normalizes_alias(self) -> None:
        expected = [
            {"id": "first", "evidence_segment_id": "s1", "owner_speaker": "Zachary"},
            {"id": "second", "evidence_segment_id": "s2", "owner_speaker": "Zachary"},
        ]
        proposals = [
            SimpleNamespace(
                timestamp_ms=100, summary="first", owner_speaker="Zac", confidence=0.9
            ),
            SimpleNamespace(
                timestamp_ms=200,
                summary="second, owner missing",
                owner_speaker=None,
                confidence=0.8,
            ),
            SimpleNamespace(
                timestamp_ms=200,
                summary="duplicate",
                owner_speaker="Zachary",
                confidence=0.7,
            ),
            SimpleNamespace(
                timestamp_ms=300,
                summary="wrong task assigned to target",
                owner_speaker="Zachary",
                confidence=0.6,
            ),
            SimpleNamespace(
                timestamp_ms=400,
                summary="other person's task",
                owner_speaker=None,
                confidence=0.5,
            ),
        ]

        result = _score(
            proposals,
            expected,
            {100: "s1", 200: "s2", 300: "s3", 400: "s4"},
            "Zachary",
        )

        self.assertEqual(result["true_positives"], 2)
        self.assertEqual(result["false_positives"], 2)
        self.assertEqual(result["false_negatives"], 0)
        self.assertEqual(result["duplicate_predictions"], 1)
        self.assertEqual(result["out_of_scope_proposals_ignored"], 1)
        self.assertEqual(result["correctly_assigned_to_target"], 1)
        self.assertEqual(result["owner_mismatches_on_matched_tasks"], 1)
        self.assertEqual(result["false_target_assignments"], 2)


if __name__ == "__main__":
    unittest.main()
