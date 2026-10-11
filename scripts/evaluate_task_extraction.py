from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from app.config import settings
from app.integrations.task_extraction import extract_tasks
from app.integrations.transcript import normalize_segments

DEFAULT_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "synthetic_daily_sync.json"
)


def _canonical_owner(owner: str | None) -> str | None:
    if (owner or "").strip().casefold() in {"zac", "zachary"}:
        return "Zachary"
    return owner


def _score(
    proposals: list[Any],
    expected_tasks: list[dict[str, Any]],
    segment_by_timestamp: dict[int, str],
    scope_owner: str,
) -> dict[str, Any]:
    target_owner = _canonical_owner(scope_owner) or scope_owner
    expected_by_segment = {task["evidence_segment_id"]: task for task in expected_tasks}
    in_scope_proposals: list[tuple[Any, str | None, dict[str, Any] | None]] = []
    out_of_scope_proposals = 0
    for proposal in proposals:
        segment_id = segment_by_timestamp.get(proposal.timestamp_ms)
        expected = expected_by_segment.get(segment_id or "")
        if (
            expected is not None
            or _canonical_owner(proposal.owner_speaker) == target_owner
        ):
            in_scope_proposals.append((proposal, segment_id, expected))
        else:
            out_of_scope_proposals += 1

    matched_segments: set[str] = set()
    true_positives = 0
    false_positives = 0
    duplicate_predictions = 0
    owner_matches = 0
    owner_mismatches = 0
    false_target_assignments = 0
    target_assignment_predictions = 0
    correctly_assigned = 0
    details: list[dict[str, Any]] = []

    for proposal, segment_id, expected in in_scope_proposals:
        predicted_owner = _canonical_owner(proposal.owner_speaker)
        matched = (
            expected is not None
            and segment_id is not None
            and segment_id not in matched_segments
        )
        owner_matches_task = False
        if matched:
            assert expected is not None and segment_id is not None
            matched_segments.add(segment_id)
            true_positives += 1
            owner_matches_task = predicted_owner == _canonical_owner(
                expected["owner_speaker"]
            )
            if owner_matches_task:
                owner_matches += 1
                correctly_assigned += 1
            else:
                owner_mismatches += 1
        else:
            false_positives += 1
            if expected is not None:
                duplicate_predictions += 1

        if predicted_owner == target_owner:
            target_assignment_predictions += 1
            if not matched or not owner_matches_task:
                false_target_assignments += 1

        details.append(
            {
                "matched_gold_task": (
                    expected["id"] if matched and expected is not None else None
                ),
                "source_segment_id": segment_id,
                "summary": proposal.summary,
                "owner_speaker": proposal.owner_speaker,
                "canonical_owner": predicted_owner,
                "confidence": proposal.confidence,
                "duplicate": expected is not None and not matched,
            }
        )

    false_negatives = len(expected_tasks) - true_positives
    denominator = true_positives + false_positives
    precision = true_positives / denominator if denominator else 0.0
    recall = true_positives / len(expected_tasks) if expected_tasks else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    assignment_precision = (
        correctly_assigned / target_assignment_predictions
        if target_assignment_predictions
        else 0.0
    )
    assignment_recall = (
        correctly_assigned / len(expected_tasks) if expected_tasks else 1.0
    )
    return {
        "target_owner": target_owner,
        "gold_tasks": len(expected_tasks),
        "in_scope_proposals": len(in_scope_proposals),
        "out_of_scope_proposals_ignored": out_of_scope_proposals,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "duplicate_predictions": duplicate_predictions,
        "task_precision": round(precision, 3),
        "task_recall": round(recall, 3),
        "task_f1": round(f1, 3),
        "correctly_assigned_to_target": correctly_assigned,
        "owner_mismatches_on_matched_tasks": owner_mismatches,
        "target_assignment_precision": round(assignment_precision, 3),
        "target_assignment_recall": round(assignment_recall, 3),
        "false_target_assignments": false_target_assignments,
        "details": details,
    }


async def _evaluate(fixture_path: Path) -> None:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    segments = normalize_segments(fixture["segments"])
    timestamp_to_segment = {
        int(segment["timestamp_ms"]): str(segment["id"])
        for segment in segments
        if segment.get("timestamp_ms") is not None and segment.get("id")
    }

    extraction = await extract_tasks(segments)
    scope_owner = fixture["scope_owner"]
    raw = _score(
        extraction.tasks,
        fixture["expected_tasks"],
        timestamp_to_segment,
        scope_owner,
    )
    create_ready_proposals = [
        task
        for task in extraction.tasks
        if task.confidence >= settings.llm_confidence_threshold
    ]
    create_ready = _score(
        create_ready_proposals,
        fixture["expected_tasks"],
        timestamp_to_segment,
        scope_owner,
    )

    print(f"Fixture: {fixture['name']}")
    print(f"Model: {settings.llm_model}")
    print(f"Confidence threshold: {settings.llm_confidence_threshold:.2f}")
    print("Raw extraction:")
    print(json.dumps(raw, indent=2, ensure_ascii=False))
    print("After pipeline confidence filter:")
    print(json.dumps(create_ready, indent=2, ensure_ascii=False))
    print(
        "\nReview summary/evidence quality manually; this script does not create Jira issues."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run task extraction against the synthetic daily-sync fixture."
    )
    parser.add_argument(
        "fixture",
        nargs="?",
        type=Path,
        default=DEFAULT_FIXTURE,
        help="Fixture JSON path (defaults to the included daily-sync fixture).",
    )
    args = parser.parse_args()
    asyncio.run(_evaluate(args.fixture))


if __name__ == "__main__":
    main()
