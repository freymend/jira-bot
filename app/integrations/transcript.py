from __future__ import annotations

from typing import Any


def _text_from_item(item: dict[str, Any]) -> str:
    transcription = item.get("transcription")
    if isinstance(transcription, dict):
        transcription = transcription.get("transcript") or transcription.get("text")
    if transcription is None:
        transcription = item.get("transcript") or item.get("text")
    return str(transcription or "").strip()


def normalize_segments(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        for key in ("results", "segments", "transcriptions", "items", "transcript"):
            nested = value.get(key)
            if isinstance(nested, list):
                value = nested
                break
        else:
            value = [value]
    if not isinstance(value, list):
        return []
    segments: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = _text_from_item(item)
        if not text:
            continue
        segments.append(
            {
                "id": item.get("id") or item.get("segment_id"),
                "timestamp_ms": item.get("timestamp_ms"),
                "duration_ms": item.get("duration_ms"),
                "speaker_name": item.get("speaker_name"),
                "speaker_uuid": item.get("speaker_uuid"),
                "text": text,
                "raw": item,
            }
        )
    return sorted(
        segments,
        key=lambda segment: (
            segment["timestamp_ms"] is None,
            segment["timestamp_ms"] or 0,
        ),
    )
