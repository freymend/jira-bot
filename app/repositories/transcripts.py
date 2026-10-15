from __future__ import annotations

import hashlib
import json

from app.repositories.database import _connect, _now
from app.schemas import RecentTranscriptSegment, TranscriptSegment


def get_recent_transcript_segments(
    bot_id: str, limit: int = 40
) -> list[RecentTranscriptSegment]:
    bounded_limit = min(max(limit, 1), 100)
    with _connect() as connection:
        rows = connection.execute(
            """SELECT timestamp_ms, speaker_name, text
               FROM transcript_segments WHERE bot_id=?
               ORDER BY rowid DESC LIMIT ?""",
            (bot_id, bounded_limit),
        ).fetchall()
        result: list[RecentTranscriptSegment] = []
        for row in reversed(rows):
            result.append(
                {
                    "timestamp_ms": row["timestamp_ms"],
                    "speaker_name": row["speaker_name"],
                    "text": row["text"],
                }
            )
        return result


def save_segments(bot_id: str, segments: list[TranscriptSegment]) -> None:
    with _connect() as connection:
        changed = False
        for segment in segments:
            timestamp = segment.get("timestamp_ms")
            speaker_uuid = segment.get("speaker_uuid")
            speaker_name = segment.get("speaker_name")
            identity = segment.get("id") or segment.get("segment_id")
            if identity is None and timestamp is not None:
                identity = f"{speaker_uuid or speaker_name or ''}:{timestamp}"
            if identity is None:
                identity = json.dumps(segment, sort_keys=True, ensure_ascii=False)
            key = hashlib.sha256(str(identity).encode("utf-8")).hexdigest()
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            result = connection.execute(
                """INSERT INTO transcript_segments
                   (bot_id, segment_key, timestamp_ms, speaker_name, speaker_uuid, text, raw_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(bot_id, segment_key) DO UPDATE SET
                     timestamp_ms=excluded.timestamp_ms,
                     speaker_name=excluded.speaker_name,
                     speaker_uuid=excluded.speaker_uuid,
                     text=excluded.text,
                     raw_json=excluded.raw_json
                   WHERE transcript_segments.timestamp_ms IS NOT excluded.timestamp_ms
                      OR transcript_segments.speaker_name IS NOT excluded.speaker_name
                      OR transcript_segments.speaker_uuid IS NOT excluded.speaker_uuid
                      OR transcript_segments.text IS NOT excluded.text
                      OR transcript_segments.raw_json IS NOT excluded.raw_json""",
                (
                    bot_id,
                    key,
                    timestamp,
                    speaker_name,
                    speaker_uuid,
                    text,
                    json.dumps(segment.get("raw", segment), ensure_ascii=False),
                ),
            )
            changed = changed or result.rowcount == 1
        connection.execute(
            """UPDATE meetings SET transcription_state='in_progress'
               WHERE bot_id=? AND transcription_state!='complete'""",
            (bot_id,),
        )
        if changed:
            connection.execute(
                "UPDATE meetings SET last_transcript_received_at=? WHERE bot_id=?",
                (_now(), bot_id),
            )


def save_final_transcript(bot_id: str, segments: list[TranscriptSegment]) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE meetings SET transcript_json=?, transcription_state='complete' WHERE bot_id=?",
            (json.dumps(segments, ensure_ascii=False), bot_id),
        )
    save_segments(bot_id, segments)
