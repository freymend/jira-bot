from __future__ import annotations

import json
import time

from app.repositories.database import _connect
from app.schemas import ExtractionCheckpointRow, JSONValue


def clear_stale_extraction_checkpoints(bot_id: str, transcript_hash: str) -> None:
    with _connect() as connection:
        connection.execute(
            "DELETE FROM task_extraction_cache WHERE bot_id=? AND transcript_hash!=?",
            (bot_id, transcript_hash),
        )


def get_extraction_checkpoint(
    bot_id: str, transcript_hash: str, stage: str, item_key: str
) -> JSONValue | None:
    with _connect() as connection:
        row = connection.execute(
            """SELECT result_json FROM task_extraction_cache
               WHERE bot_id=? AND transcript_hash=? AND stage=? AND item_key=?""",
            (bot_id, transcript_hash, stage, item_key),
        ).fetchone()
        if row is None:
            return None
        checkpoint: ExtractionCheckpointRow = {"result_json": row["result_json"]}
        try:
            return json.loads(checkpoint["result_json"])
        except json.JSONDecodeError:
            connection.execute(
                """DELETE FROM task_extraction_cache
                   WHERE bot_id=? AND transcript_hash=? AND stage=? AND item_key=?""",
                (bot_id, transcript_hash, stage, item_key),
            )
            return None


def delete_extraction_checkpoint(
    bot_id: str, transcript_hash: str, stage: str, item_key: str
) -> None:
    with _connect() as connection:
        connection.execute(
            """DELETE FROM task_extraction_cache
               WHERE bot_id=? AND transcript_hash=? AND stage=? AND item_key=?""",
            (bot_id, transcript_hash, stage, item_key),
        )


def save_extraction_checkpoint(
    bot_id: str,
    transcript_hash: str,
    stage: str,
    item_key: str,
    result: JSONValue,
) -> None:
    with _connect() as connection:
        connection.execute(
            """INSERT INTO task_extraction_cache
               (bot_id, transcript_hash, stage, item_key, result_json, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(bot_id, transcript_hash, stage, item_key)
               DO UPDATE SET result_json=excluded.result_json,
                             updated_at=excluded.updated_at""",
            (
                bot_id,
                transcript_hash,
                stage,
                item_key,
                json.dumps(result, ensure_ascii=False),
                time.time(),
            ),
        )
