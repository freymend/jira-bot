from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import time
from datetime import UTC, datetime

from fastapi import (
    APIRouter,
    Header,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)

from app import db
from app.config import settings
from app.integrations.attendee import verify_attendee_signature
from app.integrations.transcript import normalize_segments
from app.schemas import JSONObject

router = APIRouter()


def _timestamp_ms(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


def _event_is_final(payload: JSONObject) -> bool:
    if payload.get("trigger") != "bot.state_change":
        return False
    data = payload.get("data") or {}
    return (
        isinstance(data, dict) and data.get("event_type") == "post_processing_completed"
    )


@router.post("/webhooks/attendee")
async def attendee_webhook(
    request: Request,
    x_webhook_signature: str | None = Header(default=None),
) -> dict[str, str]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc
    if not isinstance(payload, dict) or not verify_attendee_signature(
        payload, x_webhook_signature
    ):
        raise HTTPException(status_code=401, detail="Invalid Attendee signature")
    trigger = str(payload.get("trigger") or "unknown")
    bot_id = str(payload.get("bot_id") or "")
    if not bot_id:
        raise HTTPException(status_code=400, detail="Webhook is missing bot_id")

    event_id = str(
        payload.get("idempotency_key")
        or hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
    )
    recorded = db.record_webhook(
        event_id,
        bot_id,
        trigger,
        payload,
        enqueue_finalization=_event_is_final(payload),
    )
    if not recorded:
        return {"status": "received"}
    event_data = payload.get("data") or {}
    if trigger == "bot.state_change" and isinstance(event_data, dict):
        new_state = event_data.get("new_state")
        if _event_is_final(payload):
            db.update_bot_state(bot_id, "post_processing_completed")
            db.finish_meeting_leave_watchdog(bot_id, "completed")
        elif new_state:
            db.update_bot_state(bot_id, str(new_state))
        if not _event_is_final(payload) and new_state == "joined_recording":
            recording_started_at_ms = _timestamp_ms(event_data.get("created_at"))
            if recording_started_at_ms is not None:
                db.set_recording_started_at(bot_id, recording_started_at_ms)
    if trigger == "transcript.update":
        segment = normalize_segments(event_data)
        db.save_segments(bot_id, segment)
    if trigger == "participant_events.join_leave" and isinstance(event_data, dict):
        participant_uuid = event_data.get("participant_uuid")
        event_type = event_data.get("event_type")
        timestamp_ms = event_data.get("timestamp_ms")
        if (
            isinstance(participant_uuid, str)
            and participant_uuid.strip()
            and event_type in {"join", "leave"}
            and isinstance(timestamp_ms, int)
            and not isinstance(timestamp_ms, bool)
            and 0 <= timestamp_ms <= 2**63 - 1
        ):
            participant_event_id = event_data.get("id") or payload.get(
                "idempotency_key"
            )
            if not isinstance(participant_event_id, str) or not participant_event_id:
                participant_event_id = hashlib.sha256(
                    json.dumps(event_data, sort_keys=True, ensure_ascii=False).encode(
                        "utf-8"
                    )
                ).hexdigest()
            participant_name = event_data.get("participant_name")
            is_host = event_data.get("participant_is_host")
            db.record_participant_event(
                bot_id,
                participant_event_id[:255],
                participant_uuid.strip()[:255],
                participant_name[:255] if isinstance(participant_name, str) else None,
                is_host if isinstance(is_host, bool) else None,
                event_type,
                timestamp_ms,
            )
    return {"status": "received"}


@router.websocket("/webhooks/attendee/video")
async def attendee_video_websocket(websocket: WebSocket) -> None:
    token = websocket.query_params.get("token", "")
    if (
        not settings.screenshare_capture_enabled
        or not settings.screenshare_websocket_token
        or not hmac.compare_digest(
            token.encode("utf-8"), settings.screenshare_websocket_token.encode("utf-8")
        )
    ):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    max_frame_bytes = 8 * 1024 * 1024
    max_encoded_bytes = ((max_frame_bytes + 2) // 3) * 4
    try:
        while True:
            try:
                payload = await websocket.receive_json()
            except ValueError:
                await websocket.close(code=1003)
                return
            if not isinstance(payload, dict):
                continue
            event_data = payload.get("data")
            if (
                payload.get("trigger") != "realtime_video.per_participant"
                or not isinstance(event_data, dict)
                or event_data.get("source") != "screenshare"
                or event_data.get("format") != "jpeg"
            ):
                continue

            bot_id = str(payload.get("bot_id") or "")
            if not bot_id or not db.is_screenshare_capture_enabled(bot_id):
                continue
            encoded_frame = event_data.get("frame")
            if (
                not isinstance(encoded_frame, str)
                or not encoded_frame
                or len(encoded_frame) > max_encoded_bytes
            ):
                continue
            try:
                frame_data = base64.b64decode(encoded_frame, validate=True)
            except binascii.Error:
                continue
            except ValueError:
                continue
            if (
                len(frame_data) > max_frame_bytes
                or not frame_data.startswith(b"\xff\xd8")
                or not frame_data.endswith(b"\xff\xd9")
            ):
                continue
            participant_uuid = event_data.get("participant_uuid")
            db.store_screen_frame(
                bot_id,
                frame_data,
                received_at_ms=int(time.time() * 1000),
                participant_uuid=(
                    str(participant_uuid)[:255]
                    if isinstance(participant_uuid, str)
                    else None
                ),
            )
    except WebSocketDisconnect:
        return
