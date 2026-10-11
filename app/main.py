from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse, urlsplit

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, Response
import httpx

from app import db
from app.calendar_scheduler import (
    accept_calendar_notification,
    run_calendar_auto_join,
)
from app.config import settings
from app.integrations.attendee import verify_attendee_signature
from app.integrations.transcript import normalize_segments
from app.meeting_lifecycle import run_meeting_leave_monitor
from app.pipeline import attendee
from app.schemas import MeetingStartRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jira-bot")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    db.init_db()
    db.delete_expired_screen_frames()
    background_tasks: list[asyncio.Task[None]] = []
    if settings.google_calendar_auto_join_enabled:
        calendar_sync_wakeup = asyncio.Event()
        background_tasks.append(
            asyncio.create_task(
                run_calendar_auto_join(attendee, calendar_sync_wakeup),
                name="google-calendar-auto-join",
            )
        )
    if settings.meeting_auto_leave_enabled:
        background_tasks.append(
            asyncio.create_task(
                run_meeting_leave_monitor(attendee), name="meeting-auto-leave"
            )
        )
    try:
        yield
    finally:
        for task in background_tasks:
            task.cancel()
        for task in background_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        await attendee.close()


app = FastAPI(
    title="Google Meet Jira Task Assistant", version="0.1.0", lifespan=lifespan
)


async def require_app_auth(authorization: str | None = Header(default=None)) -> None:
    if not settings.app_api_token:
        raise HTTPException(status_code=503, detail="APP_API_TOKEN is not configured")
    expected = f"Bearer {settings.app_api_token}"
    if authorization is None or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Invalid bearer token")


@app.get("/health/live")
async def health_live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
async def health() -> JSONResponse:
    checks = db.health_checks()
    healthy = all(checks.values())
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "ok" if healthy else "not_ready",
            "checks": {
                name: "ok" if passed else "unavailable"
                for name, passed in checks.items()
            },
        },
    )


async def _probe_service(
    client: httpx.AsyncClient, url: str, *, headers: dict[str, str] | None = None
) -> str:
    if not url:
        return "not_configured"
    try:
        response = await client.get(url, headers=headers)
    except httpx.HTTPError:
        return "unavailable"
    return "healthy" if response.is_success else "unavailable"


@app.get("/dashboard/api/overview", dependencies=[Depends(require_app_auth)])
async def dashboard_overview(limit: int = 25) -> dict[str, Any]:
    local_checks = db.health_checks()
    attendee_parts = urlsplit(settings.attendee_api_base)
    attendee_health_url = (
        f"{attendee_parts.scheme}://{attendee_parts.netloc}/health/"
        if attendee_parts.scheme and attendee_parts.netloc
        else ""
    )
    model_base_url = settings.llm_base_url.rstrip("/")
    if model_base_url.endswith("/v1"):
        model_base_url = model_base_url[:-3]
    model_health_url = f"{model_base_url}/health" if model_base_url else ""
    model_headers = (
        {"Authorization": f"Bearer {settings.llm_api_key}"}
        if settings.llm_api_key
        else None
    )

    async with httpx.AsyncClient(timeout=httpx.Timeout(3, connect=1)) as client:
        attendee_status, model_status, transcriber_status = await asyncio.gather(
            _probe_service(client, attendee_health_url),
            _probe_service(client, model_health_url, headers=model_headers),
            _probe_service(client, settings.mlx_transcriber_health_url),
        )

    services = [
        {"name": "API", "status": "healthy"},
        {
            "name": "Database",
            "status": "healthy" if local_checks["database"] else "unavailable",
        },
        {
            "name": "Job worker",
            "status": "healthy" if local_checks["worker"] else "unavailable",
        },
        {"name": "Attendee", "status": attendee_status},
        {"name": "Task model", "status": model_status},
        {"name": "MLX Whisper", "status": transcriber_status},
        {
            "name": "Jira",
            "status": (
                "configured"
                if all(
                    (
                        settings.jira_base_url,
                        settings.jira_email,
                        settings.jira_api_token,
                        settings.jira_project_key,
                    )
                )
                else "not_configured"
            ),
        },
        {
            "name": "Calendar auto-join",
            "status": (
                "enabled" if settings.google_calendar_auto_join_enabled else "disabled"
            ),
        },
    ]
    critical_statuses = {"healthy"}
    overall_status = (
        "healthy"
        if all(service["status"] in critical_statuses for service in services[:6])
        else "degraded"
    )
    now = datetime.now(UTC)
    calendar_events: list[dict[str, Any]] = []
    if settings.google_calendar_auto_join_enabled:
        event_rows = db.list_due_calendar_events(
            settings.google_calendar_id,
            now.isoformat(),
            (now + timedelta(days=14)).isoformat(),
        )
        attempts = {
            (str(item["calendar_event_id"]), str(item["event_start"])): item
            for item in db.list_calendar_auto_joins(limit=100)
        }
        for event in event_rows:
            attempt = attempts.get(
                (str(event["calendar_event_id"]), str(event["event_start"])), {}
            )
            calendar_events.append(
                {
                    "event_id": event["calendar_event_id"],
                    "event_title": settings.google_calendar_event_title,
                    "event_start": event["event_start"],
                    "event_end": event.get("event_end"),
                    "meeting_url": event["meeting_url"],
                    "join_status": attempt.get("status", "scheduled"),
                    "bot_id": attempt.get("bot_id"),
                    "error": attempt.get("error"),
                }
            )
    return {
        "status": overall_status,
        "updated_at": now.isoformat(),
        "services": services,
        "recent_meetings": db.list_recent_meetings(limit),
        "calendar": {
            "enabled": settings.google_calendar_auto_join_enabled,
            "event_title_filter": settings.google_calendar_event_title or None,
            "time_zone": settings.google_calendar_timezone,
            "upcoming_events": calendar_events,
        },
    }


@app.post("/webhooks/google-calendar")
async def google_calendar_webhook(request: Request) -> Response:
    if not request.headers.get("X-Goog-Resource-State"):
        raise HTTPException(status_code=400, detail="Missing Calendar resource state")
    if not accept_calendar_notification(
        request.headers.get("X-Goog-Channel-ID"),
        request.headers.get("X-Goog-Channel-Token"),
        request.headers.get("X-Goog-Resource-ID"),
    ):
        raise HTTPException(status_code=401, detail="Invalid Calendar notification")
    return Response(status_code=204)


@app.get("/calendar/auto-join", dependencies=[Depends(require_app_auth)])
async def calendar_auto_join_status(limit: int = 20) -> dict[str, Any]:
    watch = db.get_current_calendar_watch(settings.google_calendar_id)
    push_is_active = watch is not None and int(watch["expiration_ms"]) > int(
        datetime.now(UTC).timestamp() * 1000
    )
    mode = (
        "disabled"
        if not settings.google_calendar_auto_join_enabled
        else ("push" if push_is_active else "polling")
    )
    return {
        "enabled": settings.google_calendar_auto_join_enabled,
        "mode": mode,
        "event_title": settings.google_calendar_event_title or None,
        "start_time": settings.google_calendar_start_time,
        "time_zone": settings.google_calendar_timezone,
        "recent_requests": db.list_calendar_auto_joins(limit),
    }


@app.post("/meetings", dependencies=[Depends(require_app_auth)])
async def start_meeting(body: MeetingStartRequest) -> dict[str, Any]:
    parsed = urlparse(str(body.meeting_url))
    if parsed.hostname not in {"meet.google.com", "www.meet.google.com"}:
        raise HTTPException(
            status_code=400, detail="Only Google Meet URLs are supported"
        )
    screen_share_capture = (
        settings.screenshare_capture_enabled
        if body.screen_share_capture is None
        else body.screen_share_capture
    )
    if screen_share_capture and not settings.screenshare_capture_enabled:
        raise HTTPException(
            status_code=400,
            detail="Screen-share capture is disabled by service configuration",
        )
    try:
        bot = await attendee.create_bot(
            str(body.meeting_url), screen_share_capture=screen_share_capture
        )
    except Exception as exc:
        logger.exception("Could not create Attendee bot")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    bot_id = str(bot["id"])
    db.create_meeting(
        bot_id,
        str(body.meeting_url),
        bot.get("state"),
        screenshare_capture_enabled=screen_share_capture,
    )
    return {
        "bot_id": bot_id,
        "state": bot.get("state"),
        "meeting_url": body.meeting_url,
        "screenshare_capture_enabled": screen_share_capture,
    }


@app.post("/meetings/{bot_id}/leave", dependencies=[Depends(require_app_auth)])
async def leave_meeting(bot_id: str) -> dict[str, Any]:
    claimed = db.claim_meeting_leave(
        bot_id, "manual", require_recording=False
    )
    if not claimed:
        meeting = db.get_meeting(bot_id)
        if meeting is not None and meeting["leave_status"] in {
            "requesting",
            "requested",
            "uncertain",
        }:
            return {"ok": True, "status": meeting["leave_status"]}
    try:
        result = await attendee.leave_bot(bot_id)
    except Exception as exc:
        if claimed:
            db.finish_meeting_leave(bot_id, "uncertain")
        logger.exception("Could not ask Attendee bot to leave")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if claimed:
        db.finish_meeting_leave(bot_id, "requested")
    return result


@app.get("/meetings/{bot_id}", dependencies=[Depends(require_app_auth)])
async def meeting_status(bot_id: str) -> dict[str, Any]:
    meeting = db.get_meeting(bot_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting


@app.get(
    "/meetings/{bot_id}/screen-frames/{frame_id}",
    dependencies=[Depends(require_app_auth)],
)
async def screen_share_frame(bot_id: str, frame_id: int) -> Response:
    frame = db.get_screen_frame(bot_id, frame_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="Screen-share frame not found")
    return Response(
        content=frame["frame_data"],
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.post("/meetings/{bot_id}/reprocess", dependencies=[Depends(require_app_auth)])
async def reprocess_meeting(bot_id: str) -> dict[str, str]:
    if db.get_meeting(bot_id) is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    status = db.enqueue_finalization(bot_id, allow_reprocess=True)
    if status == "completed":
        raise HTTPException(status_code=409, detail="Meeting is already completed")
    return {"bot_id": bot_id, "status": status}


def _timestamp_ms(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


def _event_is_final(payload: dict[str, Any]) -> bool:
    if payload.get("trigger") != "bot.state_change":
        return False
    data = payload.get("data") or {}
    return (
        isinstance(data, dict) and data.get("event_type") == "post_processing_completed"
    )


@app.post("/webhooks/attendee")
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
    db.record_webhook(
        event_id,
        bot_id,
        trigger,
        payload,
        enqueue_finalization=_event_is_final(payload),
    )
    event_data = payload.get("data") or {}
    if trigger == "bot.state_change" and isinstance(event_data, dict):
        new_state = event_data.get("new_state")
        if _event_is_final(payload):
            db.update_bot_state(bot_id, "post_processing_completed")
        elif new_state:
            db.update_bot_state(bot_id, str(new_state))
        if not _event_is_final(payload) and new_state == "joined_recording":
            recording_started_at_ms = _timestamp_ms(event_data.get("created_at"))
            if recording_started_at_ms is not None:
                db.set_recording_started_at(bot_id, recording_started_at_ms)
    if trigger == "transcript.update":
        segment = normalize_segments(event_data)
        db.save_segments(bot_id, segment)
    return {"status": "received"}


@app.websocket("/webhooks/attendee/video")
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
