from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from app import db
from app.calendar_scheduler import run_calendar_auto_join
from app.config import settings
from app.integrations.attendee import verify_attendee_signature
from app.integrations.transcript import normalize_segments
from app.pipeline import attendee
from app.schemas import MeetingStartRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jira-bot")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    db.init_db()
    calendar_task: asyncio.Task[None] | None = None
    if settings.google_calendar_auto_join_enabled:
        calendar_task = asyncio.create_task(
            run_calendar_auto_join(attendee), name="google-calendar-auto-join"
        )
    try:
        yield
    finally:
        if calendar_task is not None:
            calendar_task.cancel()
            try:
                await calendar_task
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


@app.get("/calendar/auto-join", dependencies=[Depends(require_app_auth)])
async def calendar_auto_join_status(limit: int = 20) -> dict[str, Any]:
    return {
        "enabled": settings.google_calendar_auto_join_enabled,
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
    try:
        bot = await attendee.create_bot(str(body.meeting_url))
    except Exception as exc:
        logger.exception("Could not create Attendee bot")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    bot_id = str(bot["id"])
    db.create_meeting(bot_id, str(body.meeting_url), bot.get("state"))
    return {
        "bot_id": bot_id,
        "state": bot.get("state"),
        "meeting_url": body.meeting_url,
    }


@app.post("/meetings/{bot_id}/leave", dependencies=[Depends(require_app_auth)])
async def leave_meeting(bot_id: str) -> dict[str, Any]:
    try:
        return await attendee.leave_bot(bot_id)
    except Exception as exc:
        logger.exception("Could not ask Attendee bot to leave")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/meetings/{bot_id}", dependencies=[Depends(require_app_auth)])
async def meeting_status(bot_id: str) -> dict[str, Any]:
    meeting = db.get_meeting(bot_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting


@app.post("/meetings/{bot_id}/reprocess", dependencies=[Depends(require_app_auth)])
async def reprocess_meeting(bot_id: str) -> dict[str, str]:
    if db.get_meeting(bot_id) is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    status = db.enqueue_finalization(bot_id, allow_reprocess=True)
    if status == "completed":
        raise HTTPException(status_code=409, detail="Meeting is already completed")
    return {"bot_id": bot_id, "status": status}


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
        if new_state:
            db.update_bot_state(bot_id, str(new_state))
    if trigger == "transcript.update":
        segment = normalize_segments(event_data)
        db.save_segments(bot_id, segment)
    return {"status": "received"}
