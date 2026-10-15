from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from app import db
from app.calendar_scheduler import accept_calendar_notification
from app.config import settings
from app.controllers.dependencies import require_app_auth
from app.schemas import CalendarAutoJoinStatus

router = APIRouter()


@router.post("/webhooks/google-calendar")
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


@router.get("/calendar/auto-join", dependencies=[Depends(require_app_auth)])
async def calendar_auto_join_status(limit: int = 20) -> CalendarAutoJoinStatus:
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
