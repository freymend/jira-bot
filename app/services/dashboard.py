from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit

import httpx

from app import db
from app.calendar_scheduler import calendar_meet
from app.config import settings
from app.integrations.google_calendar import CalendarEventsClient, GoogleCalendarClient
from app.schemas import (
    CalendarEventRow,
    DashboardCalendarEvent,
    DashboardOverview,
    DashboardService,
)

logger = logging.getLogger(__name__)

_DASHBOARD_CALENDAR_CACHE: tuple[str, float, list[DashboardCalendarEvent]] | None = None


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


def _calendar_cache_key() -> str:
    return "\n".join(
        (
            settings.google_calendar_id,
            settings.google_calendar_event_title,
            settings.google_calendar_organizer_email,
            settings.google_calendar_timezone,
            settings.google_calendar_start_time,
            str(settings.google_calendar_join_grace_minutes),
        )
    )


def _calendar_event_rows(
    rows: Sequence[CalendarEventRow],
) -> list[DashboardCalendarEvent]:
    attempts = {
        (str(item["calendar_event_id"]), str(item["event_start"])): item
        for item in db.list_calendar_auto_joins(limit=100)
    }
    result: list[DashboardCalendarEvent] = []
    for event in rows:
        attempt = attempts.get(
            (str(event["calendar_event_id"]), str(event["event_start"])), {}
        )
        result.append(
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
    return result


async def _dashboard_calendar_schedule(
    now: datetime,
) -> tuple[list[DashboardCalendarEvent], str, str | None]:
    global _DASHBOARD_CALENDAR_CACHE
    if not settings.google_calendar_auto_join_enabled:
        return [], "disabled", None

    start_at = now.isoformat()
    end_at = (now + timedelta(days=14)).isoformat()
    watch = db.get_current_calendar_watch(settings.google_calendar_id)
    push_is_active = watch is not None and int(watch["expiration_ms"]) > int(
        now.timestamp() * 1000
    )
    stored_rows = db.list_due_calendar_events(
        settings.google_calendar_id, start_at, end_at
    )
    if push_is_active and stored_rows:
        return _calendar_event_rows(stored_rows), "push_sync", None

    cache_key = _calendar_cache_key()
    if (
        _DASHBOARD_CALENDAR_CACHE is not None
        and _DASHBOARD_CALENDAR_CACHE[0] == cache_key
        and _DASHBOARD_CALENDAR_CACHE[1] > time.monotonic()
    ):
        cached = _DASHBOARD_CALENDAR_CACHE[2]
        return cached, "calendar_api_cache", None

    calendar: CalendarEventsClient = GoogleCalendarClient()
    try:
        events = await calendar.list_events(now, now + timedelta(days=14))
        scheduled = [
            candidate
            for event in events
            if (candidate := calendar_meet(event)) is not None
        ]
        rows: list[CalendarEventRow] = [
            {
                "calendar_event_id": candidate.event_id,
                "event_start": candidate.event_start,
                "event_end": candidate.event_end,
                "meeting_url": candidate.meeting_url,
            }
            for candidate in scheduled
        ]
        result = _calendar_event_rows(rows)
        _DASHBOARD_CALENDAR_CACHE = (
            cache_key,
            time.monotonic() + 60,
            result,
        )
        return result, "calendar_api", None
    except Exception:
        logger.warning(
            "Could not refresh upcoming Google Calendar events for the dashboard",
            exc_info=True,
        )
        fallback = _calendar_event_rows(stored_rows)
        return (
            fallback,
            "push_sync" if push_is_active else "unavailable",
            "Calendar refresh failed; showing any last-synchronized events.",
        )
    finally:
        await calendar.close()


async def dashboard_overview(limit: int = 25) -> DashboardOverview:
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

    services: list[DashboardService] = [
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
    now = datetime.now(UTC)
    (
        calendar_events,
        calendar_source,
        calendar_error,
    ) = await _dashboard_calendar_schedule(now)
    calendar_service = services[-1]
    if settings.google_calendar_auto_join_enabled:
        calendar_service["status"] = "unavailable" if calendar_error else "healthy"
        calendar_service["detail"] = (
            calendar_error or f"Upcoming events loaded from {calendar_source}."
        )
    critical_statuses = {"healthy"}
    services_healthy = all(
        service["status"] in critical_statuses for service in services[:6]
    )
    calendar_healthy = (
        not settings.google_calendar_auto_join_enabled or not calendar_error
    )
    overall_status = "healthy" if services_healthy and calendar_healthy else "degraded"
    return {
        "status": overall_status,
        "updated_at": now.isoformat(),
        "services": services,
        "recent_meetings": db.list_recent_meetings(limit),
        "calendar": {
            "enabled": settings.google_calendar_auto_join_enabled,
            "event_title_filter": settings.google_calendar_event_title or None,
            "start_time_filter": settings.google_calendar_start_time,
            "organizer_email_filter": settings.google_calendar_organizer_email or None,
            "time_zone": settings.google_calendar_timezone,
            "source": calendar_source,
            "error": calendar_error,
            "upcoming_events": calendar_events,
        },
    }
