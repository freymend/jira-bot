from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import db
from app.calendar_scheduler import run_calendar_auto_join
from app.config import settings
from app.controllers.calendar import (
    calendar_auto_join_status,
    google_calendar_webhook,
)
from app.controllers.calendar import (
    router as calendar_router,
)
from app.controllers.dashboard import router as dashboard_router
from app.controllers.dependencies import require_app_auth
from app.controllers.health import health, health_live
from app.controllers.health import router as health_router
from app.controllers.meetings import (
    leave_meeting,
    meeting_status,
    reprocess_meeting,
    screen_share_frame,
    start_meeting,
)
from app.controllers.meetings import (
    router as meetings_router,
)
from app.controllers.webhooks import (
    attendee_video_websocket,
    attendee_webhook,
)
from app.controllers.webhooks import (
    router as webhooks_router,
)
from app.meeting_lifecycle import (
    run_meeting_leave_monitor,
    run_meeting_leave_watchdog,
)
from app.pipeline import attendee
from app.services.dashboard import (
    _DASHBOARD_CALENDAR_CACHE,
    _calendar_cache_key,
    _calendar_event_rows,
    _dashboard_calendar_schedule,
    _probe_service,
    dashboard_overview,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jira-bot")
_BACKGROUND_TASK_RETRY_SECONDS = 5


async def _run_resilient_background_task(
    name: str, runner: Callable[[], Awaitable[None]]
) -> None:
    while True:
        try:
            await runner()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Background task %s failed; restarting", name)
        else:
            logger.error("Background task %s exited; restarting", name)
        await asyncio.sleep(_BACKGROUND_TASK_RETRY_SECONDS)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    db.delete_expired_screen_frames()
    background_tasks: list[asyncio.Task[None]] = [
        asyncio.create_task(
            _run_resilient_background_task(
                "meeting-leave-watchdog",
                lambda: run_meeting_leave_watchdog(attendee),
            ),
            name="meeting-leave-watchdog",
        )
    ]
    if settings.google_calendar_auto_join_enabled:
        calendar_sync_wakeup = asyncio.Event()
        background_tasks.append(
            asyncio.create_task(
                _run_resilient_background_task(
                    "google-calendar-auto-join",
                    lambda: run_calendar_auto_join(attendee, calendar_sync_wakeup),
                ),
                name="google-calendar-auto-join",
            )
        )
    if settings.meeting_auto_leave_enabled:
        background_tasks.append(
            asyncio.create_task(
                _run_resilient_background_task(
                    "meeting-auto-leave",
                    lambda: run_meeting_leave_monitor(attendee),
                ),
                name="meeting-auto-leave",
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
app.include_router(health_router)
app.include_router(dashboard_router)
app.include_router(calendar_router)
app.include_router(meetings_router)
app.include_router(webhooks_router)

__all__ = [
    "_DASHBOARD_CALENDAR_CACHE",
    "_BACKGROUND_TASK_RETRY_SECONDS",
    "_calendar_cache_key",
    "_calendar_event_rows",
    "_dashboard_calendar_schedule",
    "_probe_service",
    "_run_resilient_background_task",
    "app",
    "attendee",
    "attendee_video_websocket",
    "attendee_webhook",
    "calendar_auto_join_status",
    "dashboard_overview",
    "google_calendar_webhook",
    "health",
    "health_live",
    "leave_meeting",
    "lifespan",
    "logger",
    "meeting_status",
    "reprocess_meeting",
    "require_app_auth",
    "screen_share_frame",
    "settings",
    "start_meeting",
]
