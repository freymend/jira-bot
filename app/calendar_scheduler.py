from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import secrets
import time as monotonic_time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from typing import Literal
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import httpx

from app import db
from app.config import settings
from app.integrations.attendee import AttendeeCalendarClient
from app.integrations.google_calendar import (
    CalendarSyncClient,
    CalendarWatchClient,
    GoogleCalendarClient,
    GoogleCalendarClientProtocol,
)
from app.schemas import CalendarEvent, CalendarSyncChange

logger = logging.getLogger("jira-bot.calendar")
_SYNC_WAKEUP: asyncio.Event | None = None


def set_calendar_sync_wakeup(event: asyncio.Event | None) -> None:
    global _SYNC_WAKEUP
    _SYNC_WAKEUP = event


def accept_calendar_notification(
    channel_id: str | None, channel_token: str | None, resource_id: str | None
) -> bool:
    if not channel_id or not channel_token or not resource_id:
        return False
    channel = db.get_calendar_watch_channel(channel_id)
    if channel is None:
        return False
    if channel["calendar_id"] != settings.google_calendar_id:
        return False
    if channel["status"] not in {"pending", "active"}:
        return False
    if not hmac.compare_digest(str(channel["channel_token"]), channel_token):
        return False
    expected_resource_id = channel.get("resource_id")
    if expected_resource_id and not hmac.compare_digest(
        str(expected_resource_id), resource_id
    ):
        return False
    if _SYNC_WAKEUP is not None:
        _SYNC_WAKEUP.set()
    return True


@dataclass(frozen=True)
class ScheduledMeet:
    event_key: str
    event_id: str
    event_start: str
    meeting_url: str
    event_end: str | None = None


def _parse_event_datetime(
    event: CalendarEvent, key: Literal["start", "end"]
) -> datetime | None:
    value = event.get(key)
    if not isinstance(value, dict) or not isinstance(value.get("dateTime"), str):
        return None
    try:
        parsed = datetime.fromisoformat(value["dateTime"].replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=ZoneInfo(
                    str(value.get("timeZone") or settings.google_calendar_timezone)
                )
            )
        return parsed
    except ValueError, KeyError:
        return None


def _parse_event_start(event: CalendarEvent) -> datetime | None:
    return _parse_event_datetime(event, "start")


def _parse_event_end(event: CalendarEvent) -> datetime | None:
    return _parse_event_datetime(event, "end")


def _meet_url(event: CalendarEvent) -> str | None:
    candidates: list[str] = []
    hangout_link = event.get("hangoutLink")
    if isinstance(hangout_link, str):
        candidates.append(hangout_link)
    conference = event.get("conferenceData")
    if isinstance(conference, dict):
        entry_points = conference.get("entryPoints")
        if isinstance(entry_points, list):
            candidates.extend(
                str(entry.get("uri"))
                for entry in entry_points
                if isinstance(entry, dict)
                and entry.get("entryPointType") == "video"
                and isinstance(entry.get("uri"), str)
            )

    for candidate in candidates:
        parsed = urlsplit(candidate)
        if (
            parsed.scheme == "https"
            and parsed.hostname in {"meet.google.com", "www.meet.google.com"}
            and parsed.username is None
            and parsed.password is None
        ):
            return candidate
    return None


def _event_key(event: CalendarEvent, start: datetime | None) -> tuple[str, str] | None:
    event_id = event.get("id")
    if not isinstance(event_id, str) or not event_id:
        return None
    recurring_event_id = event.get("recurringEventId")
    original_start = event.get("originalStartTime")
    original_value = (
        original_start.get("dateTime") or original_start.get("date")
        if isinstance(original_start, dict)
        else None
    )
    if isinstance(recurring_event_id, str):
        if original_value is not None:
            occurrence_start = str(original_value)
        elif start is not None:
            occurrence_start = start.isoformat()
        else:
            return None
        series_id = recurring_event_id
    else:
        series_id = event_id
        occurrence_start = "single-event"
    key_material = "\n".join((settings.google_calendar_id, series_id, occurrence_start))
    return hashlib.sha256(key_material.encode("utf-8")).hexdigest(), event_id


def calendar_meet(event: CalendarEvent) -> ScheduledMeet | None:
    if event.get("status") == "cancelled":
        return None
    title = event.get("summary")
    if not isinstance(title, str) or title.strip().casefold() != (
        settings.google_calendar_event_title.strip().casefold()
    ):
        return None
    if settings.google_calendar_organizer_email:
        organizer = event.get("organizer")
        organizer_email = (
            organizer.get("email", "").strip().casefold()
            if isinstance(organizer, dict)
            else ""
        )
        if organizer_email != settings.google_calendar_organizer_email:
            return None

    start = _parse_event_start(event)
    meeting_url = _meet_url(event)
    key_and_id = _event_key(event, start)
    if start is None or meeting_url is None or key_and_id is None:
        return None

    meeting_local_start = start.astimezone(ZoneInfo(settings.google_calendar_timezone))
    target_time = time.fromisoformat(settings.google_calendar_start_time)
    if (meeting_local_start.hour, meeting_local_start.minute) != (
        target_time.hour,
        target_time.minute,
    ):
        return None
    if start < datetime.now(UTC) - timedelta(
        minutes=settings.google_calendar_join_grace_minutes
    ):
        return None

    event_key, event_id = key_and_id
    event_end = _parse_event_end(event)
    return ScheduledMeet(
        event_key=event_key,
        event_id=event_id,
        event_start=start.astimezone(UTC).isoformat(),
        meeting_url=meeting_url,
        event_end=(event_end.astimezone(UTC).isoformat() if event_end else None),
    )


def _is_due(candidate: ScheduledMeet, now: datetime) -> bool:
    start = datetime.fromisoformat(candidate.event_start)
    return start >= now - timedelta(
        minutes=settings.google_calendar_join_grace_minutes
    ) and start <= now + timedelta(minutes=settings.google_calendar_join_lead_minutes)


def scheduled_meet(
    event: CalendarEvent, now: datetime | None = None
) -> ScheduledMeet | None:
    candidate = calendar_meet(event)
    if candidate is None or not _is_due(candidate, now or datetime.now(UTC)):
        return None
    return candidate


async def _request_candidates(
    candidates: list[ScheduledMeet], attendee_client: AttendeeCalendarClient
) -> int:
    requested = 0
    for candidate in candidates:
        if not db.claim_calendar_auto_join(
            candidate.event_key, candidate.event_id, candidate.event_start
        ):
            continue

        bot_id: str | None = None
        try:
            bot = await attendee_client.create_bot(candidate.meeting_url)
            bot_id = str(bot["id"])
            db.create_meeting(
                bot_id,
                candidate.meeting_url,
                bot.get("state"),
                screenshare_capture_enabled=settings.screenshare_capture_enabled,
                scheduled_end_at=candidate.event_end,
                participant_tracking_enabled=True,
            )
            db.finish_calendar_auto_join(
                candidate.event_key, "requested", bot_id=bot_id
            )
            requested += 1
            logger.info(
                "Requested Attendee for scheduled event at %s",
                candidate.event_start,
            )
        except Exception as exc:
            db.finish_calendar_auto_join(
                candidate.event_key,
                "uncertain",
                bot_id=bot_id,
                error=str(exc),
            )
            logger.exception(
                "Scheduled Attendee request is uncertain for event %s",
                candidate.event_id,
            )
    return requested


async def request_due_meetings(
    events: list[CalendarEvent],
    attendee_client: AttendeeCalendarClient,
    now: datetime | None = None,
) -> int:
    current = now or datetime.now(UTC)
    candidates = [
        candidate
        for event in events
        if (candidate := scheduled_meet(event, current)) is not None
    ]
    return await _request_candidates(candidates, attendee_client)


async def request_stored_due_meetings(
    attendee_client: AttendeeCalendarClient, now: datetime | None = None
) -> int:
    current = now or datetime.now(UTC)
    time_min = (
        current - timedelta(minutes=settings.google_calendar_join_grace_minutes)
    ).isoformat()
    time_max = (
        current + timedelta(minutes=settings.google_calendar_join_lead_minutes)
    ).isoformat()
    rows = db.list_due_calendar_events(settings.google_calendar_id, time_min, time_max)
    candidates = [
        ScheduledMeet(
            event_key=str(row["event_key"]),
            event_id=str(row["calendar_event_id"]),
            event_start=str(row["event_start"]),
            meeting_url=str(row["meeting_url"]),
            event_end=(str(row["event_end"]) if row.get("event_end") else None),
        )
        for row in rows
    ]
    return await _request_candidates(candidates, attendee_client)


def _calendar_callback_url() -> str | None:
    parsed = urlsplit(settings.app_public_url)
    if parsed.scheme != "https" or not parsed.hostname:
        return None
    return f"{settings.app_public_url.rstrip('/')}/webhooks/google-calendar"


async def _ensure_calendar_watch(calendar: CalendarWatchClient) -> bool:
    callback_url = _calendar_callback_url()
    if callback_url is None:
        return False

    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    current = db.get_current_calendar_watch(settings.google_calendar_id)
    if (
        current is not None
        and int(current["expiration_ms"]) > now_ms + 24 * 60 * 60 * 1000
    ):
        return True

    channel_id = uuid.uuid4().hex
    channel_token = secrets.token_urlsafe(32)
    requested_expiration = int(
        (datetime.now(UTC) + timedelta(days=6)).timestamp() * 1000
    )
    db.create_calendar_watch_channel(
        channel_id,
        settings.google_calendar_id,
        channel_token,
        requested_expiration,
    )
    try:
        result = await calendar.watch_events(
            channel_id, callback_url, channel_token, requested_expiration
        )
    except Exception:
        db.delete_calendar_watch_channel(channel_id)
        if current is not None and int(current["expiration_ms"]) > now_ms:
            logger.exception(
                "Could not renew Google Calendar watch; keeping old channel"
            )
            return True
        logger.exception("Could not create Google Calendar watch channel")
        return False

    resource_id = result.get("resourceId")
    expiration = result.get("expiration", requested_expiration)
    if not isinstance(resource_id, str) or not resource_id:
        db.delete_calendar_watch_channel(channel_id)
        logger.error("Google Calendar watch response omitted resourceId")
        return current is not None and int(current["expiration_ms"]) > now_ms
    try:
        if not isinstance(expiration, str | int | float):
            raise TypeError
        expiration_ms = int(expiration)
    except TypeError, ValueError:
        expiration_ms = requested_expiration
    db.activate_calendar_watch_channel(channel_id, resource_id, expiration_ms)

    if current is not None:
        try:
            if current.get("resource_id"):
                await calendar.stop_watching(
                    str(current["channel_id"]), str(current["resource_id"])
                )
            db.delete_calendar_watch_channel(str(current["channel_id"]))
        except Exception:
            logger.warning(
                "Old Google Calendar watch could not be stopped", exc_info=True
            )
    logger.info("Google Calendar push channel active until %s", expiration_ms)
    return True


def _calendar_sync_change(
    event: CalendarEvent, *, full_sync: bool
) -> CalendarSyncChange | None:
    identity = _event_key(event, _parse_event_start(event))
    if identity is None:
        return None
    event_key, event_id = identity
    candidate = calendar_meet(event)
    if candidate is not None:
        return {
            "action": "upsert",
            "event_key": event_key,
            "calendar_event_id": event_id,
            "event_start": candidate.event_start,
            "event_end": candidate.event_end,
            "meeting_url": candidate.meeting_url,
        }
    if full_sync:
        return None
    return {"action": "delete", "event_key": event_key}


async def sync_calendar_events(calendar: CalendarSyncClient) -> int:
    calendar_id = settings.google_calendar_id
    sync_token = db.get_calendar_sync_token(calendar_id)
    full_sync = sync_token is None
    try:
        events, next_sync_token = await calendar.sync_events(sync_token)
    except httpx.HTTPStatusError as exc:
        if sync_token is None or exc.response.status_code != 410:
            raise
        db.clear_calendar_sync_token(calendar_id)
        full_sync = True
        events, next_sync_token = await calendar.sync_events(None)

    changes = [
        change
        for event in events
        if (change := _calendar_sync_change(event, full_sync=full_sync)) is not None
    ]
    db.apply_calendar_sync(calendar_id, changes, next_sync_token, replace_all=full_sync)
    logger.info("Synced %s Google Calendar changes", len(changes))
    return len(changes)


async def run_calendar_auto_join(
    attendee_client: AttendeeCalendarClient, sync_wakeup: asyncio.Event
) -> None:
    if not settings.google_calendar_auto_join_enabled:
        return

    set_calendar_sync_wakeup(sync_wakeup)
    calendar: GoogleCalendarClientProtocol = GoogleCalendarClient()
    try:
        logger.info(
            "Google Calendar auto-join enabled for %s at %s %s",
            settings.google_calendar_event_title,
            settings.google_calendar_start_time,
            settings.google_calendar_timezone,
        )
        push_mode = await _ensure_calendar_watch(calendar)
        next_watch_check = monotonic_time.monotonic() + 3600
        next_sync = monotonic_time.monotonic()
        if push_mode:
            sync_wakeup.clear()
            try:
                await sync_calendar_events(calendar)
                next_sync = monotonic_time.monotonic() + 6 * 60 * 60
            except Exception:
                logger.exception("Initial Google Calendar sync failed")
                next_sync = monotonic_time.monotonic() + 60

        while True:
            db.mark_stale_calendar_auto_joins_uncertain()
            now = datetime.now(UTC)
            try:
                monotonic_now = monotonic_time.monotonic()
                if push_mode:
                    await request_stored_due_meetings(attendee_client, now)
                    if sync_wakeup.is_set() or monotonic_now >= next_sync:
                        sync_wakeup.clear()
                        try:
                            await sync_calendar_events(calendar)
                            next_sync = monotonic_time.monotonic() + 6 * 60 * 60
                        except Exception:
                            logger.exception("Google Calendar incremental sync failed")
                            next_sync = monotonic_time.monotonic() + 60
                    if monotonic_now >= next_watch_check:
                        push_mode = await _ensure_calendar_watch(calendar)
                        next_watch_check = monotonic_time.monotonic() + 3600
                else:
                    time_min = now - timedelta(
                        minutes=settings.google_calendar_join_grace_minutes
                    )
                    time_max = now + timedelta(
                        minutes=settings.google_calendar_join_lead_minutes
                    )
                    events = await calendar.list_events(time_min, time_max)
                    await request_due_meetings(events, attendee_client, now)
                    if monotonic_now >= next_watch_check:
                        push_mode = await _ensure_calendar_watch(calendar)
                        next_watch_check = monotonic_time.monotonic() + 3600
                        if push_mode:
                            sync_wakeup.clear()
                            try:
                                await sync_calendar_events(calendar)
                                next_sync = monotonic_time.monotonic() + 6 * 60 * 60
                            except Exception:
                                logger.exception("Initial Google Calendar sync failed")
                                next_sync = monotonic_time.monotonic() + 60
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Google Calendar auto-join processing failed")
            await asyncio.sleep(
                10 if push_mode else settings.google_calendar_poll_seconds
            )
    finally:
        set_calendar_sync_wakeup(None)
        await calendar.close()
