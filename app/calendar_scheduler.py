from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from app import db
from app.config import settings
from app.integrations.google_calendar import GoogleCalendarClient

logger = logging.getLogger("jira-bot.calendar")


@dataclass(frozen=True)
class ScheduledMeet:
    event_key: str
    event_id: str
    event_start: str
    meeting_url: str


def _parse_event_start(event: dict[str, Any]) -> datetime | None:
    start = event.get("start")
    if not isinstance(start, dict):
        return None
    value = start.get("dateTime")
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=ZoneInfo(
                    str(start.get("timeZone") or settings.google_calendar_timezone)
                )
            )
        return parsed
    except (ValueError, KeyError):
        return None


def _meet_url(event: dict[str, Any]) -> str | None:
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


def _event_key(event: dict[str, Any], start: datetime) -> tuple[str, str] | None:
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
        series_id = recurring_event_id
        occurrence_start = str(original_value or start.isoformat())
    else:
        series_id = event_id
        occurrence_start = "single-event"
    key_material = "\n".join(
        (settings.google_calendar_id, series_id, occurrence_start)
    )
    return hashlib.sha256(key_material.encode("utf-8")).hexdigest(), event_id


def scheduled_meet(
    event: dict[str, Any], now: datetime | None = None
) -> ScheduledMeet | None:
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
    key_and_id = _event_key(event, start) if start is not None else None
    if start is None or meeting_url is None or key_and_id is None:
        return None

    current = now or datetime.now(UTC)
    local_start = start.astimezone(ZoneInfo(settings.google_calendar_timezone))
    target_time = time.fromisoformat(settings.google_calendar_start_time)
    if (local_start.hour, local_start.minute) != (target_time.hour, target_time.minute):
        return None
    if start < current - timedelta(minutes=settings.google_calendar_join_grace_minutes):
        return None
    if start > current + timedelta(minutes=settings.google_calendar_join_lead_minutes):
        return None

    event_key, event_id = key_and_id
    return ScheduledMeet(
        event_key=event_key,
        event_id=event_id,
        event_start=start.isoformat(),
        meeting_url=meeting_url,
    )


async def request_due_meetings(
    events: list[dict[str, Any]], attendee_client: Any, now: datetime | None = None
) -> int:
    requested = 0
    for event in events:
        candidate = scheduled_meet(event, now)
        if candidate is None:
            continue
        if not db.claim_calendar_auto_join(
            candidate.event_key, candidate.event_id, candidate.event_start
        ):
            continue

        bot_id: str | None = None
        try:
            bot = await attendee_client.create_bot(candidate.meeting_url)
            bot_id = str(bot["id"])
            db.create_meeting(
                bot_id, candidate.meeting_url, bot.get("state")
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


async def run_calendar_auto_join(attendee_client: Any) -> None:
    if not settings.google_calendar_auto_join_enabled:
        return

    calendar = GoogleCalendarClient()
    try:
        logger.info(
            "Google Calendar auto-join enabled for %s at %s %s",
            settings.google_calendar_event_title,
            settings.google_calendar_start_time,
            settings.google_calendar_timezone,
        )
        while True:
            db.mark_stale_calendar_auto_joins_uncertain()
            now = datetime.now(UTC)
            time_min = now - timedelta(
                minutes=settings.google_calendar_join_grace_minutes
            )
            time_max = now + timedelta(
                minutes=settings.google_calendar_join_lead_minutes
            )
            try:
                events = await calendar.list_events(time_min, time_max)
                await request_due_meetings(events, attendee_client, now)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Google Calendar auto-join poll failed")
            await asyncio.sleep(settings.google_calendar_poll_seconds)
    finally:
        await calendar.close()
