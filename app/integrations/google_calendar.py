from __future__ import annotations

import time
from datetime import datetime
from typing import Any
from urllib.parse import quote

import httpx

from app.config import settings


class GoogleCalendarClient:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(20, connect=10),
            headers={"Accept": "application/json"},
        )
        self._owns_client = client is None
        self._access_token: str | None = None
        self._access_token_expires_at = 0.0

    async def _get_access_token(self) -> str:
        if self._access_token and time.monotonic() < self._access_token_expires_at - 60:
            return self._access_token

        response = await self.client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": settings.google_calendar_client_id,
                "client_secret": settings.google_calendar_client_secret,
                "refresh_token": settings.google_calendar_refresh_token,
                "grant_type": "refresh_token",
            },
        )
        response.raise_for_status()
        token_response = response.json()
        access_token = token_response.get("access_token")
        expires_in = token_response.get("expires_in")
        if not isinstance(access_token, str) or not isinstance(expires_in, int):
            raise RuntimeError("Google OAuth returned an invalid access token response")
        self._access_token = access_token
        self._access_token_expires_at = time.monotonic() + expires_in
        return access_token

    async def list_events(
        self, time_min: datetime, time_max: datetime
    ) -> list[dict[str, Any]]:
        access_token = await self._get_access_token()
        url = (
            "https://www.googleapis.com/calendar/v3/calendars/"
            f"{quote(settings.google_calendar_id, safe='')}/events"
        )
        params: dict[str, str | int] = {
            "timeMin": time_min.isoformat(),
            "timeMax": time_max.isoformat(),
            "singleEvents": "true",
            "orderBy": "startTime",
            "showDeleted": "false",
            "maxResults": 250,
        }
        events: list[dict[str, Any]] = []
        while True:
            response = await self.client.get(
                url,
                params=params,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            response.raise_for_status()
            payload = response.json()
            items = payload.get("items", [])
            if not isinstance(items, list) or not all(
                isinstance(item, dict) for item in items
            ):
                raise RuntimeError("Google Calendar returned an invalid events list")
            events.extend(items)
            page_token = payload.get("nextPageToken")
            if not isinstance(page_token, str) or not page_token:
                break
            params["pageToken"] = page_token
        return events

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
