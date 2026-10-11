from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
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

    @staticmethod
    def _raise_for_google_error(response: httpx.Response) -> None:
        if not response.is_error:
            return
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            message = error.get("message")
            errors = error.get("errors")
            reason = next(
                (
                    item.get("reason")
                    for item in errors
                    if isinstance(item, dict) and isinstance(item.get("reason"), str)
                ),
                None,
            ) if isinstance(errors, list) else None
            detail = f"Google Calendar returned HTTP {response.status_code}"
            if isinstance(reason, str):
                detail += f" ({reason})"
            if isinstance(message, str):
                detail += f": {message}"
            raise httpx.HTTPStatusError(
                detail, request=response.request, response=response
            )
        response.raise_for_status()

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
        self._raise_for_google_error(response)
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
            self._raise_for_google_error(response)
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

    async def sync_events(
        self, sync_token: str | None
    ) -> tuple[list[dict[str, Any]], str]:
        access_token = await self._get_access_token()
        url = (
            "https://www.googleapis.com/calendar/v3/calendars/"
            f"{quote(settings.google_calendar_id, safe='')}/events"
        )
        params: dict[str, str | int] = {
            "singleEvents": "true",
            "showDeleted": "true",
            "maxResults": 2500,
        }
        if sync_token:
            params["syncToken"] = sync_token
        else:
            params["timeMin"] = (
                datetime.now(UTC)
                - timedelta(minutes=settings.google_calendar_join_grace_minutes + 1)
            ).isoformat()

        events: list[dict[str, Any]] = []
        page_token: str | None = None
        next_sync_token: str | None = None
        while True:
            request_params = dict(params)
            if page_token:
                request_params["pageToken"] = page_token
            response = await self.client.get(
                url,
                params=request_params,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            self._raise_for_google_error(response)
            payload = response.json()
            items = payload.get("items", [])
            if not isinstance(items, list) or not all(
                isinstance(item, dict) for item in items
            ):
                raise RuntimeError("Google Calendar returned an invalid events list")
            events.extend(items)
            page_token_value = payload.get("nextPageToken")
            if isinstance(page_token_value, str) and page_token_value:
                page_token = page_token_value
                continue
            token_value = payload.get("nextSyncToken")
            if not isinstance(token_value, str) or not token_value:
                raise RuntimeError("Google Calendar sync response had no nextSyncToken")
            next_sync_token = token_value
            break
        return events, next_sync_token

    async def watch_events(
        self,
        channel_id: str,
        callback_url: str,
        channel_token: str,
        expiration_ms: int,
    ) -> dict[str, Any]:
        access_token = await self._get_access_token()
        url = (
            "https://www.googleapis.com/calendar/v3/calendars/"
            f"{quote(settings.google_calendar_id, safe='')}/events/watch"
        )
        response = await self.client.post(
            url,
            json={
                "id": channel_id,
                "type": "web_hook",
                "address": callback_url,
                "token": channel_token,
                "expiration": expiration_ms,
            },
            headers={"Authorization": f"Bearer {access_token}"},
        )
        self._raise_for_google_error(response)
        result = response.json()
        if not isinstance(result, dict):
            raise RuntimeError("Google Calendar returned an invalid watch response")
        return result

    async def stop_watching(self, channel_id: str, resource_id: str) -> None:
        access_token = await self._get_access_token()
        response = await self.client.post(
            "https://www.googleapis.com/calendar/v3/channels/stop",
            json={"id": channel_id, "resourceId": resource_id},
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if response.status_code != 404:
            self._raise_for_google_error(response)

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
