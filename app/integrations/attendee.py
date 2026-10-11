from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

import httpx

from app.config import settings


class AttendeeClient:
    def __init__(self) -> None:
        self.client = httpx.AsyncClient(
            base_url=settings.attendee_api_base.rstrip("/") + "/",
            headers={"Authorization": f"Token {settings.attendee_api_key}"},
            timeout=httpx.Timeout(60, connect=15, write=30, pool=15),
        )

    async def create_bot(
        self,
        meeting_url: str,
        *,
        screen_share_capture: bool | None = None,
    ) -> dict[str, Any]:
        capture_screen_share = (
            settings.screenshare_capture_enabled
            if screen_share_capture is None
            else screen_share_capture
        )
        if capture_screen_share and not settings.screenshare_capture_enabled:
            raise RuntimeError("Screen-share capture is disabled by configuration")
        if not settings.attendee_api_key:
            raise RuntimeError("ATTENDEE_API_KEY is not configured")
        callback_url = urlsplit(settings.app_public_url)
        is_public_https_url = (
            callback_url.scheme == "https" and callback_url.hostname is not None
        )
        is_local_compose_url = (
            callback_url.scheme == "http"
            and callback_url.hostname == "api"
            and callback_url.port == 5005
        )
        if not (is_public_https_url or is_local_compose_url):
            raise RuntimeError(
                "APP_PUBLIC_URL must be a public HTTPS URL or http://api:5005 "
                "on the local Compose network"
            )
        if capture_screen_share and not is_public_https_url:
            raise RuntimeError(
                "Screen-share capture requires APP_PUBLIC_URL to be a public "
                "HTTPS URL because Attendee requires a wss:// video endpoint"
            )
        if not settings.mlx_transcriber_api_key:
            raise RuntimeError("MLX_TRANSCRIBER_API_KEY is not configured")
        transcription_settings: dict[str, Any] = {
            "custom_async_v2": {
                "headers": {
                    "Authorization": f"Bearer {settings.mlx_transcriber_api_key}"
                }
            }
        }
        if settings.mlx_whisper_language:
            transcription_settings["custom_async_v2"]["form_data"] = {
                "language": settings.mlx_whisper_language
            }
        bot_settings: dict[str, Any] = {
            "meeting_url": meeting_url,
            "bot_name": settings.attendee_bot_name,
            "transcription_settings": transcription_settings,
            "webhooks": [
                {
                    "url": f"{settings.app_public_url}/webhooks/attendee",
                    "triggers": [
                        "bot.state_change",
                        "transcript.update",
                        "participant_events.join_leave",
                    ],
                }
            ],
        }
        if capture_screen_share:
            websocket_path = callback_url.path.rstrip("/") + "/webhooks/attendee/video"
            websocket_url = urlunsplit(
                (
                    "wss",
                    callback_url.netloc,
                    websocket_path,
                    urlencode({"token": settings.screenshare_websocket_token}),
                    "",
                )
            )
            bot_settings["websocket_settings"] = {
                "per_participant_video": {
                    "url": websocket_url,
                    "webcam_resolution": "none",
                    "screenshare_resolution": settings.screenshare_resolution,
                }
            }
        response = await self.client.post("bots", json=bot_settings)
        response.raise_for_status()
        return response.json()

    async def leave_bot(self, bot_id: str) -> dict[str, Any]:
        response = await self.client.post(f"bots/{bot_id}/leave")
        response.raise_for_status()
        return response.json() if response.content else {"ok": True}

    async def get_bot(self, bot_id: str) -> dict[str, Any]:
        response = await self.client.get(f"bots/{bot_id}")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Attendee returned an invalid bot response")
        return payload

    async def get_transcript(self, bot_id: str) -> Any:
        response = await self.client.get(
            f"bots/{bot_id}/transcript",
            timeout=httpx.Timeout(300, connect=15, write=60, pool=15),
        )
        response.raise_for_status()
        return response.json()

    async def get_chat_messages(self, bot_id: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            params = {"cursor": cursor} if cursor else None
            response = await self.client.get(
                f"bots/{bot_id}/chat_messages", params=params
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                items = payload
                next_url = None
            elif isinstance(payload, dict):
                items = payload.get("results", [])
                next_url = payload.get("next")
            else:
                raise ValueError("Attendee returned an invalid chat message response")
            if not isinstance(items, list):
                raise ValueError("Attendee returned an invalid chat message list")

            for item in items:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                timestamp_ms = item.get("timestamp_ms")
                if (
                    not isinstance(timestamp_ms, int)
                    or isinstance(timestamp_ms, bool)
                    or timestamp_ms < 0
                ):
                    timestamp = item.get("timestamp")
                    timestamp_ms = (
                        timestamp * 1000
                        if isinstance(timestamp, int)
                        and not isinstance(timestamp, bool)
                        and timestamp >= 0
                        else None
                    )
                messages.append(
                    {
                        "id": item.get("id"),
                        "timestamp_ms": timestamp_ms,
                        "speaker_name": str(
                            item.get("sender_name") or "Unknown speaker"
                        ),
                        "text": text.strip(),
                        "source": "chat",
                    }
                )

            if not isinstance(next_url, str) or not next_url:
                break
            next_cursor = parse_qs(urlsplit(next_url).query).get("cursor", [None])[0]
            if not next_cursor or next_cursor in seen_cursors:
                break
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        return messages

    async def close(self) -> None:
        await self.client.aclose()


def verify_attendee_signature(payload: dict[str, Any], signature: str | None) -> bool:
    if not signature or not settings.attendee_webhook_secret:
        return False
    try:
        secret = base64.b64decode(settings.attendee_webhook_secret, validate=True)
    except ValueError, binascii.Error:
        return False
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    expected = base64.b64encode(
        hmac.new(secret, canonical, hashlib.sha256).digest()
    ).decode("ascii")
    return hmac.compare_digest(expected, signature)
