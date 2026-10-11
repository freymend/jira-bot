from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.config import settings


class AttendeeClient:
    def __init__(self) -> None:
        self.client = httpx.AsyncClient(
            base_url=settings.attendee_api_base.rstrip("/") + "/",
            headers={"Authorization": f"Token {settings.attendee_api_key}"},
            timeout=httpx.Timeout(60, connect=15, write=30, pool=15),
        )

    async def create_bot(self, meeting_url: str) -> dict[str, Any]:
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
        response = await self.client.post(
            "bots",
            json={
                "meeting_url": meeting_url,
                "bot_name": settings.attendee_bot_name,
                "transcription_settings": transcription_settings,
                "webhooks": [
                    {
                        "url": f"{settings.app_public_url}/webhooks/attendee",
                        "triggers": ["bot.state_change", "transcript.update"],
                    }
                ],
            },
        )
        response.raise_for_status()
        return response.json()

    async def leave_bot(self, bot_id: str) -> dict[str, Any]:
        response = await self.client.post(f"bots/{bot_id}/leave")
        response.raise_for_status()
        return response.json() if response.content else {"ok": True}

    async def get_transcript(self, bot_id: str) -> Any:
        response = await self.client.get(
            f"bots/{bot_id}/transcript",
            timeout=httpx.Timeout(300, connect=15, write=60, pool=15),
        )
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        await self.client.aclose()


def verify_attendee_signature(payload: dict[str, Any], signature: str | None) -> bool:
    if not signature or not settings.attendee_webhook_secret:
        return False
    try:
        secret = base64.b64decode(settings.attendee_webhook_secret, validate=True)
    except (ValueError, binascii.Error):
        return False
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    expected = base64.b64encode(
        hmac.new(secret, canonical, hashlib.sha256).digest()
    ).decode("ascii")
    return hmac.compare_digest(expected, signature)
