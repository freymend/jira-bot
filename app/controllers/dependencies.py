from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from app.config import settings


async def require_app_auth(authorization: str | None = Header(default=None)) -> None:
    if not settings.app_api_token:
        raise HTTPException(status_code=503, detail="APP_API_TOKEN is not configured")
    expected = f"Bearer {settings.app_api_token}"
    if authorization is None or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Invalid bearer token")
