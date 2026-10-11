from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from math import isfinite

import httpx
from openai import APIConnectionError

_RETRYABLE_STATUS_CODES = {408, 425, 429}


def is_transient_network_error(error: Exception) -> bool:
    if isinstance(
        error,
        (
            httpx.TransportError,
            APIConnectionError,
            TimeoutError,
            ConnectionError,
        ),
    ):
        return True

    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None) or getattr(
        error, "status_code", None
    )
    return isinstance(status_code, int) and (
        status_code in _RETRYABLE_STATUS_CODES or status_code >= 500
    )


def is_safe_to_retry_write(error: Exception) -> bool:
    if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout)):
        return True
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None) or getattr(
        error, "status_code", None
    )
    return status_code == 429


def is_ambiguous_write_error(error: Exception) -> bool:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None) or getattr(
        error, "status_code", None
    )
    if not isinstance(status_code, int):
        return True
    return status_code in {408, 425} or status_code >= 500


def retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    value = headers.get("Retry-After") if headers is not None else None
    if not value:
        return None

    try:
        seconds = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        seconds = (retry_at - datetime.now(UTC)).total_seconds()
    return max(0.0, seconds) if isfinite(seconds) else None
