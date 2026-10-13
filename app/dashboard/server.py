from __future__ import annotations

import hmac
import ipaddress
import json
import os
import shutil
import tempfile
import threading
from datetime import time as datetime_time
from pathlib import Path

from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from dotenv import dotenv_values, set_key, unset_key
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.schemas import (
    ConfigField,
    ConfigResponse,
    ConfigUpdateResponse,
    JSONValue,
    JSONObject,
)

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"
STATIC_DIR = Path(__file__).resolve().parent / "static"
_CONFIG_LOCK = threading.Lock()

CONFIG_FIELDS: list[ConfigField] = [
    {
        "key": "APP_API_TOKEN",
        "label": "API bearer token",
        "group": "Application",
        "type": "secret",
        "help": "Required to access management and dashboard APIs.",
    },
    {
        "key": "APP_PUBLIC_URL",
        "label": "Public callback URL",
        "group": "Application",
        "type": "url",
        "help": "Use a public HTTPS URL when external webhooks or screen-share streaming are enabled.",
    },
    {
        "key": "APP_PORT",
        "label": "API host port",
        "group": "Application",
        "type": "integer",
        "help": "Restart the local stack after changing this port.",
    },
    {
        "key": "DATABASE_PATH",
        "label": "SQLite database path",
        "group": "Application",
        "type": "text",
    },
    {
        "key": "LOCAL_DASHBOARD_PORT",
        "label": "Dashboard port",
        "group": "Application",
        "type": "integer",
        "help": "Restart the local stack after changing this port.",
    },
    {
        "key": "ATTENDEE_API_KEY",
        "label": "Attendee API key",
        "group": "Attendee",
        "type": "secret",
    },
    {
        "key": "ATTENDEE_WEBHOOK_SECRET",
        "label": "Attendee webhook secret",
        "group": "Attendee",
        "type": "secret",
    },
    {
        "key": "ATTENDEE_API_BASE",
        "label": "Attendee API base URL",
        "group": "Attendee",
        "type": "url",
    },
    {
        "key": "ATTENDEE_BOT_NAME",
        "label": "Meeting bot name",
        "group": "Attendee",
        "type": "text",
    },
    {
        "key": "MLX_TRANSCRIBER_API_KEY",
        "label": "Whisper API key",
        "group": "Transcription",
        "type": "secret",
    },
    {
        "key": "MLX_TRANSCRIBER_HEALTH_URL",
        "label": "Whisper health URL",
        "group": "Transcription",
        "type": "url",
    },
    {
        "key": "MLX_WHISPER_MODEL",
        "label": "Whisper model",
        "group": "Transcription",
        "type": "text",
    },
    {
        "key": "MLX_WHISPER_LANGUAGE",
        "label": "Whisper language",
        "group": "Transcription",
        "type": "text",
        "help": "Leave empty for automatic language detection.",
    },
    {
        "key": "LLM_BASE_URL",
        "label": "Local model URL",
        "group": "Task model",
        "type": "url",
    },
    {
        "key": "LLM_BASE_URL_DOCKER",
        "label": "Model URL from Docker",
        "group": "Task model",
        "type": "url",
    },
    {
        "key": "LLM_API_KEY",
        "label": "Model API key",
        "group": "Task model",
        "type": "secret",
    },
    {"key": "LLM_MODEL", "label": "Model name", "group": "Task model", "type": "text"},
    {
        "key": "LLM_CONTEXT_TOKENS",
        "label": "Model context tokens",
        "group": "Task model",
        "type": "integer",
    },
    {
        "key": "LLM_MAX_COMPLETION_TOKENS",
        "label": "Maximum completion tokens",
        "group": "Task model",
        "type": "integer",
    },
    {
        "key": "LLM_CONFIDENCE_THRESHOLD",
        "label": "Task confidence threshold",
        "group": "Task model",
        "type": "number",
    },
    {
        "key": "JIRA_DRY_RUN",
        "label": "Jira dry-run mode",
        "group": "Jira",
        "type": "boolean",
        "help": "Keep enabled until task proposals are reviewed.",
    },
    {"key": "JIRA_BASE_URL", "label": "Jira base URL", "group": "Jira", "type": "url"},
    {
        "key": "JIRA_EMAIL",
        "label": "Jira account email",
        "group": "Jira",
        "type": "text",
    },
    {
        "key": "JIRA_API_TOKEN",
        "label": "Jira API token",
        "group": "Jira",
        "type": "secret",
    },
    {
        "key": "JIRA_PROJECT_KEY",
        "label": "Jira project key",
        "group": "Jira",
        "type": "text",
    },
    {
        "key": "JIRA_ISSUE_TYPE",
        "label": "Jira issue type",
        "group": "Jira",
        "type": "text",
    },
    {
        "key": "JIRA_SPEAKER_ACCOUNT_IDS",
        "label": "Speaker account IDs (JSON)",
        "group": "Jira",
        "type": "json",
    },
    {
        "key": "JIRA_PRIORITY_MAP",
        "label": "Priority mapping (JSON)",
        "group": "Jira",
        "type": "json",
    },
    {
        "key": "SCREENSHARE_CAPTURE_ENABLED",
        "label": "Capture screen shares",
        "group": "Screen share",
        "type": "boolean",
    },
    {
        "key": "SCREENSHARE_WEBSOCKET_TOKEN",
        "label": "Screen-share websocket token",
        "group": "Screen share",
        "type": "secret",
    },
    {
        "key": "SCREENSHARE_RESOLUTION",
        "label": "Capture resolution",
        "group": "Screen share",
        "type": "text",
    },
    {
        "key": "SCREENSHARE_SAMPLE_INTERVAL_SECONDS",
        "label": "Frame sample interval (seconds)",
        "group": "Screen share",
        "type": "integer",
    },
    {
        "key": "SCREENSHARE_RETENTION_HOURS",
        "label": "Frame retention (hours)",
        "group": "Screen share",
        "type": "integer",
    },
    {
        "key": "SCREENSHARE_MAX_FRAMES_PER_MEETING",
        "label": "Maximum frames per meeting",
        "group": "Screen share",
        "type": "integer",
    },
    {
        "key": "SCREENSHARE_CONTEXT_WINDOW_SECONDS",
        "label": "Context window (seconds)",
        "group": "Screen share",
        "type": "integer",
    },
    {
        "key": "SCREENSHARE_VISION_ENABLED",
        "label": "Summarize frames with vision",
        "group": "Screen share",
        "type": "boolean",
    },
    {
        "key": "JIRA_ATTACH_SCREENSHARE_EVIDENCE",
        "label": "Attach source frames to Jira",
        "group": "Screen share",
        "type": "boolean",
    },
    {
        "key": "GOOGLE_CALENDAR_AUTO_JOIN_ENABLED",
        "label": "Enable Calendar auto-join",
        "group": "Google Calendar",
        "type": "boolean",
    },
    {
        "key": "GOOGLE_CALENDAR_CLIENT_ID",
        "label": "OAuth client ID",
        "group": "Google Calendar",
        "type": "text",
    },
    {
        "key": "GOOGLE_CALENDAR_CLIENT_SECRET",
        "label": "OAuth client secret",
        "group": "Google Calendar",
        "type": "secret",
    },
    {
        "key": "GOOGLE_CALENDAR_REFRESH_TOKEN",
        "label": "OAuth refresh token",
        "group": "Google Calendar",
        "type": "secret",
    },
    {
        "key": "GOOGLE_CALENDAR_ID",
        "label": "Calendar ID",
        "group": "Google Calendar",
        "type": "text",
    },
    {
        "key": "GOOGLE_CALENDAR_EVENT_TITLE",
        "label": "Auto-join event title",
        "group": "Google Calendar",
        "type": "text",
    },
    {
        "key": "GOOGLE_CALENDAR_ORGANIZER_EMAIL",
        "label": "Organizer email filter",
        "group": "Google Calendar",
        "type": "text",
    },
    {
        "key": "GOOGLE_CALENDAR_TIME_ZONE",
        "label": "Calendar time zone",
        "group": "Google Calendar",
        "type": "text",
    },
    {
        "key": "GOOGLE_CALENDAR_START_TIME",
        "label": "Daily meeting start time",
        "group": "Google Calendar",
        "type": "text",
    },
    {
        "key": "GOOGLE_CALENDAR_POLL_SECONDS",
        "label": "Calendar poll interval (seconds)",
        "group": "Google Calendar",
        "type": "integer",
    },
    {
        "key": "GOOGLE_CALENDAR_JOIN_LEAD_MINUTES",
        "label": "Join lead time (minutes)",
        "group": "Google Calendar",
        "type": "integer",
    },
    {
        "key": "GOOGLE_CALENDAR_JOIN_GRACE_MINUTES",
        "label": "Late join grace period (minutes)",
        "group": "Google Calendar",
        "type": "integer",
    },
]
_FIELD_BY_KEY = {field["key"]: field for field in CONFIG_FIELDS}
_CLEARABLE_SECRETS = {
    key
    for key, field in _FIELD_BY_KEY.items()
    if field["type"] == "secret" and key != "APP_API_TOKEN"
}


class ConfigUpdate(BaseModel):
    values: dict[str, JSONValue] = Field(default_factory=dict)
    clear: list[str] = Field(default_factory=list)


def _env_values() -> dict[str, str]:
    if ENV_FILE.is_symlink():
        raise HTTPException(
            status_code=500, detail="The .env file must not be a symlink"
        )
    if not ENV_FILE.is_file():
        return {}
    return {
        key: value
        for key, value in dotenv_values(ENV_FILE).items()
        if isinstance(key, str) and isinstance(value, str)
    }


def _dashboard_token() -> str:
    return _env_values().get("APP_API_TOKEN", "").strip()


def require_dashboard_auth(
    request: Request, authorization: str | None = Header(default=None)
) -> str:
    client_host = request.client.host if request.client is not None else ""
    try:
        is_local = ipaddress.ip_address(client_host).is_loopback
    except ValueError:
        is_local = False
    if not is_local:
        raise HTTPException(status_code=403, detail="Dashboard is local-only")

    token = _dashboard_token()
    if not token:
        raise HTTPException(status_code=503, detail="APP_API_TOKEN is not configured")
    expected = f"Bearer {token}"
    if authorization is None or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    return token


def _current_config_response() -> ConfigResponse:
    values = _env_values()
    defaults = {
        key: value
        for key, value in dotenv_values(ROOT / ".env.example").items()
        if isinstance(key, str) and isinstance(value, str)
    }
    return {
        "fields": CONFIG_FIELDS,
        "values": {
            key: values.get(key, defaults.get(key, ""))
            for key, field in _FIELD_BY_KEY.items()
            if field["type"] != "secret"
        },
        "secret_configured": {
            key: bool(values.get(key))
            for key, field in _FIELD_BY_KEY.items()
            if field["type"] == "secret"
        },
    }


def _normalize_value(key: str, raw: JSONValue) -> str:
    field = _FIELD_BY_KEY[key]
    kind = field["type"]
    if kind == "boolean":
        if isinstance(raw, bool):
            return "true" if raw else "false"
        value = str(raw).strip().lower()
        if value not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
            raise ValueError(f"{key} must be a boolean")
        return "true" if value in {"true", "1", "yes", "on"} else "false"
    if kind == "integer":
        if not isinstance(raw, str | int | float):
            raise ValueError(f"{key} must be an integer")
        try:
            return str(int(raw))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{key} must be an integer") from exc
    if kind == "number":
        if not isinstance(raw, str | int | float):
            raise ValueError(f"{key} must be a number")
        try:
            value = float(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{key} must be a number") from exc
        return str(value)
    if kind == "json":
        try:
            value = json.loads(str(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{key} must contain valid JSON") from exc
        if not isinstance(value, dict) or not all(
            isinstance(name, str) and isinstance(item, str)
            for name, item in value.items()
        ):
            raise ValueError(f"{key} must map strings to strings")
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if not isinstance(raw, (str, int, float)):
        raise ValueError(f"{key} must be text")
    value = str(raw)
    if len(value) > 8192 or "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(f"{key} contains invalid or oversized text")
    if kind == "url" and value:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"{key} must be an HTTP or HTTPS URL")
    return value


def _candidate_bool(values: dict[str, str], key: str, default: bool = False) -> bool:
    value = values.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _candidate_int(values: dict[str, str], key: str, default: int) -> int:
    try:
        return int(values.get(key, str(default)))
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc


def _validate_candidate(values: dict[str, str]) -> None:
    context_tokens = _candidate_int(values, "LLM_CONTEXT_TOKENS", 16384)
    completion_tokens = _candidate_int(values, "LLM_MAX_COMPLETION_TOKENS", 2048)
    if context_tokens < 8192:
        raise ValueError("LLM_CONTEXT_TOKENS must be at least 8192")
    minimum_prompt_room = 1024 + 256 + 4096 + 512
    if not 256 <= completion_tokens <= context_tokens - minimum_prompt_room:
        raise ValueError(
            "LLM_MAX_COMPLETION_TOKENS must leave enough room for prompt content"
        )
    try:
        confidence = float(values.get("LLM_CONFIDENCE_THRESHOLD", "0.85"))
    except ValueError as exc:
        raise ValueError("LLM_CONFIDENCE_THRESHOLD must be a number") from exc
    if not 0 <= confidence <= 1:
        raise ValueError("LLM_CONFIDENCE_THRESHOLD must be between 0 and 1")

    resolution = values.get("SCREENSHARE_RESOLUTION", "360p")
    if resolution not in {"360p", "720p", "1080p"}:
        raise ValueError("SCREENSHARE_RESOLUTION must be 360p, 720p, or 1080p")
    interval = _candidate_int(values, "SCREENSHARE_SAMPLE_INTERVAL_SECONDS", 10)
    retention = _candidate_int(values, "SCREENSHARE_RETENTION_HOURS", 24)
    max_frames = _candidate_int(values, "SCREENSHARE_MAX_FRAMES_PER_MEETING", 600)
    context_window = _candidate_int(values, "SCREENSHARE_CONTEXT_WINDOW_SECONDS", 30)
    if not 1 <= interval <= 60:
        raise ValueError("SCREENSHARE_SAMPLE_INTERVAL_SECONDS must be 1-60")
    if not 1 <= retention <= 8760:
        raise ValueError("SCREENSHARE_RETENTION_HOURS must be 1-8760")
    if not 1 <= max_frames <= 10000:
        raise ValueError("SCREENSHARE_MAX_FRAMES_PER_MEETING must be 1-10000")
    if not 0 <= context_window <= 300:
        raise ValueError("SCREENSHARE_CONTEXT_WINDOW_SECONDS must be 0-300")
    if (
        _candidate_bool(values, "SCREENSHARE_CAPTURE_ENABLED")
        and len(values.get("SCREENSHARE_WEBSOCKET_TOKEN", "")) < 32
    ):
        raise ValueError("SCREENSHARE_WEBSOCKET_TOKEN must be at least 32 characters")

    start_time = values.get("GOOGLE_CALENDAR_START_TIME", "10:00")
    timezone = values.get("GOOGLE_CALENDAR_TIME_ZONE", "America/Los_Angeles")
    try:
        datetime_time.fromisoformat(start_time)
        ZoneInfo(timezone)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError("Google Calendar start time or time zone is invalid") from exc
    poll_seconds = _candidate_int(values, "GOOGLE_CALENDAR_POLL_SECONDS", 60)
    lead_minutes = _candidate_int(values, "GOOGLE_CALENDAR_JOIN_LEAD_MINUTES", 5)
    grace_minutes = _candidate_int(values, "GOOGLE_CALENDAR_JOIN_GRACE_MINUTES", 2)
    if not 10 <= poll_seconds <= 3600:
        raise ValueError("GOOGLE_CALENDAR_POLL_SECONDS must be 10-3600")
    if not 1 <= lead_minutes <= 30:
        raise ValueError("GOOGLE_CALENDAR_JOIN_LEAD_MINUTES must be 1-30")
    if not 0 <= grace_minutes <= 15:
        raise ValueError("GOOGLE_CALENDAR_JOIN_GRACE_MINUTES must be 0-15")
    if _candidate_bool(values, "GOOGLE_CALENDAR_AUTO_JOIN_ENABLED"):
        required = (
            "GOOGLE_CALENDAR_CLIENT_ID",
            "GOOGLE_CALENDAR_CLIENT_SECRET",
            "GOOGLE_CALENDAR_REFRESH_TOKEN",
            "GOOGLE_CALENDAR_EVENT_TITLE",
        )
        missing = [key for key in required if not values.get(key)]
        if missing:
            raise ValueError(
                "Google Calendar auto-join is missing: " + ", ".join(missing)
            )

    for key in ("APP_PORT", "LOCAL_DASHBOARD_PORT"):
        if key in values:
            port = _candidate_int(values, key, 5005 if key == "APP_PORT" else 5006)
            if not 1 <= port <= 65535:
                raise ValueError(f"{key} must be between 1 and 65535")


def _write_env_file(updates: dict[str, str], clear: list[str]) -> None:
    if ENV_FILE.is_symlink():
        raise ValueError("The .env file must not be a symlink")
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".env-dashboard-", dir=ENV_FILE.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if ENV_FILE.is_file():
            shutil.copyfile(ENV_FILE, temporary)
        for key, value in updates.items():
            set_key(temporary, key, value, quote_mode="always")
        for key in clear:
            unset_key(temporary, key)
        os.chmod(temporary, 0o600)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, ENV_FILE)
        try:
            directory_fd = os.open(ENV_FILE.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


app = FastAPI(title="Local Meeting Agent Dashboard")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="dashboard-static")


@app.get("/", include_in_schema=False)
async def dashboard_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/health/live", include_in_schema=False)
async def dashboard_liveness() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
async def read_config(
    _: str = Depends(require_dashboard_auth),
) -> ConfigResponse:
    return _current_config_response()


@app.put("/api/config")
async def update_config(
    update: ConfigUpdate,
    token: str = Depends(require_dashboard_auth),
) -> ConfigUpdateResponse:
    del token
    unknown = (set(update.values) | set(update.clear)) - set(_FIELD_BY_KEY)
    if unknown:
        raise HTTPException(
            status_code=400,
            detail="Unsupported configuration keys: " + ", ".join(sorted(unknown)),
        )
    if not set(update.clear) <= _CLEARABLE_SECRETS:
        raise HTTPException(
            status_code=400, detail="Only optional secrets can be cleared"
        )
    if set(update.clear) & set(update.values):
        raise HTTPException(
            status_code=400,
            detail="A secret cannot be replaced and cleared in one update",
        )

    normalized: dict[str, str] = {}
    try:
        for key, raw in update.values.items():
            if _FIELD_BY_KEY[key]["type"] == "secret" and raw == "":
                continue
            normalized[key] = _normalize_value(key, raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with _CONFIG_LOCK:
        candidate = _env_values()
        for key in update.clear:
            candidate.pop(key, None)
        candidate.update(normalized)
        try:
            _validate_candidate(candidate)
            _write_env_file(normalized, update.clear)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Configuration was not saved: {exc}",
            ) from exc
    return {
        "saved": True,
        "restart_required": True,
        "api_token_changed": "APP_API_TOKEN" in normalized,
    }


@app.get("/api/overview")
async def read_overview(token: str = Depends(require_dashboard_auth)) -> JSONObject:
    values = _env_values()
    try:
        api_port = int(values.get("APP_PORT", "5005"))
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="APP_PORT is invalid") from exc
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8, connect=2)) as client:
            response = await client.get(
                f"http://127.0.0.1:{api_port}/dashboard/api/overview",
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail="The API dashboard status endpoint is unavailable; start or restart the local stack.",
        ) from exc
    payload = response.json()
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=502, detail="The API returned invalid dashboard data"
        )
    return payload
