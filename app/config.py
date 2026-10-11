from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import time as datetime_time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv


def _json_mapping(name: str) -> dict[str, str]:
    raw = os.getenv(name, "{}").strip()
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} must contain a JSON object") from exc
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise RuntimeError(f"{name} must map strings to strings")
    return value


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_api_token: str
    app_public_url: str
    app_port: int
    database_path: str
    attendee_api_key: str
    attendee_webhook_secret: str
    attendee_api_base: str
    attendee_bot_name: str
    mlx_transcriber_api_key: str
    mlx_whisper_language: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_context_tokens: int
    llm_max_completion_tokens: int
    llm_confidence_threshold: float
    jira_dry_run: bool
    jira_base_url: str
    jira_email: str
    jira_api_token: str
    jira_project_key: str
    jira_issue_type: str
    jira_speaker_account_ids: dict[str, str]
    jira_priority_map: dict[str, str]
    google_calendar_auto_join_enabled: bool
    google_calendar_client_id: str
    google_calendar_client_secret: str
    google_calendar_refresh_token: str
    google_calendar_id: str
    google_calendar_event_title: str
    google_calendar_organizer_email: str
    google_calendar_timezone: str
    google_calendar_start_time: str
    google_calendar_poll_seconds: int
    google_calendar_join_lead_minutes: int
    google_calendar_join_grace_minutes: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        threshold = float(os.getenv("LLM_CONFIDENCE_THRESHOLD", "0.85"))
        if not 0 <= threshold <= 1:
            raise RuntimeError("LLM_CONFIDENCE_THRESHOLD must be between 0 and 1")
        llm_context_tokens = int(os.getenv("LLM_CONTEXT_TOKENS", "16384"))
        llm_max_completion_tokens = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "2048"))
        if llm_context_tokens < 8192:
            raise RuntimeError("LLM_CONTEXT_TOKENS must be at least 8192")
        minimum_prompt_room = 1024 + 256 + 4096 + 512
        if not 256 <= llm_max_completion_tokens <= (
            llm_context_tokens - minimum_prompt_room
        ):
            raise RuntimeError(
                "LLM_MAX_COMPLETION_TOKENS must be at least 256 and leave "
                "enough room for prompt content and safety margins"
            )
        calendar_auto_join_enabled = _bool("GOOGLE_CALENDAR_AUTO_JOIN_ENABLED", False)
        calendar_client_id = os.getenv("GOOGLE_CALENDAR_CLIENT_ID", "").strip()
        calendar_client_secret = os.getenv("GOOGLE_CALENDAR_CLIENT_SECRET", "").strip()
        calendar_refresh_token = os.getenv("GOOGLE_CALENDAR_REFRESH_TOKEN", "").strip()
        calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary").strip() or "primary"
        calendar_event_title = os.getenv("GOOGLE_CALENDAR_EVENT_TITLE", "").strip()
        calendar_organizer_email = (
            os.getenv("GOOGLE_CALENDAR_ORGANIZER_EMAIL", "").strip().casefold()
        )
        calendar_timezone = os.getenv(
            "GOOGLE_CALENDAR_TIME_ZONE", "America/Los_Angeles"
        ).strip()
        calendar_start_time = os.getenv("GOOGLE_CALENDAR_START_TIME", "10:00").strip()
        calendar_poll_seconds = int(os.getenv("GOOGLE_CALENDAR_POLL_SECONDS", "60"))
        calendar_join_lead_minutes = int(
            os.getenv("GOOGLE_CALENDAR_JOIN_LEAD_MINUTES", "5")
        )
        calendar_join_grace_minutes = int(
            os.getenv("GOOGLE_CALENDAR_JOIN_GRACE_MINUTES", "2")
        )
        try:
            datetime_time.fromisoformat(calendar_start_time)
            ZoneInfo(calendar_timezone)
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise RuntimeError(
                "Google Calendar start time or time zone is invalid"
            ) from exc
        if not 10 <= calendar_poll_seconds <= 3600:
            raise RuntimeError("GOOGLE_CALENDAR_POLL_SECONDS must be 10-3600")
        if not 1 <= calendar_join_lead_minutes <= 30:
            raise RuntimeError("GOOGLE_CALENDAR_JOIN_LEAD_MINUTES must be 1-30")
        if not 0 <= calendar_join_grace_minutes <= 15:
            raise RuntimeError("GOOGLE_CALENDAR_JOIN_GRACE_MINUTES must be 0-15")
        if calendar_auto_join_enabled:
            required_calendar_values = {
                "GOOGLE_CALENDAR_CLIENT_ID": calendar_client_id,
                "GOOGLE_CALENDAR_CLIENT_SECRET": calendar_client_secret,
                "GOOGLE_CALENDAR_REFRESH_TOKEN": calendar_refresh_token,
                "GOOGLE_CALENDAR_EVENT_TITLE": calendar_event_title,
            }
            missing_calendar_values = [
                name for name, value in required_calendar_values.items() if not value
            ]
            if missing_calendar_values:
                raise RuntimeError(
                    "Google Calendar auto-join is enabled but missing: "
                    + ", ".join(missing_calendar_values)
                )
        return cls(
            app_api_token=os.getenv("APP_API_TOKEN", ""),
            app_public_url=os.getenv("APP_PUBLIC_URL", "").rstrip("/"),
            app_port=int(os.getenv("APP_PORT", "5005")),
            database_path=os.getenv("DATABASE_PATH", "./data/jira_bot.sqlite3"),
            attendee_api_key=os.getenv("ATTENDEE_API_KEY", ""),
            attendee_webhook_secret=os.getenv("ATTENDEE_WEBHOOK_SECRET", ""),
            attendee_api_base=os.getenv(
                "ATTENDEE_API_BASE", "http://127.0.0.1:8000/api/v1"
            ).rstrip("/"),
            attendee_bot_name=os.getenv("ATTENDEE_BOT_NAME", "Jira Meeting Assistant"),
            mlx_transcriber_api_key=os.getenv("MLX_TRANSCRIBER_API_KEY", ""),
            mlx_whisper_language=os.getenv("MLX_WHISPER_LANGUAGE", "").strip(),
            llm_base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8100/v1").rstrip(
                "/"
            ),
            llm_api_key=os.getenv("LLM_API_KEY", "local"),
            llm_model=os.getenv("LLM_MODEL", "mlx-community/gemma-4-e4b-it-4bit"),
            llm_context_tokens=llm_context_tokens,
            llm_max_completion_tokens=llm_max_completion_tokens,
            llm_confidence_threshold=threshold,
            jira_dry_run=_bool("JIRA_DRY_RUN", True),
            jira_base_url=os.getenv("JIRA_BASE_URL", "").rstrip("/"),
            jira_email=os.getenv("JIRA_EMAIL", ""),
            jira_api_token=os.getenv("JIRA_API_TOKEN", ""),
            jira_project_key=os.getenv("JIRA_PROJECT_KEY", ""),
            jira_issue_type=os.getenv("JIRA_ISSUE_TYPE", "Task"),
            jira_speaker_account_ids=_json_mapping("JIRA_SPEAKER_ACCOUNT_IDS"),
            jira_priority_map=_json_mapping("JIRA_PRIORITY_MAP"),
            google_calendar_auto_join_enabled=calendar_auto_join_enabled,
            google_calendar_client_id=calendar_client_id,
            google_calendar_client_secret=calendar_client_secret,
            google_calendar_refresh_token=calendar_refresh_token,
            google_calendar_id=calendar_id,
            google_calendar_event_title=calendar_event_title,
            google_calendar_organizer_email=calendar_organizer_email,
            google_calendar_timezone=calendar_timezone,
            google_calendar_start_time=calendar_start_time,
            google_calendar_poll_seconds=calendar_poll_seconds,
            google_calendar_join_lead_minutes=calendar_join_lead_minutes,
            google_calendar_join_grace_minutes=calendar_join_grace_minutes,
        )


settings = Settings.from_env()
