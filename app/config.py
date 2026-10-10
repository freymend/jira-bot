from __future__ import annotations

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _json_mapping(name: str) -> dict[str, str]:
    raw = os.getenv(name, "{}").strip()
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} must contain a JSON object") from exc
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in value.items()
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
    llm_confidence_threshold: float
    jira_dry_run: bool
    jira_base_url: str
    jira_email: str
    jira_api_token: str
    jira_project_key: str
    jira_issue_type: str
    jira_speaker_account_ids: dict[str, str]
    jira_priority_map: dict[str, str]

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        threshold = float(os.getenv("LLM_CONFIDENCE_THRESHOLD", "0.85"))
        if not 0 <= threshold <= 1:
            raise RuntimeError("LLM_CONFIDENCE_THRESHOLD must be between 0 and 1")
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
            llm_base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8100/v1").rstrip("/"),
            llm_api_key=os.getenv("LLM_API_KEY", "local"),
            llm_model=os.getenv("LLM_MODEL", "mlx-community/gemma-4-e4b-it-4bit"),
            llm_confidence_threshold=threshold,
            jira_dry_run=_bool("JIRA_DRY_RUN", True),
            jira_base_url=os.getenv("JIRA_BASE_URL", "").rstrip("/"),
            jira_email=os.getenv("JIRA_EMAIL", ""),
            jira_api_token=os.getenv("JIRA_API_TOKEN", ""),
            jira_project_key=os.getenv("JIRA_PROJECT_KEY", ""),
            jira_issue_type=os.getenv("JIRA_ISSUE_TYPE", "Task"),
            jira_speaker_account_ids=_json_mapping("JIRA_SPEAKER_ACCOUNT_IDS"),
            jira_priority_map=_json_mapping("JIRA_PRIORITY_MAP"),
        )


settings = Settings.from_env()
