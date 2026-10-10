from __future__ import annotations

import hashlib
import hmac
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request

from app import db
from app.config import settings
from app.integrations import (
    AttendeeClient,
    JiraClient,
    extract_tasks,
    normalize_segments,
    verify_attendee_signature,
)
from app.schemas import MeetingStartRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jira-bot")
attendee = AttendeeClient()
jira = JiraClient()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    db.init_db()
    try:
        yield
    finally:
        await attendee.close()


app = FastAPI(title="Google Meet Jira Task Assistant", version="0.1.0", lifespan=lifespan)


async def require_app_auth(authorization: str | None = Header(default=None)) -> None:
    if not settings.app_api_token:
        raise HTTPException(status_code=503, detail="APP_API_TOKEN is not configured")
    expected = f"Bearer {settings.app_api_token}"
    if authorization is None or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Invalid bearer token")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/meetings", dependencies=[Depends(require_app_auth)])
async def start_meeting(body: MeetingStartRequest) -> dict[str, Any]:
    parsed = urlparse(str(body.meeting_url))
    if parsed.hostname not in {"meet.google.com", "www.meet.google.com"}:
        raise HTTPException(status_code=400, detail="Only Google Meet URLs are supported")
    try:
        bot = await attendee.create_bot(str(body.meeting_url))
    except Exception as exc:
        logger.exception("Could not create Attendee bot")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    bot_id = str(bot["id"])
    db.create_meeting(bot_id, str(body.meeting_url), bot.get("state"))
    return {"bot_id": bot_id, "state": bot.get("state"), "meeting_url": body.meeting_url}


@app.post("/meetings/{bot_id}/leave", dependencies=[Depends(require_app_auth)])
async def leave_meeting(bot_id: str) -> dict[str, Any]:
    try:
        return await attendee.leave_bot(bot_id)
    except Exception as exc:
        logger.exception("Could not ask Attendee bot to leave")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/meetings/{bot_id}", dependencies=[Depends(require_app_auth)])
async def meeting_status(bot_id: str) -> dict[str, Any]:
    meeting = db.get_meeting(bot_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting


@app.post("/meetings/{bot_id}/reprocess", dependencies=[Depends(require_app_auth)])
async def reprocess_meeting(bot_id: str, background_tasks: BackgroundTasks) -> dict[str, str]:
    if db.get_meeting(bot_id) is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    background_tasks.add_task(_finalize_meeting, bot_id)
    return {"bot_id": bot_id, "status": "queued"}


def _event_is_final(payload: dict[str, Any]) -> bool:
    if payload.get("trigger") != "bot.state_change":
        return False
    data = payload.get("data") or {}
    return isinstance(data, dict) and data.get("event_type") == "post_processing_completed"


@app.post("/webhooks/attendee")
async def attendee_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_webhook_signature: str | None = Header(default=None),
) -> dict[str, str]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc
    if not isinstance(payload, dict) or not verify_attendee_signature(payload, x_webhook_signature):
        raise HTTPException(status_code=401, detail="Invalid Attendee signature")
    trigger = str(payload.get("trigger") or "unknown")
    bot_id = str(payload.get("bot_id") or "")
    if not bot_id:
        raise HTTPException(status_code=400, detail="Webhook is missing bot_id")

    event_id = str(
        payload.get("idempotency_key")
        or hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
    )
    is_new_event = db.record_webhook(event_id, bot_id, trigger, payload)
    event_data = payload.get("data") or {}
    if trigger == "bot.state_change" and isinstance(event_data, dict):
        new_state = event_data.get("new_state")
        if new_state:
            db.update_bot_state(bot_id, str(new_state))
    if trigger == "transcript.update":
        segment = normalize_segments(event_data)
        db.save_segments(bot_id, segment)
    if is_new_event and _event_is_final(payload):
        background_tasks.add_task(_finalize_meeting, bot_id)
    return {"status": "received"}


async def _finalize_meeting(bot_id: str) -> None:
    if not db.claim_finalization(bot_id):
        return
    try:
        transcript_response = await attendee.get_transcript(bot_id)
        segments = normalize_segments(transcript_response)
        if not segments:
            raise RuntimeError("Attendee returned an empty transcript")
        db.save_final_transcript(bot_id, segments)
        meeting = db.get_meeting(bot_id) or {}
        proposals = await extract_tasks(segments)
        transcript_text = "\n".join(segment["text"] for segment in segments).casefold()
        approved = []
        for proposal in proposals.tasks:
            item = proposal.model_dump(mode="json")
            if proposal.confidence < settings.llm_confidence_threshold:
                continue
            if proposal.evidence_quote.casefold() not in transcript_text:
                logger.warning("Ignoring task without verifiable transcript evidence for %s", bot_id)
                continue
            approved.append(item)

        has_jira_error = False
        for item in approved:
            row, inserted = db.add_task(bot_id, item)
            if row["jira_key"]:
                continue
            if not settings.jira_dry_run and not inserted and row["status"] != "dry_run":
                # Avoid duplicate Jira issues after an ambiguous earlier failure.
                has_jira_error = True
                continue
            if settings.jira_dry_run:
                db.set_task_status(row["id"], "dry_run")
                continue
            db.set_task_status(row["id"], "creating")
            try:
                key = await jira.create_issue(item, meeting.get("meeting_url"))
                db.set_task_status(row["id"], "created", jira_key=key)
            except Exception as exc:
                has_jira_error = True
                db.set_task_status(row["id"], "uncertain", error=str(exc)[:1000])
                logger.exception("Jira issue creation failed for bot %s", bot_id)

        status = "error" if has_jira_error else ("dry_run" if settings.jira_dry_run else "completed")
        db.set_meeting_status(bot_id, status)
    except Exception as exc:
        logger.exception("Meeting finalization failed for bot %s", bot_id)
        db.set_meeting_status(bot_id, "error", str(exc)[:1000])
