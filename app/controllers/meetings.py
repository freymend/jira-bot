from __future__ import annotations

import logging
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from app import db
from app.config import settings
from app.controllers.dependencies import require_app_auth
from app.pipeline import attendee
from app.schemas import (
    JSONValue,
    MeetingRecord,
    MeetingStartRequest,
    MeetingStartResponse,
)

router = APIRouter()

logger = logging.getLogger("jira-bot")


@router.post("/meetings", dependencies=[Depends(require_app_auth)])
async def start_meeting(body: MeetingStartRequest) -> MeetingStartResponse:
    parsed = urlparse(str(body.meeting_url))
    if parsed.hostname not in {"meet.google.com", "www.meet.google.com"}:
        raise HTTPException(
            status_code=400, detail="Only Google Meet URLs are supported"
        )
    screen_share_capture = (
        settings.screenshare_capture_enabled
        if body.screen_share_capture is None
        else body.screen_share_capture
    )
    if screen_share_capture and not settings.screenshare_capture_enabled:
        raise HTTPException(
            status_code=400,
            detail="Screen-share capture is disabled by service configuration",
        )
    try:
        bot = await attendee.create_bot(
            str(body.meeting_url), screen_share_capture=screen_share_capture
        )
    except Exception as exc:
        logger.exception("Could not create Attendee bot")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    bot_id = str(bot["id"])
    db.create_meeting(
        bot_id,
        str(body.meeting_url),
        bot.get("state"),
        screenshare_capture_enabled=screen_share_capture,
        participant_tracking_enabled=True,
    )
    return {
        "bot_id": bot_id,
        "state": bot.get("state"),
        "meeting_url": str(body.meeting_url),
        "screenshare_capture_enabled": screen_share_capture,
    }


@router.post("/meetings/{bot_id}/leave", dependencies=[Depends(require_app_auth)])
async def leave_meeting(bot_id: str) -> JSONValue:
    claimed = db.claim_meeting_leave(bot_id, "manual", require_recording=False)
    if not claimed:
        meeting = db.get_meeting(bot_id)
        if meeting is not None and meeting["leave_status"] in {
            "requesting",
            "requested",
            "uncertain",
            "stuck",
            "processing",
            "completed",
            "failed",
        }:
            return {"ok": True, "status": meeting["leave_status"]}
    try:
        result = await attendee.leave_bot(bot_id)
    except Exception as exc:
        if claimed:
            db.finish_meeting_leave(bot_id, "uncertain")
        logger.exception("Could not ask Attendee bot to leave")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if claimed:
        db.finish_meeting_leave(bot_id, "requested")
    return result


@router.get("/meetings/{bot_id}", dependencies=[Depends(require_app_auth)])
async def meeting_status(bot_id: str) -> MeetingRecord:
    meeting = db.get_meeting(bot_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting


@router.get(
    "/meetings/{bot_id}/screen-frames/{frame_id}",
    dependencies=[Depends(require_app_auth)],
)
async def screen_share_frame(bot_id: str, frame_id: int) -> Response:
    frame = db.get_screen_frame(bot_id, frame_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="Screen-share frame not found")
    return Response(
        content=frame["frame_data"],
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@router.post("/meetings/{bot_id}/reprocess", dependencies=[Depends(require_app_auth)])
async def reprocess_meeting(bot_id: str) -> dict[str, str]:
    if db.get_meeting(bot_id) is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    status = db.enqueue_finalization(bot_id, allow_reprocess=True)
    if status == "completed":
        raise HTTPException(status_code=409, detail="Meeting is already completed")
    return {"bot_id": bot_id, "status": status}
