from __future__ import annotations

import logging

from app import db
from app.config import settings
from app.integrations.attendee import AttendeeClient
from app.integrations.jira import JiraClient
from app.integrations.screenshare import add_screenshare_context
from app.integrations.task_extraction import extract_tasks
from app.integrations.transcript import normalize_segments
from app.network import is_ambiguous_write_error, is_safe_to_retry_write

logger = logging.getLogger("jira-bot.pipeline")
attendee = AttendeeClient()
jira = JiraClient()


async def finalize_meeting(bot_id: str) -> None:
    db.set_meeting_status(bot_id, "processing")
    try:
        meeting = db.get_meeting(bot_id) or {}
        segments = normalize_segments(meeting.get("transcript"))
        if not segments:
            transcript_response = await attendee.get_transcript(bot_id)
            segments = normalize_segments(transcript_response)
        if not segments:
            raise RuntimeError("Attendee returned an empty transcript")
        db.save_final_transcript(bot_id, segments)
        meeting = db.get_meeting(bot_id) or meeting
        try:
            chat_messages = await attendee.get_chat_messages(bot_id)
        except Exception:
            logger.warning(
                "Could not fetch meeting chat messages for bot %s",
                bot_id,
                exc_info=True,
            )
            chat_messages = []
        proposals = await extract_tasks(
            segments, bot_id=bot_id, chat_messages=chat_messages
        )
        transcript_text = "\n".join(segment["text"] for segment in segments).casefold()
        eligible = []
        for proposal in proposals.tasks:
            if proposal.confidence < settings.llm_confidence_threshold:
                continue
            if proposal.evidence_quote.casefold() not in transcript_text:
                logger.warning(
                    "Ignoring task without verifiable transcript evidence for %s",
                    bot_id,
                )
                continue
            eligible.append(proposal)
        eligible = await add_screenshare_context(eligible, bot_id)
        approved = [
            proposal.model_dump(mode="json", exclude={"context_query"})
            for proposal in eligible
        ]

        has_jira_error = False
        for item in approved:
            row, inserted = db.add_task(bot_id, item)
            if row["jira_key"]:
                continue
            if (
                not settings.jira_dry_run
                and not inserted
                and row["status"] in {"creating", "uncertain", "created"}
            ):
                # An interrupted Jira request may have succeeded; avoid duplicates.
                if row["status"] == "creating":
                    db.set_task_status(
                        row["id"],
                        "uncertain",
                        error="Issue creation was interrupted; verify Jira before retrying.",
                    )
                has_jira_error = True
                continue
            if settings.jira_dry_run:
                db.set_task_status(row["id"], "dry_run")
                continue

            db.set_task_status(row["id"], "creating")
            try:
                screen_frames = []
                if settings.jira_attach_screenshare_evidence:
                    screen_frames = [
                        frame
                        for evidence in item.get("visual_evidence", [])
                        if (frame := db.get_screen_frame(bot_id, evidence["frame_id"]))
                    ]
                key = await jira.create_issue(
                    item,
                    meeting.get("meeting_url"),
                    screen_frames=screen_frames,
                )
                db.set_task_status(row["id"], "created", jira_key=key)
            except Exception as exc:
                if is_safe_to_retry_write(exc):
                    db.set_task_status(row["id"], "pending", error=str(exc)[:1000])
                    raise
                has_jira_error = True
                status = "uncertain" if is_ambiguous_write_error(exc) else "error"
                db.set_task_status(row["id"], status, error=str(exc)[:1000])
                logger.exception("Jira issue creation failed for bot %s", bot_id)

        status = (
            "error"
            if has_jira_error
            else ("dry_run" if settings.jira_dry_run else "completed")
        )
        db.set_meeting_status(bot_id, status)
    except Exception as exc:
        db.set_meeting_status(bot_id, "error", str(exc)[:1000])
        raise
