from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, HttpUrl


class MeetingStartRequest(BaseModel):
    meeting_url: HttpUrl


class TaskProposal(BaseModel):
    summary: str = Field(min_length=1, max_length=255)
    description: str = ""
    owner_speaker: str | None = None
    due_date: date | None = None
    priority: str | None = None
    evidence_quote: str = Field(min_length=1)
    timestamp_ms: int | None = Field(default=None, ge=0)
    confidence: float = Field(ge=0, le=1)


class TaskExtraction(BaseModel):
    tasks: list[TaskProposal] = Field(default_factory=list)
