from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class MeetingStartRequest(BaseModel):
    meeting_url: HttpUrl
    screen_share_capture: bool | None = None


class TaskContextEvidence(BaseModel):
    quote: str = Field(min_length=1)
    timestamp_ms: int | None = Field(default=None, ge=0)
    speaker_name: str | None = None
    source: Literal["transcript", "chat"] = "transcript"


class TaskVisualEvidence(BaseModel):
    frame_id: int = Field(gt=0)
    timestamp_ms: int = Field(ge=0)
    time_source: str
    summary: str = ""


class TaskProposal(BaseModel):
    summary: str = Field(min_length=1, max_length=255)
    description: str = ""
    owner_speaker: str | None = None
    owner_evidence_quote: str | None = None
    owner_timestamp_ms: int | None = Field(default=None, ge=0)
    due_date: date | None = None
    priority: str | None = None
    evidence_quote: str = Field(min_length=1)
    timestamp_ms: int | None = Field(default=None, ge=0)
    context_query: str = ""
    context_evidence: list[TaskContextEvidence] = Field(default_factory=list)
    visual_evidence: list[TaskVisualEvidence] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class TaskExtraction(BaseModel):
    tasks: list[TaskProposal] = Field(default_factory=list)


class TaskContextResult(BaseModel):
    description: str = Field(min_length=1)
    evidence_segment_ids: list[str] = Field(default_factory=list)
