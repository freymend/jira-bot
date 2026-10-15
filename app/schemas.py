from __future__ import annotations

from datetime import date
from typing import Literal, NotRequired, Required, TypedDict

from pydantic import BaseModel, Field, HttpUrl


type JSONValue = (
    str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]
)
type JSONObject = dict[str, JSONValue]


class TranscriptSegment(TypedDict):
    id: JSONValue
    timestamp_ms: JSONValue
    duration_ms: JSONValue
    speaker_name: JSONValue
    speaker_uuid: JSONValue
    text: str
    raw: JSONObject
    source: NotRequired[Literal["transcript", "chat"]]


class ChatMessage(TypedDict):
    id: JSONValue
    timestamp_ms: int | None
    speaker_name: str
    text: str
    source: Literal["chat"]


class MeetingContextSegment(TypedDict):
    text: str
    timestamp_ms: JSONValue
    speaker_name: JSONValue
    source: Literal["transcript", "chat"]


class CalendarEventDateTime(TypedDict, total=False):
    dateTime: str
    date: str
    timeZone: str


class CalendarOrganizer(TypedDict, total=False):
    email: str


class CalendarConferenceEntryPoint(TypedDict, total=False):
    entryPointType: str
    uri: str


class CalendarConferenceData(TypedDict, total=False):
    entryPoints: list[CalendarConferenceEntryPoint]


class CalendarEvent(TypedDict, total=False):
    id: str
    status: str
    summary: str
    start: CalendarEventDateTime
    end: CalendarEventDateTime
    hangoutLink: str
    conferenceData: CalendarConferenceData
    recurringEventId: str
    originalStartTime: CalendarEventDateTime
    organizer: CalendarOrganizer


class CalendarWatchResponse(TypedDict, total=False):
    resourceId: JSONValue
    expiration: JSONValue


class AttendeeBotResponse(TypedDict, total=False):
    id: JSONValue
    state: JSONValue


class CreatedAttendeeBot(TypedDict):
    id: str
    state: NotRequired[str]


class AttendeeWebhookSettings(TypedDict):
    url: str
    triggers: list[str]


class AttendeeTranscriptionProvider(TypedDict, total=False):
    headers: dict[str, str]
    form_data: dict[str, str]


class AttendeeTranscriptionSettings(TypedDict):
    custom_async_v2: AttendeeTranscriptionProvider


class AttendeeVideoSettings(TypedDict):
    url: str
    webcam_resolution: str
    screenshare_resolution: str


class AttendeeWebsocketSettings(TypedDict):
    per_participant_video: AttendeeVideoSettings


class AttendeeBotSettings(TypedDict, total=False):
    meeting_url: str
    bot_name: str
    transcription_settings: AttendeeTranscriptionSettings
    webhooks: list[AttendeeWebhookSettings]
    websocket_settings: AttendeeWebsocketSettings


class CalendarSyncTokenRow(TypedDict):
    sync_token: str


class ExtractionCheckpointRow(TypedDict):
    result_json: str


class CalendarSyncUpsert(TypedDict):
    action: Literal["upsert"]
    event_key: str
    calendar_event_id: str
    event_start: str
    event_end: str | None
    meeting_url: str


class CalendarSyncDelete(TypedDict):
    action: Literal["delete"]
    event_key: str


type CalendarSyncChange = CalendarSyncUpsert | CalendarSyncDelete


class CalendarScheduledEvent(TypedDict):
    event_key: str
    calendar_event_id: str
    event_start: str
    event_end: str | None
    meeting_url: str


class CalendarEventRow(TypedDict):
    calendar_event_id: str
    event_start: str
    event_end: str | None
    meeting_url: str


class DashboardCalendarEvent(TypedDict):
    event_id: str
    event_title: str
    event_start: str
    event_end: str | None
    meeting_url: str
    join_status: str
    bot_id: str | None
    error: str | None


class CalendarAutoJoin(TypedDict):
    calendar_event_id: str
    event_start: str
    status: str
    bot_id: str | None
    error: str | None
    created_at: str
    updated_at: str


class CalendarWatchChannel(TypedDict):
    channel_id: str
    calendar_id: str
    channel_token: str
    resource_id: str | None
    expiration_ms: int
    status: str
    created_at: NotRequired[str]
    updated_at: NotRequired[str]


class MeetingLeaveCandidate(TypedDict):
    bot_id: str
    created_at: str
    scheduled_end_at: str | None
    last_transcript_received_at: str | None
    leave_evaluated_activity_at: str | None
    leave_evaluation_started_at: str | None
    participant_tracking_enabled: int
    participant_presence_initialized: int
    last_participant_event_at: str | None
    active_participant_count: int
    meeting_host_names: str | None


class MeetingLeaveRetry(TypedDict):
    bot_id: str
    leave_requested_at: str


class MeetingLeaveWatchdogCandidate(TypedDict):
    bot_id: str
    leave_requested_at: str


class MeetingRecordingOriginRow(TypedDict):
    recording_started_at_ms: int | None


class ParticipantEventTimestampRow(TypedDict):
    last_event_timestamp_ms: int | None


class MeetingLeaveWatchdogStateRow(TypedDict):
    leave_status: str
    bot_state: str | None


class RecentTranscriptSegment(TypedDict):
    timestamp_ms: int | None
    speaker_name: str | None
    text: str


class StatusRow(TypedDict):
    status: str


class BackgroundJobClaimCandidate(TypedDict):
    id: int
    bot_id: str
    status: str


class WorkerHeartbeatRow(TypedDict):
    latest: float | None


class HealthChecks(TypedDict):
    database: bool
    worker: bool


class BackgroundJob(TypedDict):
    id: int
    kind: str
    bot_id: str
    status: str
    attempts: int
    available_at: float
    lease_until: float | None
    locked_by: str | None
    last_error: str | None
    created_at: float
    updated_at: float


class RecentMeeting(TypedDict):
    bot_id: str
    meeting_url: str | None
    created_at: str
    bot_state: str | None
    transcription_state: str
    status: str
    error: str | None
    leave_status: str
    leave_stuck_at: str | None
    task_count: int


class DashboardService(TypedDict):
    name: str
    status: str
    detail: NotRequired[str]


class DashboardCalendar(TypedDict):
    enabled: bool
    event_title_filter: str | None
    start_time_filter: str
    organizer_email_filter: str | None
    time_zone: str
    source: str
    error: str | None
    upcoming_events: list[DashboardCalendarEvent]


class DashboardOverview(TypedDict):
    status: str
    updated_at: str
    services: list[DashboardService]
    recent_meetings: list[RecentMeeting]
    calendar: DashboardCalendar


class ConfigField(TypedDict):
    key: str
    label: str
    group: str
    type: str
    help: NotRequired[str]


class ConfigResponse(TypedDict):
    fields: list[ConfigField]
    values: dict[str, str]
    secret_configured: dict[str, bool]


class ConfigUpdateResponse(TypedDict):
    saved: bool
    restart_required: bool
    api_token_changed: bool


class CalendarAutoJoinStatus(TypedDict):
    enabled: bool
    mode: str
    event_title: str | None
    start_time: str
    time_zone: str
    recent_requests: list[CalendarAutoJoin]


class MeetingStartResponse(TypedDict):
    bot_id: str
    state: str | None
    meeting_url: str
    screenshare_capture_enabled: bool


class TaskPersistenceResult(TypedDict):
    id: int
    status: str
    jira_key: str | None


class MeetingTaskRecord(TypedDict):
    id: int
    bot_id: str
    fingerprint: str
    summary: str
    description: str
    owner_speaker: str | None
    owner_evidence_quote: str | None
    owner_timestamp_ms: int | None
    due_date: str | None
    priority: str | None
    evidence_quote: str
    timestamp_ms: int | None
    context_evidence: list[TaskContextEvidenceData]
    visual_evidence: list[TaskVisualEvidenceData]
    confidence: float
    status: str
    jira_key: str | None
    error: str | None


class MeetingRecord(TypedDict):
    bot_id: str
    meeting_url: str | None
    created_at: str
    transcription_state: str
    status: str
    error: str | None
    bot_state: str | None
    transcript: JSONValue
    recording_started_at_ms: int | None
    screen_frame_origin_ms: int | None
    screen_frame_origin_source: str | None
    screenshare_capture_enabled: int
    scheduled_end_at: str | None
    last_transcript_received_at: str | None
    leave_evaluated_activity_at: str | None
    leave_evaluation_started_at: str | None
    leave_status: str
    leave_requested_at: str | None
    leave_reason: str | None
    leave_watchdog_checked_at: str | None
    leave_stuck_at: str | None
    participant_tracking_enabled: int
    participant_presence_initialized: int
    last_participant_event_at: str | None
    leave_attempt_count: int
    tasks: list[MeetingTaskRecord]


class ScreenCaptureStateRow(TypedDict):
    recording_started_at_ms: int | None
    screen_frame_origin_ms: int | None
    screenshare_capture_enabled: int


class ScreenFrameReceivedAtRow(TypedDict):
    received_at_ms: int


class ScreenFrameCountRow(TypedDict):
    frame_count: int


class ScreenCaptureEnabledRow(TypedDict):
    screenshare_capture_enabled: int


class ScreenFrameRecord(TypedDict):
    id: int
    bot_id: str
    timestamp_ms: int
    time_source: str
    participant_uuid: str | None
    frame_data: bytes
    summary: str | None


class JiraScreenFrame(TypedDict):
    id: int
    timestamp_ms: int
    frame_data: bytes


class JiraTextNode(TypedDict):
    type: Literal["text"]
    text: str


class JiraParagraph(TypedDict):
    type: Literal["paragraph"]
    content: list[JiraTextNode]


class JiraHeading(TypedDict):
    type: Literal["heading"]
    attrs: dict[str, int]
    content: list[JiraTextNode]


type JiraBlock = JiraParagraph | JiraHeading


class JiraDocument(TypedDict):
    type: Literal["doc"]
    version: Literal[1]
    content: list[JiraBlock]


class JiraIssueFields(TypedDict):
    project: dict[str, str]
    summary: str
    issuetype: dict[str, str]
    description: JiraDocument
    assignee: NotRequired[dict[str, str]]
    priority: NotRequired[dict[str, str]]
    duedate: NotRequired[str]


class TaskContextEvidenceData(TypedDict):
    quote: str
    timestamp_ms: int | None
    speaker_name: str | None
    source: Literal["transcript", "chat"]


class TaskVisualEvidenceData(TypedDict):
    frame_id: int
    timestamp_ms: int
    time_source: str
    summary: str


class TaskPayload(TypedDict):
    summary: str
    evidence_quote: str
    description: NotRequired[str]
    owner_speaker: NotRequired[str | None]
    owner_evidence_quote: NotRequired[str | None]
    owner_timestamp_ms: NotRequired[int | None]
    due_date: NotRequired[str | None]
    priority: NotRequired[str | None]
    timestamp_ms: NotRequired[int | None]
    context_evidence: NotRequired[list[TaskContextEvidenceData]]
    visual_evidence: NotRequired[list[TaskVisualEvidenceData]]


class PersistedTaskPayload(TaskPayload):
    confidence: Required[float]


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
