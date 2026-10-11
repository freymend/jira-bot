# Development plan

## Current baseline

The service joins a manually submitted Google Meet, transcribes audio locally, extracts action items, and can create Jira issues. Issue creation remains opt-in through `JIRA_DRY_RUN`. The API exposes meeting and pipeline state, but there is no operator dashboard or attendee-facing task feedback yet.

## Roadmap

### 1. Add screen-share context to task creation

Use relevant shared-screen content to help interpret action items and identify the right Jira task details.

- Investigate what screen capture the meeting integration can provide, including its permissions, timing, and retention behavior.
- Link any captured frame to the corresponding transcript time so visual context can be reviewed alongside what was said.
- Define how visual evidence informs task proposals, and make its source visible to reviewers.
- Add safeguards for sensitive on-screen content, including clear capture controls and a retention policy.
- Keep audio transcription as the source of spoken commitments; screen content provides context and should not invent an owner or commitment.

**Done when:** task proposals can include relevant, time-linked screen context; reviewers can inspect how it informed a proposal; and capture and retention behavior are documented and configurable.

### 2. Build a central service dashboard

Provide one place to see whether the pipeline is working and to diagnose failures without piecing together container status, logs, and individual API responses.

- Show health and status for the API, Attendee, its worker and scheduler, Gemma, and the local Whisper transcriber.
- Show recent meetings and their pipeline stage, webhook activity, task proposals, Jira outcome, and actionable errors.
- Link status details to relevant service logs and meeting records.
- Let authorized users configure integration and model-provider credentials through a secure settings flow; never reveal saved secrets in dashboard views or logs.
- Let users switch supported transcription and task-extraction models, configure their endpoints, and validate the selected setup.
- Restrict access to configuration, transcripts, and meeting data, and audit sensitive configuration changes.

**Done when:** an operator can identify which service is unhealthy, find the affected meeting or task, diagnose common failures, securely update credentials, and switch between supported models from the dashboard.

### 3. Post created-task links to meeting attendees

After Jira successfully creates an issue, have the bot post a concise meeting message that links to it.

- Confirm the meeting integration supports posting messages and determine the required permissions.
- Include the Jira issue key, title, and direct link.
- Post only after successful issue creation; do not announce dry-run proposals or failed/uncertain requests as created issues.
- Make posting idempotent so webhook retries do not send duplicate messages, and record delivery status for debugging.

**Done when:** each successfully created issue can produce at most one corresponding meeting message, with its delivery outcome visible in the meeting record.

### 4. Support asynchronous updates from people who cannot attend

Let someone who is not part of a meeting receive relevant outcomes and contribute information or status updates later, without joining the live call.

- Choose an asynchronous channel for sending meeting outcomes and task context to authorized people; the channel remains an implementation decision.
- Let recipients reply later with clarifications, new information, or task status updates, and connect each reply to the right meeting and task.
- Share relevant replies with the meeting participants and reflect accepted status changes in Jira or the local task record, according to a defined policy.
- Resolve ambiguous task references or conflicting updates before changing status.
- Verify the sender's identity and permissions, make processing idempotent, and retain an audit trail of messages and resulting changes.

**Done when:** an authorized person who missed the meeting can receive its relevant context, reply asynchronously with information or a status update, and have that contribution delivered to the right participants and recorded against the right task.

### Far-future exploration: Claire Dev harness

Explore whether this service should integrate with Claire Dev, an internal development harness actively being developed by Shenxing. Treat this as a research item until the harness has a documented, stable integration surface.

- Identify the use cases where Claire Dev would add value to meeting follow-up or Jira task workflows.
- Review its available interfaces, authentication model, data boundaries, and failure behavior before choosing an integration approach.
- If there is a clear benefit, prototype with non-production tasks and keep the integration optional and isolated from the core meeting pipeline.

## Ongoing constraints

- Keep Jira issue creation in dry-run mode by default until the end-to-end workflow has been reviewed.
- Currently, assign tasks only to Zachary or Zac when the transcript explicitly names that person as accepting; make eligible people and their Jira identities configurable in a future iteration.
- Preserve local processing for audio transcription and the language model unless a later decision changes that requirement.
- Do not require an off-call contributor to join a meeting in order to receive context or provide an update.
- Treat meeting transcripts, recordings, screen content, and asynchronous messages as sensitive data; define storage, access, and deletion behavior for each new data type.
