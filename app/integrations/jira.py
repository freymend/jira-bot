from __future__ import annotations

from typing import Any

import httpx

from app.config import settings
from app.integrations.ownership import validated_owner_speaker


def _paragraph(text: str) -> dict[str, Any]:
    return {"type": "paragraph", "content": [{"type": "text", "text": text}]}


def _heading(text: str) -> dict[str, Any]:
    return {
        "type": "heading",
        "attrs": {"level": 3},
        "content": [{"type": "text", "text": text}],
    }


def _timestamp_label(value: Any) -> str:
    if not isinstance(value, int) or isinstance(value, bool):
        return "time unavailable"
    seconds = value // 1000
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class JiraClient:
    async def create_issue(self, task: dict[str, Any], meeting_url: str | None) -> str:
        required = {
            "JIRA_BASE_URL": settings.jira_base_url,
            "JIRA_EMAIL": settings.jira_email,
            "JIRA_API_TOKEN": settings.jira_api_token,
            "JIRA_PROJECT_KEY": settings.jira_project_key,
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise RuntimeError(f"Missing Jira configuration: {', '.join(missing)}")

        owner_quote = task.get("owner_evidence_quote") or task.get("evidence_quote")
        owner_speaker = validated_owner_speaker(
            task.get("owner_speaker"), owner_quote or ""
        )
        content = [
            _heading("Context"),
            _paragraph(task.get("description") or task["summary"]),
            _paragraph(f"Meeting: {meeting_url or 'Unknown'}"),
            _heading("Evidence"),
            _paragraph(
                f"Task discussion ({_timestamp_label(task.get('timestamp_ms'))}): "
                f"\"{task['evidence_quote']}\""
            ),
        ]
        if owner_speaker and owner_quote:
            content.append(
                _paragraph(
                    f"Assignment ({_timestamp_label(task.get('owner_timestamp_ms'))}): "
                    f"\"{owner_quote}\""
                )
            )
        context_evidence = task.get("context_evidence") or []
        if context_evidence:
            content.extend(
                [
                    _heading("Related meeting excerpts"),
                    *[
                        _paragraph(
                            f"{_timestamp_label(evidence.get('timestamp_ms'))} — "
                            f"{evidence.get('speaker_name') or 'Unknown speaker'}: "
                            f"\"{evidence.get('quote', '')}\""
                        )
                        for evidence in context_evidence
                        if isinstance(evidence, dict) and evidence.get("quote")
                    ],
                ]
            )
        fields: dict[str, Any] = {
            "project": {"key": settings.jira_project_key},
            "summary": task["summary"],
            "issuetype": {"name": settings.jira_issue_type},
            "description": {"type": "doc", "version": 1, "content": content},
        }
        account_id = settings.jira_speaker_account_ids.get(owner_speaker or "")
        if account_id:
            fields["assignee"] = {"accountId": account_id}
        mapped_priority = settings.jira_priority_map.get(task.get("priority") or "")
        if mapped_priority:
            fields["priority"] = {"name": mapped_priority}
        if task.get("due_date"):
            fields["duedate"] = task["due_date"]
        async with httpx.AsyncClient(
            auth=(settings.jira_email, settings.jira_api_token),
            headers={"Accept": "application/json"},
            timeout=30,
        ) as client:
            response = await client.post(
                f"{settings.jira_base_url}/rest/api/3/issue",
                json={"fields": fields},
            )
            response.raise_for_status()
            return response.json()["key"]
