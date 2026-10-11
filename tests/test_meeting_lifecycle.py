from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app import db
from app.main import app as api_app
from app.config import settings
from app.meeting_lifecycle import (
    _assessment_means_leave,
    check_meeting_leave_decisions,
    transcript_indicates_wrap_up,
)


class FakeAttendee:
    def __init__(
        self, *, fail: bool = False, bot_state: str = "joined_recording"
    ) -> None:
        self.calls: list[str] = []
        self.get_bot_calls: list[str] = []
        self.fail = fail
        self.bot_state = bot_state

    async def get_bot(self, bot_id: str) -> dict[str, str]:
        self.get_bot_calls.append(bot_id)
        return {"id": bot_id, "state": self.bot_state}

    async def leave_bot(self, bot_id: str) -> dict[str, bool]:
        self.calls.append(bot_id)
        if self.fail:
            raise RuntimeError("leave outcome unknown")
        return {"ok": True}


class MeetingLeaveTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        test_settings = replace(
            settings,
            database_path=str(Path(self.temp_dir.name) / "meeting-leave.sqlite3"),
            meeting_auto_leave_enabled=True,
            meeting_leave_quiet_seconds=90,
            meeting_leave_max_overrun_minutes=60,
            meeting_leave_max_runtime_minutes=120,
            meeting_leave_confidence_threshold=0.9,
        )
        self.db_settings_patch = patch("app.db.settings", test_settings)
        self.lifecycle_settings_patch = patch(
            "app.meeting_lifecycle.settings", test_settings
        )
        self.db_settings_patch.start()
        self.lifecycle_settings_patch.start()
        db.init_db()

    def tearDown(self) -> None:
        self.lifecycle_settings_patch.stop()
        self.db_settings_patch.stop()
        self.temp_dir.cleanup()

    def _active_meeting(
        self,
        bot_id: str = "bot-1",
        *,
        scheduled_end_at: str | None = None,
        participant_tracking_enabled: bool = False,
    ) -> dict[str, object]:
        db.create_meeting(
            bot_id,
            "https://meet.google.com/abc-defg-hij",
            "joined_recording",
            scheduled_end_at=scheduled_end_at,
            participant_tracking_enabled=participant_tracking_enabled,
        )
        db.save_segments(
            bot_id,
            [
                {
                    "id": "segment-1",
                    "timestamp_ms": 1000,
                    "speaker_name": "Alex",
                    "text": "We should review the launch plan.",
                }
            ],
        )
        meeting = db.get_meeting(bot_id)
        assert meeting is not None
        return meeting

    def _quiet_time(self, meeting: dict[str, object], seconds: int = 90) -> datetime:
        activity_at = meeting["last_transcript_received_at"]
        assert isinstance(activity_at, str)
        return datetime.fromisoformat(activity_at) + timedelta(seconds=seconds)

    async def test_wrap_up_requires_quiet_window_then_requests_leave_once(self) -> None:
        meeting = self._active_meeting()
        attendee = FakeAttendee()
        quiet_at = self._quiet_time(meeting)
        classifier = AsyncMock(return_value=True)

        with patch(
            "app.meeting_lifecycle.transcript_indicates_wrap_up", new=classifier
        ):
            self.assertEqual(
                await check_meeting_leave_decisions(
                    attendee, quiet_at - timedelta(seconds=1)
                ),
                0,
            )
            classifier.assert_not_awaited()
            self.assertEqual(await check_meeting_leave_decisions(attendee, quiet_at), 1)
            self.assertEqual(await check_meeting_leave_decisions(attendee, quiet_at), 0)

        self.assertEqual(attendee.calls, ["bot-1"])
        saved = db.get_meeting("bot-1")
        assert saved is not None
        self.assertEqual(saved["leave_status"], "requested")
        self.assertEqual(saved["leave_reason"], "wrap_up")

    async def test_negative_assessment_is_not_repeated_until_new_transcript(
        self,
    ) -> None:
        meeting = self._active_meeting()
        attendee = FakeAttendee()
        quiet_at = self._quiet_time(meeting)
        classifier = AsyncMock(side_effect=[False, True])

        with patch(
            "app.meeting_lifecycle.transcript_indicates_wrap_up", new=classifier
        ):
            self.assertEqual(await check_meeting_leave_decisions(attendee, quiet_at), 0)
            self.assertEqual(await check_meeting_leave_decisions(attendee, quiet_at), 0)
            classifier.assert_awaited_once()

            db.save_segments(
                "bot-1",
                [
                    {
                        "id": "segment-2",
                        "timestamp_ms": 2000,
                        "speaker_name": "Alex",
                        "text": "That covers everything. Thanks, everyone.",
                    }
                ],
            )
            updated = db.get_meeting("bot-1")
            assert updated is not None
            self.assertEqual(
                await check_meeting_leave_decisions(
                    attendee, self._quiet_time(updated)
                ),
                1,
            )

        self.assertEqual(classifier.await_count, 2)
        self.assertEqual(attendee.calls, ["bot-1"])

    async def test_calendar_end_is_soft_until_configured_overrun_cap(self) -> None:
        now = datetime.now(UTC)
        event_end = now - timedelta(minutes=59)
        self._active_meeting(scheduled_end_at=event_end.isoformat())
        attendee = FakeAttendee()
        classifier = AsyncMock(return_value=False)

        with patch(
            "app.meeting_lifecycle.transcript_indicates_wrap_up", new=classifier
        ):
            self.assertEqual(await check_meeting_leave_decisions(attendee, now), 0)
            self.assertEqual(attendee.calls, [])
            self.assertEqual(
                await check_meeting_leave_decisions(
                    attendee, event_end + timedelta(minutes=61)
                ),
                1,
            )

        self.assertEqual(attendee.calls, ["bot-1"])
        saved = db.get_meeting("bot-1")
        assert saved is not None
        self.assertEqual(saved["leave_reason"], "maximum_runtime")

    async def test_meeting_without_calendar_end_has_maximum_runtime_cap(self) -> None:
        meeting = self._active_meeting()
        created_at = meeting["created_at"]
        assert isinstance(created_at, str)
        cap_time = datetime.fromisoformat(created_at) + timedelta(minutes=121)
        attendee = FakeAttendee()

        self.assertEqual(await check_meeting_leave_decisions(attendee, cap_time), 1)
        self.assertEqual(attendee.calls, ["bot-1"])

    async def test_uncertain_leave_retries_only_after_live_recording_confirmation(
        self,
    ) -> None:
        now = datetime.now(UTC)
        event_end = now - timedelta(minutes=61)
        self._active_meeting(scheduled_end_at=event_end.isoformat())
        attendee = FakeAttendee(fail=True)

        self.assertEqual(await check_meeting_leave_decisions(attendee, now), 0)
        self.assertEqual(await check_meeting_leave_decisions(attendee, now), 0)
        self.assertEqual(
            await check_meeting_leave_decisions(attendee, now + timedelta(seconds=59)),
            0,
        )
        self.assertEqual(attendee.calls, ["bot-1"])
        self.assertEqual(
            await check_meeting_leave_decisions(attendee, now + timedelta(seconds=61)),
            0,
        )
        self.assertEqual(attendee.get_bot_calls, ["bot-1"])
        self.assertEqual(attendee.calls, ["bot-1", "bot-1"])
        saved = db.get_meeting("bot-1")
        assert saved is not None
        self.assertEqual(saved["leave_status"], "uncertain")
        self.assertEqual(saved["leave_attempt_count"], 2)

    async def test_uncertain_leave_does_not_retry_when_bot_is_no_longer_recording(
        self,
    ) -> None:
        now = datetime.now(UTC)
        event_end = now - timedelta(minutes=61)
        self._active_meeting(scheduled_end_at=event_end.isoformat())
        attendee = FakeAttendee(fail=True)

        await check_meeting_leave_decisions(attendee, now)
        attendee.bot_state = "fatal_error"
        await check_meeting_leave_decisions(attendee, now + timedelta(seconds=61))

        self.assertEqual(attendee.calls, ["bot-1"])
        self.assertEqual(attendee.get_bot_calls, ["bot-1"])
        saved = db.get_meeting("bot-1")
        assert saved is not None
        self.assertEqual(saved["bot_state"], "fatal_error")
        self.assertEqual(saved["leave_status"], "uncertain")

    async def test_empty_roster_leaves_only_after_bot_is_confirmed_recording(
        self,
    ) -> None:
        self._active_meeting(participant_tracking_enabled=True)
        self.assertTrue(
            db.record_participant_event(
                "bot-1", "join-1", "participant-1", "Alex", False, "join", 100
            )
        )
        self.assertTrue(
            db.record_participant_event(
                "bot-1", "leave-1", "participant-1", "Alex", False, "leave", 200
            )
        )
        meeting = db.get_meeting("bot-1")
        assert meeting is not None
        participant_activity = meeting["last_participant_event_at"]
        assert isinstance(participant_activity, str)
        attendee = FakeAttendee()
        quiet_at = datetime.fromisoformat(participant_activity) + timedelta(seconds=90)

        self.assertEqual(
            await check_meeting_leave_decisions(
                attendee, quiet_at - timedelta(seconds=1)
            ),
            0,
        )
        self.assertEqual(await check_meeting_leave_decisions(attendee, quiet_at), 1)
        self.assertEqual(attendee.get_bot_calls, ["bot-1"])
        self.assertEqual(attendee.calls, ["bot-1"])
        saved = db.get_meeting("bot-1")
        assert saved is not None
        self.assertEqual(saved["leave_reason"], "all_participants_left")

    async def test_empty_roster_does_not_leave_if_attendee_lost_connection(
        self,
    ) -> None:
        self._active_meeting(participant_tracking_enabled=True)
        db.record_participant_event(
            "bot-1", "join-1", "participant-1", "Alex", False, "join", 100
        )
        db.record_participant_event(
            "bot-1", "leave-1", "participant-1", "Alex", False, "leave", 200
        )
        meeting = db.get_meeting("bot-1")
        assert meeting is not None
        participant_activity = meeting["last_participant_event_at"]
        assert isinstance(participant_activity, str)
        attendee = FakeAttendee(bot_state="fatal_error")
        quiet_at = datetime.fromisoformat(participant_activity) + timedelta(seconds=90)

        self.assertEqual(await check_meeting_leave_decisions(attendee, quiet_at), 0)
        self.assertEqual(attendee.get_bot_calls, ["bot-1"])
        self.assertEqual(attendee.calls, [])
        saved = db.get_meeting("bot-1")
        assert saved is not None
        self.assertEqual(saved["bot_state"], "fatal_error")

    async def test_join_during_empty_room_confirmation_cancels_leave(self) -> None:
        self._active_meeting(participant_tracking_enabled=True)
        db.record_participant_event(
            "bot-1", "join-1", "participant-1", "Alex", False, "join", 100
        )
        db.record_participant_event(
            "bot-1", "leave-1", "participant-1", "Alex", False, "leave", 200
        )
        meeting = db.get_meeting("bot-1")
        assert meeting is not None
        participant_activity = meeting["last_participant_event_at"]
        assert isinstance(participant_activity, str)
        quiet_at = datetime.fromisoformat(participant_activity) + timedelta(seconds=90)

        class RejoiningAttendee(FakeAttendee):
            async def get_bot(self, bot_id: str) -> dict[str, str]:
                await super().get_bot(bot_id)
                db.record_participant_event(
                    bot_id, "join-2", "participant-1", "Alex", False, "join", 300
                )
                return {"id": bot_id, "state": "joined_recording"}

        attendee = RejoiningAttendee()
        self.assertEqual(await check_meeting_leave_decisions(attendee, quiet_at), 0)
        self.assertEqual(attendee.calls, [])
        self.assertEqual(
            db.list_meetings_for_leave_check()[0]["active_participant_count"], 1
        )

    async def test_wrap_up_assessment_receives_active_human_count(self) -> None:
        self._active_meeting(participant_tracking_enabled=True)
        db.record_participant_event(
            "bot-1", "join-1", "participant-1", "Alex", False, "join", 100
        )
        updated = db.get_meeting("bot-1")
        assert updated is not None
        transcript_at = updated["last_transcript_received_at"]
        participant_at = updated["last_participant_event_at"]
        assert isinstance(transcript_at, str)
        assert isinstance(participant_at, str)
        quiet_at = max(
            datetime.fromisoformat(transcript_at),
            datetime.fromisoformat(participant_at),
        ) + timedelta(seconds=90)
        attendee = FakeAttendee()
        classifier = AsyncMock(return_value=True)

        with patch(
            "app.meeting_lifecycle.transcript_indicates_wrap_up", new=classifier
        ):
            self.assertEqual(await check_meeting_leave_decisions(attendee, quiet_at), 1)

        assert classifier.await_args is not None
        self.assertEqual(classifier.await_args.args[1], 1)
        self.assertEqual(attendee.calls, ["bot-1"])

    async def test_transcript_replay_does_not_reset_quiet_window(self) -> None:
        meeting = self._active_meeting()
        before = meeting["last_transcript_received_at"]
        db.save_segments(
            "bot-1",
            [
                {
                    "id": "segment-1",
                    "timestamp_ms": 1000,
                    "speaker_name": "Alex",
                    "text": "We should review the launch plan.",
                }
            ],
        )
        after = db.get_meeting("bot-1")
        assert after is not None
        self.assertEqual(after["last_transcript_received_at"], before)

    async def test_model_assessment_requires_boolean_and_confidence_threshold(
        self,
    ) -> None:
        self.assertTrue(
            _assessment_means_leave(
                '{"meeting_over":true,"confidence":0.95,"reason":"finished"}'
            )
        )
        self.assertFalse(
            _assessment_means_leave(
                '{"meeting_over":true,"confidence":0.89,"reason":"finished"}'
            )
        )
        self.assertFalse(
            _assessment_means_leave(
                '{"meeting_over":true,"confidence":true,"reason":"finished"}'
            )
        )
        self.assertFalse(
            _assessment_means_leave(
                '{"meeting_over":true,"confidence":1.5,"reason":"finished"}'
            )
        )
        self.assertFalse(_assessment_means_leave("not JSON"))

    async def test_wrap_up_call_uses_configured_local_model_and_untrusted_transcript(
        self,
    ) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"meeting_over":true,"confidence":0.95,"reason":"done"}'
                    )
                )
            ]
        )
        create = AsyncMock(return_value=response)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=AsyncMock(),
        )
        segments = [{"speaker_name": "Alex", "text": "Thanks, that is all for today."}]

        with patch("app.meeting_lifecycle.AsyncOpenAI", return_value=client) as factory:
            self.assertTrue(
                await transcript_indicates_wrap_up(
                    segments,
                    2,
                    host_names="Alex",
                    scheduled_end_reached=True,
                )
            )

        self.assertEqual(factory.call_args.kwargs["base_url"], settings.llm_base_url)
        user_content = create.call_args.kwargs["messages"][1]["content"]
        self.assertIn("Current human participant count: 2", user_content)
        self.assertIn("Meeting host names identified by Attendee: Alex", user_content)
        self.assertIn("Has the scheduled end passed: true", user_content)
        self.assertIn(
            "untrusted data", create.call_args.kwargs["messages"][0]["content"]
        )
        client.close.assert_awaited_once()


class AttendeeWebhookLeaveStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_settings_patch = patch(
            "app.db.settings",
            replace(
                settings,
                database_path=str(Path(self.temp_dir.name) / "webhook.sqlite3"),
            ),
        )
        self.app_settings_patch = patch(
            "app.main.settings",
            replace(
                settings,
                app_api_token="test-app-token",
                google_calendar_auto_join_enabled=False,
                meeting_auto_leave_enabled=False,
            ),
        )
        self.db_settings_patch.start()
        self.app_settings_patch.start()
        db.init_db()

    def tearDown(self) -> None:
        self.app_settings_patch.stop()
        self.db_settings_patch.stop()
        self.temp_dir.cleanup()

    def test_manual_meeting_creation_enables_participant_tracking(self) -> None:
        create_bot = AsyncMock(return_value={"id": "bot-created", "state": "waiting"})
        headers = {"Authorization": "Bearer test-app-token"}
        with (
            patch("app.main.attendee.create_bot", new=create_bot),
            patch("app.main.attendee.close", new=AsyncMock()),
            TestClient(api_app) as client,
        ):
            response = client.post(
                "/meetings",
                headers=headers,
                json={"meeting_url": "https://meet.google.com/abc-defg-hij"},
            )

        self.assertEqual(response.status_code, 200)
        create_bot.assert_awaited_once()
        meeting = db.get_meeting("bot-created")
        assert meeting is not None
        self.assertEqual(meeting["participant_tracking_enabled"], 1)

    def test_manual_leave_is_persisted_and_not_sent_twice(self) -> None:
        db.create_meeting("bot-manual", None, "waiting")
        leave_bot = AsyncMock(return_value={"ok": True})
        headers = {"Authorization": "Bearer test-app-token"}
        with patch("app.main.attendee.leave_bot", new=leave_bot):
            with TestClient(api_app) as client:
                first = client.post("/meetings/bot-manual/leave", headers=headers)
                second = client.post("/meetings/bot-manual/leave", headers=headers)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json(), {"ok": True})
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json(), {"ok": True, "status": "requested"})
        leave_bot.assert_awaited_once_with("bot-manual")
        meeting = db.get_meeting("bot-manual")
        assert meeting is not None
        self.assertEqual(meeting["leave_reason"], "manual")
        self.assertEqual(meeting["leave_status"], "requested")

    def test_participant_webhooks_update_deduplicated_active_roster(self) -> None:
        db.create_meeting(
            "bot-participants",
            None,
            "joined_recording",
            participant_tracking_enabled=True,
        )
        join_payload = {
            "idempotency_key": "delivery-join-1",
            "trigger": "participant_events.join_leave",
            "bot_id": "bot-participants",
            "data": {
                "id": "join-event-1",
                "participant_uuid": "participant-1",
                "participant_name": "Alex",
                "participant_is_host": True,
                "event_type": "join",
                "timestamp_ms": 100,
            },
        }
        leave_payload = {
            **join_payload,
            "idempotency_key": "delivery-leave-1",
            "data": {
                **join_payload["data"],
                "id": "leave-event-1",
                "event_type": "leave",
                "timestamp_ms": 200,
            },
        }
        with patch("app.main.verify_attendee_signature", return_value=True):
            with TestClient(api_app) as client:
                join = client.post("/webhooks/attendee", json=join_payload)
                duplicate_join = client.post("/webhooks/attendee", json=join_payload)
                joined = db.list_meetings_for_leave_check()[0]
                leave = client.post("/webhooks/attendee", json=leave_payload)

        self.assertEqual(join.status_code, 200)
        self.assertEqual(duplicate_join.status_code, 200)
        self.assertEqual(joined["active_participant_count"], 1)
        self.assertEqual(leave.status_code, 200)
        updated = db.list_meetings_for_leave_check()[0]
        self.assertEqual(updated["active_participant_count"], 0)
        self.assertTrue(updated["participant_presence_initialized"])
        self.assertEqual(updated["meeting_host_names"], "Alex")

    def test_final_state_webhook_stops_auto_leave_monitoring(self) -> None:
        db.create_meeting("bot-finished", None, "joined_recording")
        started_payload = {
            "idempotency_key": "state-started",
            "trigger": "bot.state_change",
            "bot_id": "bot-finished",
            "data": {"new_state": "joined_recording"},
        }
        final_payload = {
            "idempotency_key": "state-finished",
            "trigger": "bot.state_change",
            "bot_id": "bot-finished",
            "data": {"event_type": "post_processing_completed"},
        }
        with patch("app.main.verify_attendee_signature", return_value=True):
            with TestClient(api_app) as client:
                client.post("/webhooks/attendee", json=started_payload)
                response = client.post("/webhooks/attendee", json=final_payload)
                duplicate_state = client.post(
                    "/webhooks/attendee", json=started_payload
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(duplicate_state.status_code, 200)
        meeting = db.get_meeting("bot-finished")
        assert meeting is not None
        self.assertEqual(meeting["bot_state"], "post_processing_completed")
        self.assertEqual(db.list_meetings_for_leave_check(), [])


if __name__ == "__main__":
    unittest.main()
