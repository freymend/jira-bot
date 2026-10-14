from __future__ import annotations

import asyncio
import unittest
from collections.abc import Awaitable
from unittest.mock import patch

from app import worker
from app.main import _run_resilient_background_task


class ProcessResilienceTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_continues_after_iteration_exception(self) -> None:
        loop = asyncio.get_running_loop()
        signal_handlers = []
        original_wait_for = asyncio.wait_for

        async def stop_after_failure(
            awaitable: Awaitable[object], timeout: float
        ) -> object:
            signal_handlers[0]()
            return await original_wait_for(awaitable, timeout)

        with (
            patch.object(
                loop,
                "add_signal_handler",
                side_effect=lambda _signal, callback: signal_handlers.append(callback),
            ),
            patch("app.worker.db.init_db") as init_db,
            patch("app.worker.db.delete_expired_screen_frames", return_value=0),
            patch(
                "app.worker.db.record_worker_heartbeat",
                side_effect=RuntimeError("temporary database failure"),
            ) as heartbeat,
            patch("app.worker.logger.exception"),
            patch("app.worker.asyncio.wait_for", new=stop_after_failure),
        ):
            await worker.run_worker()

        heartbeat.assert_called_once()
        init_db.assert_not_called()

    async def test_lease_renewal_retries_after_exception(self) -> None:
        async def no_wait(_delay: float) -> None:
            return None

        with (
            patch("app.worker.asyncio.sleep", new=no_wait),
            patch(
                "app.worker.db.renew_job_lease",
                side_effect=[RuntimeError("temporary database failure"), False],
            ) as renew,
            patch("app.worker.logger.exception"),
            patch("app.worker.logger.error"),
        ):
            await worker._maintain_lease(1, "worker-1")

        self.assertEqual(renew.call_count, 2)

    async def test_background_task_restarts_after_unexpected_exception(self) -> None:
        calls = 0

        async def no_wait(_delay: float) -> None:
            return None

        async def runner() -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary background failure")
            raise asyncio.CancelledError

        with (
            patch("app.main.asyncio.sleep", new=no_wait),
            patch("app.main.logger.exception"),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await _run_resilient_background_task("test-task", runner)

        self.assertEqual(calls, 2)


if __name__ == "__main__":
    unittest.main()
