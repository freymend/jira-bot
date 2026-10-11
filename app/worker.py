from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
from typing import Any

from app import db
from app.network import is_transient_network_error, retry_after_seconds
from app.pipeline import attendee, finalize_meeting

POLL_INTERVAL_SECONDS = 2
HEARTBEAT_INTERVAL_SECONDS = 10
LEASE_SECONDS = 60

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jira-bot.worker")


async def _maintain_lease(job_id: int, worker_id: str) -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
        if not db.renew_job_lease(job_id, worker_id, lease_seconds=LEASE_SECONDS):
            logger.error("Lost lease for background job %s", job_id)
            return


async def _process_job(job: dict[str, Any], worker_id: str) -> None:
    job_id = int(job["id"])
    bot_id = str(job["bot_id"])
    attempts = int(job["attempts"])
    lease_task = asyncio.create_task(_maintain_lease(job_id, worker_id))
    try:
        await finalize_meeting(bot_id)
    except Exception as exc:
        retryable = is_transient_network_error(exc)
        status = db.fail_job(
            job_id,
            worker_id,
            str(exc),
            attempts=attempts,
            retryable=retryable,
            retry_after_seconds=retry_after_seconds(exc),
        )
        db.set_meeting_status(
            bot_id,
            "retrying" if status == "queued" else "error",
            str(exc)[:1000],
        )
        logger.exception(
            "Finalization job %s for bot %s failed; retryable=%s status=%s",
            job_id,
            bot_id,
            retryable,
            status,
        )
    else:
        if not db.finish_job(job_id, worker_id):
            logger.error(
                "Could not mark background job %s complete: lease lost", job_id
            )
    finally:
        lease_task.cancel()
        try:
            await lease_task
        except asyncio.CancelledError:
            pass


async def run_worker() -> None:
    db.init_db()
    worker_id = os.getenv("WORKER_ID") or f"{socket.gethostname()}:{os.getpid()}"
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)

    logger.info("Background worker %s started", worker_id)
    while not stop.is_set():
        db.record_worker_heartbeat(worker_id)
        job = db.claim_next_job(worker_id, lease_seconds=LEASE_SECONDS)
        if job is not None:
            await _process_job(job, worker_id)
            continue
        try:
            await asyncio.wait_for(stop.wait(), timeout=POLL_INTERVAL_SECONDS)
        except TimeoutError:
            pass
    logger.info("Background worker %s stopped", worker_id)


async def main() -> None:
    try:
        await run_worker()
    finally:
        await attendee.close()


if __name__ == "__main__":
    asyncio.run(main())
