from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from jobs.job_base import BaseJob
from utils.concurrency.thread_event import ThreadEvent
from utils.log.log import Logger, zap


@dataclass(slots=True)
class ScheduledJob:
    name: str
    job: BaseJob
    interval_seconds: float
    run_on_start: bool = True


class JobScheduler:
    def __init__(
        self,
        *,
        stop_event: ThreadEvent,
        logger: Logger | None = None,
        tick_seconds: float = 1.0,
    ) -> None:
        self._stop_event = stop_event
        self._logger = logger or Logger.get_instance()
        self._tick_seconds = max(0.1, tick_seconds)
        self._jobs: list[ScheduledJob] = []
        self._next_run_at: dict[str, float] = {}
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def register(self, scheduled_job: ScheduledJob) -> None:
        with self._lock:
            self._jobs.append(scheduled_job)
            self._next_run_at[scheduled_job.name] = time.monotonic() + scheduled_job.interval_seconds

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            jobs = list(self._jobs)

        for scheduled_job in jobs:
            if scheduled_job.run_on_start:
                self._run_job(scheduled_job)

        self._thread = threading.Thread(name="JobScheduler", target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            now = time.monotonic()
            with self._lock:
                jobs = list(self._jobs)
            for scheduled_job in jobs:
                if now >= self._next_run_at.get(scheduled_job.name, 0.0):
                    self._run_job(scheduled_job)
                    self._next_run_at[scheduled_job.name] = time.monotonic() + scheduled_job.interval_seconds
            self._stop_event.wait(self._tick_seconds)

    def _run_job(self, scheduled_job: ScheduledJob) -> None:
        try:
            result = scheduled_job.job.run_once()
            if result.success:
                self._logger.info(
                    "Scheduled job succeeded",
                    zap.any("job", scheduled_job.name),
                    zap.any("message", result.message),
                )
            else:
                self._logger.warning(
                    "Scheduled job failed",
                    zap.any("job", scheduled_job.name),
                    zap.any("message", result.message),
                )
        except Exception as exc:
            self._logger.error(
                "Scheduled job crashed",
                zap.any("job", scheduled_job.name),
                zap.any("error", exc),
            )
