from __future__ import annotations

import threading
from pathlib import Path

from config import ConfigReader
from utils.concurrency.message_queue import AgentMessageQueue, TaskQueue, UserMessageQueue
from utils.log.log import Logger, zap
from utils.env_util.runtime_env import (
    get_project_root,
    set_task_environment,
    set_timezone_name,
)
from utils.concurrency.thread_event import ThreadEvent

from .user_thread import UserThread
from agent.application.pipeline_thread import PipelineThread
from jobs import AgentReachHealthJob, JobScheduler, ScheduledJob

class Demo:
    def __init__(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path)
        self._logger = Logger.get_instance()
        self._config: ConfigReader | None = None
        self._task_queue: TaskQueue | None = None
        self._agent_msg_queue: AgentMessageQueue | None = None
        self._user_msg_queue: UserMessageQueue | None = None
        self._stop_event = ThreadEvent()
        self._shutdown_lock = threading.Lock()
        self._job_scheduler: JobScheduler | None = None

        try:
            self._config = ConfigReader(self._config_path)
        except Exception as exc:
            self._logger.error(
                "Failed to load config",
                zap.any("config_path", self._config_path),
                zap.any("error", exc),
            )
            raise
        self._prepare_task_environment()
        self._task_queue = TaskQueue()
        self._agent_msg_queue = AgentMessageQueue()
        self._user_msg_queue = UserMessageQueue()

        try:
            self._agent_thread = PipelineThread(
                task_queue=self._task_queue,
                agent_msg_queue=self._agent_msg_queue,
                user_msg_queue=self._user_msg_queue,
                config=self._config,
                stop_event=self._stop_event,
                stop_callback=self.request_stop,
            )
            self._user_thread = UserThread(
                task_queue=self._task_queue,
                agent_msg_queue=self._agent_msg_queue,
                user_msg_queue=self._user_msg_queue,
                config=self._config,
                stop_event=self._stop_event,
                stop_callback=self.request_stop,
                logger=self._logger,
            )
            self._job_scheduler = self._build_job_scheduler()
        except Exception as exc:
            self._logger.error(
                "Failed to initialize application threads",
                zap.any("error", exc),
            )
            self.release_resources()
            raise

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> "Demo":
        return cls(config_path)

    def run(self) -> None:
        try:
            if self._job_scheduler is not None:
                self._job_scheduler.start()
            self._agent_thread.start()
            self._user_thread.start()
            self._wait_for_shutdown()
        except KeyboardInterrupt:
            self._logger.info(
                "recieved shutdown signal, stopping application",
            )
            self.request_stop(source="KeyboardInterrupt")
        except Exception as exc:
            self._logger.error(
                "Agent application exited with unexpected error",
                zap.any("error", exc),
            )
        finally:
            self._stop_threads()
            self.release_resources()
            self._logger.info(
                "Agent application stopped",
                zap.any("stop_source", self._stop_event.get_source()),
            )

    def request_stop(self, source: str | None = None) -> None:
        stop_source = source or self.__class__.__name__
        with self._shutdown_lock:
            self._stop_event.set(source=stop_source)
            if self._task_queue is not None:
                self._task_queue.close()
            if self._agent_msg_queue is not None:
                self._agent_msg_queue.close()
            if self._user_msg_queue is not None:
                self._user_msg_queue.close()

    def _wait_for_shutdown(self) -> None:
        while not self._stop_event.is_set():
            self._safe_join(self._user_thread, timeout=self._thread_join_timeout_seconds)
            self._safe_join(self._agent_thread, timeout=self._thread_join_timeout_seconds)
        # stop_event is set; wait for threads to actually finish in the required order:
        # user_thread first (must finish displaying all messages), then agent_thread.
        self._safe_join(self._user_thread)
        self._safe_join(self._agent_thread)
        if self._job_scheduler is not None:
            self._job_scheduler.stop(timeout=self._thread_join_timeout_seconds)

    def _stop_threads(self) -> None:
        self.request_stop(source="AgentApplication.stop_threads")
        self._safe_join(self._user_thread)
        self._safe_join(self._agent_thread)

    @staticmethod
    def _safe_join(thread: threading.Thread | None, timeout: float | None = None) -> None:
        if thread is None or thread.ident is None:
            return
        thread.join(timeout=timeout)

    def release_resources(self) -> None:
        if self._task_queue is not None:
            self._task_queue.release()
        if self._agent_msg_queue is not None:
            self._agent_msg_queue.release()
        if self._user_msg_queue is not None:
            self._user_msg_queue.release()

    def _build_job_scheduler(self) -> JobScheduler | None:
        if self._config is None or not bool(self._config.get("jobs.enabled", True)):
            return None
        scheduler = JobScheduler(stop_event=self._stop_event, logger=self._logger)
        if bool(self._config.get("jobs.agent_reach_health.enabled", True)):
            scheduler.register(
                ScheduledJob(
                    name="agent_reach_health",
                    job=AgentReachHealthJob(self._config),
                    interval_seconds=self._config.positive_float(
                        "jobs.agent_reach_health.interval_seconds",
                        300.0,
                    ),
                    run_on_start=bool(self._config.get("jobs.agent_reach_health.run_on_start", True)),
                )
            )
        return scheduler

    def _prepare_task_environment(self) -> None:
        if self._config is None:
            self._logger.warning(
                "Config not loaded, skipping task environment preparation")
            return

        project_root = get_project_root()
        task_name = str(self._config.get("task.name", "external_sorting")).strip() or "external_sorting"
        task_source_dir = project_root / "tests" / "integration" / "tasks" / task_name
        task_runtime_dir = project_root / "var" / "tasks" / task_name
        task_runtime_dir.mkdir(parents=True, exist_ok=True)
        set_task_environment(
            task_name=task_name,
            task_source_dir=task_source_dir,
            task_runtime_dir=task_runtime_dir,
            task_prompt_file=task_source_dir / "prompt.txt",
        )

        timezone = self._config.get("time.timezone", "shanghai")
        set_timezone_name(str(timezone))

    @property
    def _thread_join_timeout_seconds(self) -> float:
        if self._config is None:
            return 1.0
        return self._config.positive_float(
            "agent.latency.thread_join_timeout_seconds",
            1.0,
        )
