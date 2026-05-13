from __future__ import annotations

import threading
from typing import Callable

from config import ConfigReader
from agent.factory.agent_factory import AgentFactory
from agent.application.driver import PipelineDriver
from infra.eventbus.event_bus import InMemoryEventBus
from schemas.event_bus import EventBus
from schemas.types import UserMessage
from utils.log.log import Logger, zap
from utils.concurrency.message_queue import UserMessageQueue, TaskQueue, AgentMessageQueue
from utils.concurrency.thread_event import ThreadEvent


class PipelineThread(threading.Thread):
    def __init__(
        self,
        agent_msg_queue: AgentMessageQueue,
        task_queue: TaskQueue,
        user_msg_queue: UserMessageQueue,
        config: ConfigReader,
        stop_event: ThreadEvent,
        stop_callback: Callable[[str | None], None],
    ) -> None:
        super().__init__(name="PipelineThread", daemon=False)
        self._task_queue = task_queue
        self._agent_msg_queue = agent_msg_queue
        self._user_msg_queue = user_msg_queue
        self._config = config
        self._stop_event = stop_event
        self._stop_callback = stop_callback
        self._active_driver: PipelineDriver = None
        
        try:
            self._logger = Logger.get_instance()
            self._factory = AgentFactory.from_config(config)
        except Exception as exc:
            self._logger.error("PipelineThread init failed", zap.any("error", str(exc)))
            raise

    def loop_user_message(self, timeout: float) -> UserMessage:
        return self._agent_msg_queue.get(timeout=timeout)

    def publish_msg_to_user(self, msg: UserMessage) -> None:
        self._user_msg_queue.send(msg)

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            while self._is_running():
                # Wait for the next task from the user
                new_task = self._task_queue.get(timeout=None)
                if not self._is_running():
                    self._logger.info("PipelineThread stopping, exiting loop")
                    break
                if new_task is None:
                    self._logger.info("PipelineThread received shutdown signal, exiting loop")
                    break

                event_bus = InMemoryEventBus()
                active_driver = self._factory.build_pipeline_driver(thread=self, event_bus=event_bus)
                pipeline = self._factory.build_pipeline(event_bus)
                active_driver.use_pipeline(pipeline)
                result = active_driver.submit_task(user_id=new_task.user_id, 
                                task_description=new_task.content.strip())
                if not result.succeeded:
                    self._logger.error(
                        "Task execution failed",
                        zap.any("task_id", result.task_id),
                        zap.any("error", result.error_reason),
                    )
                else:
                    self._logger.info(
                        "Task execution succeed",
                        zap.any("task_id", result.task_id),
                    )

        except Exception as exc:
            self._logger.error("PipelineThread crashed", zap.any("error", exc))
        finally:
            self._stop()

    def _stop(self) -> None:
        self._stop_callback(self.name)

    def _is_running(self) -> bool:
        return not self._stop_event.is_set() and not self._is_any_queue_closed()

    def _is_any_queue_closed(self) -> bool:
        return (
            self._task_queue.is_closed()
            or self._agent_msg_queue.is_closed()
            or self._user_msg_queue.is_closed()
        )
