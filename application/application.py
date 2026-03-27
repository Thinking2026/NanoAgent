from __future__ import annotations

import threading
from pathlib import Path

from config import JsonConfig, load_config
from context.shared_context import SharedContext
from queue.message_queue import MessageQueue
from utils.log import Logger, zap

from .agent_thread import AgentThread
from .user_thread import UserThread


class AgentApplication:
    def __init__(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path)
        self._logger = Logger()
        self._config: JsonConfig | None = None
        self._message_queue: MessageQueue | None = None
        self._shared_context: SharedContext | None = None
        self._stop_event = threading.Event()
        self._agent_thread: AgentThread | None = None
        self._user_thread: UserThread | None = None

        try:
            self._config = load_config(self._config_path)
        except Exception as exc:
            self._logger.error(
                "Failed to load config",
                zap.any("config_path", self._config_path),
                zap.any("error", exc),
            )
            return #如果配置加载失败，应用程序无法继续运行，因此直接返回

        self._message_queue = MessageQueue()
        self._shared_context = SharedContext()

        try:
            self._agent_thread = AgentThread(
                message_queue=self._message_queue,
                shared_context=self._shared_context,
                config=self._config,
                stop_event=self._stop_event,
                logger=self._logger,
            )
            self._user_thread = UserThread(
                message_queue=self._message_queue,
                shared_context=self._shared_context,
                stop_event=self._stop_event,
                logger=self._logger,
            )
        except Exception as exc:
            self._logger.error(
                "Failed to initialize application threads",
                zap.any("error", exc),
            )
            self.request_stop()
            self._cleanup_shared_resources()
            return

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> "AgentApplication":
        return cls(config_path)

    def run(self) -> None:
        if self._config is None or self._agent_thread is None or self._user_thread is None:
            return
        try:
            self._agent_thread.start()
            self._user_thread.start()
            self._wait_for_shutdown()
        except KeyboardInterrupt:
            self._logger.info(
                "recieved shutdown signal, stopping application",
            )
        except Exception as exc:
            self._logger.error(
                "Agent application exited with unexpected error",
                zap.any("error", exc),
            )
        finally:
            self._stop_threads()
            self._cleanup_shared_resources()

    def request_stop(self) -> None:
        self._stop_event.set()
        if self._message_queue is not None:
            self._message_queue.close()

    def _wait_for_shutdown(self) -> None:
        while not self._stop_event.is_set():
            if not self._user_thread.is_alive() or not self._agent_thread.is_alive():
                self.request_stop()
                return
            self._user_thread.join(timeout=1)
            self._agent_thread.join(timeout=1)

    def _stop_threads(self) -> None:
        self.request_stop()
        if self._user_thread is not None:
            self._user_thread.join()
        if self._agent_thread is not None:
            self._agent_thread.join()

    def _cleanup_shared_resources(self) -> None:
        if self._message_queue is not None:
            self._message_queue.release()
        if self._shared_context is not None:
            self._shared_context.release()
