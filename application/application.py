from __future__ import annotations

import threading
from pathlib import Path

from config import JsonConfig, load_config
from context.shared_context import SharedContext
from queue.message_queue import MessageQueue

from .agent_thread import AgentThread
from .user_thread import UserThread


class AgentApplication:
    def __init__(self, config: JsonConfig) -> None:
        self._config = config
        self._message_queue = MessageQueue()
        self._shared_context = SharedContext()
        self._stop_event = threading.Event()
        self._agent_thread: AgentThread | None = None
        self._user_thread: UserThread | None = None
        try:
            self._agent_thread = AgentThread(
                message_queue=self._message_queue,
                shared_context=self._shared_context,
                config=self._config,
                stop_event=self._stop_event,
            )
            self._user_thread = UserThread(
                message_queue=self._message_queue,
                shared_context=self._shared_context,
                stop_event=self._stop_event,
            )
        except Exception:
            self.request_shutdown()
            if self._user_thread is not None:
                self._user_thread.release_resources()
            if self._agent_thread is not None:
                self._agent_thread.release_resources()
            self._release_shared_resources()
            raise

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> "AgentApplication":
        return cls(load_config(config_path))

    def run(self) -> None:
        try:
            self._agent_thread.start()
            self._user_thread.start()
            self._wait_for_shutdown()
        except KeyboardInterrupt:
            self._shutdown_threads()
        finally:
            self._shutdown_threads()
            self._release_shared_resources()

    def request_shutdown(self) -> None:
        self._stop_event.set()
        self._message_queue.close()

    def _wait_for_shutdown(self) -> None:
        while not self._stop_event.is_set():
            user_error = self._user_thread.get_run_error()
            agent_error = self._agent_thread.get_run_error()
            if user_error is not None:
                self.request_shutdown()
                raise user_error
            if agent_error is not None:
                self.request_shutdown()
                raise agent_error
            if not self._user_thread.is_alive() and not self._agent_thread.is_alive():
                return
            if not self._user_thread.is_alive() and self._agent_thread.is_alive():
                self.request_shutdown()
                self._agent_thread.join(timeout=0.1)
                continue
            if not self._agent_thread.is_alive() and self._user_thread.is_alive():
                self._shutdown_threads()
                return
            self._user_thread.join(timeout=0.1)
            self._agent_thread.join(timeout=0.1)
        self._shutdown_threads()

    def _shutdown_threads(self) -> None:
        self.request_shutdown()
        if self._user_thread is not None and self._user_thread.is_alive():
            self._user_thread.join(timeout=1)
        if self._agent_thread is not None and self._agent_thread.is_alive():
            self._agent_thread.join(timeout=1)

    def _release_shared_resources(self) -> None:
        self._message_queue.release()
        self._shared_context.release()
