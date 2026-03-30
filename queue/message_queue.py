from __future__ import annotations

import time
from collections import deque
from threading import Condition, RLock
from typing import Optional

from schemas import ChatMessage

class _BaseMessageQueue:
    def __init__(self) -> None:
        self._queue: deque[ChatMessage] = deque()
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._closed = False

    def send(self, message: ChatMessage) -> None:
        with self._condition:
            if self._closed:
                return
            self._queue.append(message)
            self._condition.notify_all()

    def get(self, timeout: float | None = None) -> Optional[ChatMessage]:
        return self._safe_get(timeout=timeout)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._condition.notify_all()

    def release(self) -> None:
        with self._lock:
            self._queue.clear()
            self._closed = True
            self._condition.notify_all()

    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def _safe_get(self, timeout: float | None = None) -> Optional[ChatMessage]:
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._queue:
                if self._closed:
                    return None
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)
            return self._queue.popleft()


class UserToAgentQueue(_BaseMessageQueue):
    def send_user_message(self, message: ChatMessage) -> None:
        self.send(message)

    def get_user_message(self, timeout: float | None = None) -> Optional[ChatMessage]:
        return self.get(timeout=timeout)


class AgentToUserQueue(_BaseMessageQueue):
    def send_agent_message(self, message: ChatMessage) -> None:
        self.send(message)

    def get_agent_message(self, timeout: float | None = None) -> Optional[ChatMessage]:
        return self.get(timeout=timeout)
