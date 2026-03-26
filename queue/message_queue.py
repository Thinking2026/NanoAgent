from __future__ import annotations

import time
from collections import deque
from threading import Condition
from typing import Optional, TypeVar

from schemas import ChatMessage, ThreadMessage

T = TypeVar("T")


class MessageQueue:
    def __init__(self) -> None:
        self._user_to_agent: deque[ThreadMessage] = deque()
        self._agent_to_user: deque[ChatMessage] = deque()
        self._condition = Condition()
        self._closed = False

    def send_user_message(self, message: ThreadMessage) -> None:
        with self._condition:
            if self._closed:
                return
            self._user_to_agent.append(message)
            self._condition.notify_all()

    def get_user_message(self, timeout: float | None = None) -> Optional[ThreadMessage]:
        return self._safe_get(self._user_to_agent, timeout)

    def send_agent_message(self, message: ChatMessage) -> None:
        with self._condition:
            if self._closed:
                return
            self._agent_to_user.append(message)
            self._condition.notify_all()

    def get_agent_message(self, timeout: float | None = None) -> Optional[ChatMessage]:
        return self._safe_get(self._agent_to_user, timeout)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def release(self) -> None:
        with self._condition:
            self._user_to_agent.clear()
            self._agent_to_user.clear()
            self._closed = True
            self._condition.notify_all()

    def is_closed(self) -> bool:
        return self._closed

    def _safe_get(
        self,
        target_queue: deque[T],
        timeout: float | None = None,
    ) -> Optional[T]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not target_queue:
                if self._closed:
                    return None
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)
            return target_queue.popleft()
