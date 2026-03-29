from __future__ import annotations

import time
from collections import deque
from threading import Condition, RLock
from typing import Optional, TypeVar

from schemas import ChatMessage, ThreadMessage

T = TypeVar("T")


class MessageQueue:
    def __init__(self) -> None:
        self._user_to_agent: deque[ThreadMessage] = deque()
        self._agent_to_user: deque[ChatMessage] = deque()
        self._lock = RLock()
        self._user_to_agent_condition = Condition(self._lock)
        self._agent_to_user_condition = Condition(self._lock)
        self._closed = False

    def send_user_message(self, message: ThreadMessage) -> None:
        with self._user_to_agent_condition:
            if self._closed:
                return
            self._user_to_agent.append(message)
            self._user_to_agent_condition.notify_all()

    def get_user_message(self, timeout: float | None = None) -> Optional[ThreadMessage]:
        return self._safe_get(
            self._user_to_agent,
            self._user_to_agent_condition,
            timeout,
        )

    def send_agent_message(self, message: ChatMessage) -> None:
        with self._agent_to_user_condition:
            if self._closed:
                return
            self._agent_to_user.append(message)
            self._agent_to_user_condition.notify_all()

    def get_agent_message(self, timeout: float) -> Optional[ChatMessage]:
        return self._safe_get(
            self._agent_to_user,
            self._agent_to_user_condition,
            timeout,
        )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._notify_all_conditions()

    def release(self) -> None:
        with self._lock:
            self._user_to_agent.clear()
            self._agent_to_user.clear()
            self._closed = True
            self._notify_all_conditions()

    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def _safe_get(
        self,
        target_queue: deque[T],
        condition: Condition,
        timeout: float | None = None,
    ) -> Optional[T]:
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        deadline = None if timeout is None else time.monotonic() + timeout
        with condition:
            while not target_queue:
                if self._closed:
                    return None
                if deadline is None:
                    condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                condition.wait(timeout=remaining)
            return target_queue.popleft()

    def _notify_all_conditions(self) -> None:
        self._user_to_agent_condition.notify_all()
        self._agent_to_user_condition.notify_all()
