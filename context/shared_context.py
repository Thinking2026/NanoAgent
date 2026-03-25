from __future__ import annotations

import threading
from typing import Any

from schemas import AgentEvent, ChatMessage


class SharedContext:
    def __init__(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt
        self._messages: list[ChatMessage] = []
        self._events: list[AgentEvent] = []
        self._lock = threading.RLock()

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def append_message(self, message: ChatMessage) -> None:
        with self._lock:
            self._messages.append(message)

    def append_event(self, event: AgentEvent) -> None:
        with self._lock:
            self._events.append(event)

    def get_conversation(self) -> list[ChatMessage]:
        with self._lock:
            return list(self._messages)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "system_prompt": self._system_prompt,
                "messages": list(self._messages),
                "events": list(self._events),
            }
