from __future__ import annotations

import threading

from schemas import ChatMessage


class SharedContext:
    def __init__(self) -> None:
        self._system_prompt = ""
        self._archived_tasks: list[list[ChatMessage]] = []
        self._current_task_messages: list[ChatMessage] = []
        self._lock = threading.RLock()

    def get_system_prompt(self) -> str:
        with self._lock:
            return self._system_prompt

    def append_system_prompt(self, text: str) -> None:
        with self._lock:
            self._system_prompt += text

    def append_system_prompt_line(self, text: str) -> None:
        with self._lock:
            self._system_prompt += f"\n{text}"

    def get_conversation_history(self) -> list[ChatMessage]:
        with self._lock:
            history: list[ChatMessage] = []
            for task_messages in self._archived_tasks:
                history.extend(self._clone_messages(task_messages))
            history.extend(self._clone_messages(self._current_task_messages))
            return history

    def append_conversation_message(self, message: ChatMessage) -> None:
        with self._lock:
            self._current_task_messages.append(
                ChatMessage(
                    role=message.role,
                    content=message.content,
                    metadata=dict(message.metadata),
                )
            )

    def clear_conversation_history(self) -> None:
        with self._lock:
            self._archived_tasks.clear()
            self._current_task_messages.clear()

    def archive_current_task(self) -> None:
        with self._lock:
            if not self._current_task_messages:
                return
            self._archived_tasks.append(self._clone_messages(self._current_task_messages))
            self._current_task_messages.clear()

    def clear_current_task(self) -> None:
        with self._lock:
            self._current_task_messages.clear()

    def get_archived_tasks(self) -> list[list[ChatMessage]]:
        with self._lock:
            return [self._clone_messages(task_messages) for task_messages in self._archived_tasks]

    def get_current_task_messages(self) -> list[ChatMessage]:
        with self._lock:
            return self._clone_messages(self._current_task_messages)

    def release(self) -> None:
        with self._lock:
            self._system_prompt = ""
            self._archived_tasks.clear()
            self._current_task_messages.clear()

    @staticmethod
    def _clone_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
        return [
            ChatMessage(
                role=message.role,
                content=message.content,
                metadata=dict(message.metadata),
            )
            for message in messages
        ]
