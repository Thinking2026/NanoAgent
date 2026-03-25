from __future__ import annotations

import select
import sys
import threading

from queue.message_queue import MessageQueue
from schemas import ChatMessage


class UserThread(threading.Thread):
    def __init__(
        self,
        message_queue: MessageQueue,
    ) -> None:
        super().__init__(name="UserThread", daemon=True)
        self._message_queue = message_queue
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        print("Agent prototype is running. Type your message, or use `exit` to quit.")
        while not self._stop_event.is_set() and not self._message_queue.is_closed():
            self._drain_agent_messages()
            user_input = self._poll_input(timeout=0.2)
            if user_input is None:
                continue

            stripped = user_input.strip()
            if not stripped:
                continue

            if stripped.lower() in {"exit", "quit"}:
                self._message_queue.send_user_message(
                    ChatMessage(
                        role="system",
                        content="shutdown",
                        metadata={"control": "shutdown"},
                    )
                )
                self._message_queue.close()
                self.stop()
                break

            message = ChatMessage(role="user", content=stripped)
            self._message_queue.send_user_message(message)

        self._drain_agent_messages()

    def _drain_agent_messages(self) -> None:
        while True:
            message = self._message_queue.get_agent_message(timeout=0.01)
            if message is None:
                return
            print(f"Agent: {message.content}")

    @staticmethod
    def _poll_input(timeout: float) -> str | None:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if not ready:
            return None
        return sys.stdin.readline()
