from __future__ import annotations

import sys
import time
import threading

from context.shared_context import SharedContext
from queue.message_queue import MessageQueue
from schemas import ChatMessage, SessionStatus


class UserThread(threading.Thread):
    def __init__(
        self,
        message_queue: MessageQueue,
        shared_context: SharedContext,
    ) -> None:
        super().__init__(name="UserThread", daemon=True)
        self._message_queue = message_queue
        self._shared_context = shared_context
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set() and not self._message_queue.is_closed():
            self._print_prompt()
            user_input = sys.stdin.readline()
            if not user_input:
                continue

            stripped = user_input.strip()
            if not stripped:
                continue

            if stripped.lower() in {"exit", "quit"}:
                self._message_queue.close()
                self.stop()
                break

            message = ChatMessage(role="user", content=stripped)
            self._message_queue.send_user_message(message)
            self._wait_for_agent_message()

        print("Goodbye!")

    def _print_prompt(self) -> None:
        status = self._shared_context.get_session_status()
        if status == SessionStatus.NEW_TASK:
            print("Can I Help You ?")
            return
        print("To better solve the problem, you can provide the AI with solution prompts")

    def _wait_for_agent_message(self) -> None:
        while not self._stop_event.is_set() and not self._message_queue.is_closed():
            message = self._message_queue.get_agent_message(timeout=0.01)
            if message is not None:
                print(f"Agent: {message.content}")
                return
            print("Solving...")
            time.sleep(5)
