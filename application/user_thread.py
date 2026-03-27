from __future__ import annotations

import select
import sys
import threading
import time

from context.shared_context import SharedContext
from queue.message_queue import MessageQueue
from schemas import ChatMessage, SessionStatus, SystemMessage
from utils.log import Logger, zap


class UserThread(threading.Thread):
    def __init__(
        self,
        message_queue: MessageQueue,
        shared_context: SharedContext,
        stop_event: threading.Event,
        logger: Logger,
    ) -> None:
        super().__init__(name="UserThread", daemon=False)
        self._message_queue = message_queue
        self._shared_context = shared_context
        self._stop_event = stop_event
        self._logger = logger

    def stop(self) -> None:
        self._stop_event.set()

    def release_resources(self) -> None:
        return None

    def run(self) -> None:
        try:
            while not self._stop_event.is_set() and not self._message_queue.is_closed():
                self._print_prompt()
                user_input = self._read_user_input()
                if not user_input:
                    continue

                stripped = user_input.strip()
                if not stripped:
                    continue

                if stripped.lower() in {"exit", "quit"}:
                    self._message_queue.send_user_message(
                        SystemMessage(command="quit", content=stripped)
                    )
                    break

                message = ChatMessage(role="user", content=stripped)
                self._message_queue.send_user_message(message)
                self._wait_for_agent_message()
        except Exception as exc:
            self._logger.error("User thread crashed", zap.any("error", exc))
        finally:
            self.release_resources()
            self.stop()
            print("Goodbye!")

    def _print_prompt(self) -> None:
        status = self._shared_context.get_session_status()
        if status == SessionStatus.NEW_TASK:
            print("Can I Help You ?")
            return
        print("To better solve the problem, you can provide the AI with solution prompts")

    def _wait_for_agent_message(self) -> None:
        while not self._stop_event.is_set() and not self._message_queue.is_closed():
            message = self._message_queue.get_agent_message(timeout=0.5)
            if message is not None:
                print(f"Assistant: {message.content}")
                return
            print("Assistant: thinking and solving...")
            time.sleep(5)

    def _read_user_input(self) -> str | None:
        while not self._stop_event.is_set() and not self._message_queue.is_closed():
            readable, _, _ = select.select([sys.stdin], [], [], 0.5)
            if readable:
                return sys.stdin.readline()
        return None
