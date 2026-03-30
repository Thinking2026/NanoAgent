from __future__ import annotations

import select
import sys
import threading
import time
from typing import Callable

from context.shared_context import SharedContext
from queue.message_queue import AgentToUserQueue, UserToAgentQueue
from schemas import ChatMessage, SessionStatus
from utils.log import Logger, zap
from utils.thread_event import ThreadEvent


class UserThread(threading.Thread):
    def __init__(
        self,
        user_to_agent_queue: UserToAgentQueue,
        agent_to_user_queue: AgentToUserQueue,
        shared_context: SharedContext,
        stop_event: ThreadEvent,
        stop_callback: Callable[[str | None], None],
        logger: Logger,
    ) -> None:
        super().__init__(name="UserThread", daemon=False)
        self._user_to_agent_queue = user_to_agent_queue
        self._agent_to_user_queue = agent_to_user_queue
        self._shared_context = shared_context
        self._stop_event = stop_event
        self._stop_callback = stop_callback
        self._logger = logger

    def stop(self) -> None:
        self._stop_callback(self.name)

    def release_resources(self) -> None:
        return None

    def run(self) -> None:
        prompt_shown = False
        try:
            while not self._stop_event.is_set() and not self._is_any_queue_closed():
                if not prompt_shown:
                    self._print_prompt()
                    prompt_shown = True
                user_input = self._read_user_input()
                if not user_input:
                    continue

                stripped = user_input.strip()
                if not stripped:
                    continue

                prompt_shown = False
                if stripped.lower() in {"exit", "quit"}:
                    self._logger.error(
                        "User requested exit, stopping user thread",
                        zap.any("input", stripped),
                    )
                    break

                message = ChatMessage(role="user", content=stripped)
                self._user_to_agent_queue.send_user_message(message)
                self._wait_for_agent_message()
        except Exception as exc:
            self._logger.error("User thread crashed", zap.any("error", exc))
        finally:
            self.release_resources()
            self.stop()

    def _print_prompt(self) -> None:
        status = self._shared_context.get_session_status()
        if status == SessionStatus.NEW_TASK:
            print("Can I Help You ?")
            return
        print("To better solve the problem, you can provide the AI with solution prompts")

    def _wait_for_agent_message(self) -> None:
        while not self._stop_event.is_set() and not self._agent_to_user_queue.is_closed():
            message = self._agent_to_user_queue.get_agent_message(timeout=1)
            if message is not None:
                print(f"Assistant: {message.content}")
                continue

            print("Assistant: thinking and solving...")
            time.sleep(3)

    def _read_user_input(self) -> str | None:
        while not self._stop_event.is_set() and not self._is_any_queue_closed():
            readable, _, _ = select.select([sys.stdin], [], [], 1)
            if readable:
                return sys.stdin.readline()
        return None

    def _is_any_queue_closed(self) -> bool:
        return self._user_to_agent_queue.is_closed() or self._agent_to_user_queue.is_closed()
