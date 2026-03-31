from __future__ import annotations

import json
import select
import sys
import threading
import time
from typing import Callable

from context.session import Session
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
        self._session = Session()
        self._stop_event = stop_event
        self._stop_callback = stop_callback
        self._logger = logger

    def stop(self) -> None:
        self._stop_callback(self.name)

    def release_resources(self) -> None:
        return None

    def run(self) -> None:
        try:
            while self._is_running():
                self._print_prompt()
                user_input = self._read_user_input()
                if user_input:
                    stripped = user_input.strip()
                    if stripped:
                        if stripped.lower() in {"exit", "quit"}:
                            self._logger.error(
                                "User requested exit, stopping user thread",
                                zap.any("input", stripped),
                            )
                            break
                        message = ChatMessage(role="user", content=stripped)
                        self._user_to_agent_queue.send_user_message(message)
                        self._session.begin()
                self._wait_for_agent_message()
        except Exception as exc:
            self._logger.error("User thread crashed", zap.any("error", exc))
        finally:
            self.release_resources()
            self.stop()

    def _print_prompt(self) -> None:
        status = self._session.get_status()
        if status == SessionStatus.NEW_TASK:
            print("Can I Help You ?")
            return
        print("To better solve the problem, you can provide the AI with solution prompts")

    def _wait_for_agent_message(self) -> None:
        while self._can_wait_for_agent_message():
            message = self._agent_to_user_queue.get_agent_message(timeout=1)
            if message is not None:
                print(self._format_agent_message(message))
                self._sync_session_status_from_agent_message(message)
                break

            print("Assistant: thinking and solving...")
            time.sleep(2)

    def _format_agent_message(self, message: ChatMessage) -> str:
        message_source = message.metadata.get("source")
        if message_source == "tool":
            return self._format_tool_message(message)
        if message_source == "rag":
            return self._format_rag_message(message)
        return f"Assistant: {message.content}"

    def _format_tool_message(self, message: ChatMessage) -> str:
        tool_name = str(message.metadata.get("tool_name", "unknown"))
        parameters = message.metadata.get("tool_arguments", {})
        result = message.metadata.get("tool_result", message.content)
        serialized_parameters = json.dumps(parameters, ensure_ascii=False)
        serialized_result = json.dumps(result, ensure_ascii=False)
        return (
            "Assistant: invoke a tool call, "
            f"[tool name]: {tool_name} "
            f"[input parameters]: {serialized_parameters}, "
            f"[result]: {serialized_result}"
        )

    def _format_rag_message(self, message: ChatMessage) -> str:
        source_name = str(message.metadata.get("rag_source", "unknown"))
        query = str(message.metadata.get("query", ""))
        result = message.metadata.get("rag_result", message.content)
        serialized_result = json.dumps(result, ensure_ascii=False)
        truncated_result = self._truncate_words(serialized_result, word_limit=32)
        return (
            "Assistant: search external datasource, "
            f"[source]: {source_name}, "
            f"[query]: {json.dumps(query, ensure_ascii=False)}, "
            f"[result]: {truncated_result}"
        )

    @staticmethod
    def _truncate_words(content: str, word_limit: int) -> str:
        words = content.split()
        if len(words) <= word_limit:
            return content
        return " ".join(words[:word_limit]) + " ..."

    def _read_user_input(self) -> str | None:
        while self._is_running():
            readable, _, _ = select.select([sys.stdin], [], [], 1)
            if readable:
                return sys.stdin.readline()
            if self._session.get_status() == SessionStatus.IN_PROGRESS:
                break
        return None

    def _sync_session_status_from_agent_message(self, message: ChatMessage) -> None:
        if message.metadata.get("session_status") == SessionStatus.NEW_TASK:
            self._session.reset()

    def _is_running(self) -> bool:
        return not self._stop_event.is_set() and not self._is_any_queue_closed()

    def _can_wait_for_agent_message(self) -> bool:
        return not self._stop_event.is_set() and not self._agent_to_user_queue.is_closed()

    def _is_any_queue_closed(self) -> bool:
        return self._user_to_agent_queue.is_closed() or self._agent_to_user_queue.is_closed()
