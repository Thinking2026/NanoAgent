from __future__ import annotations

import json
import select
import sys
import threading
import time
from typing import Callable

from queue.message_queue import AgentToUserQueue, UserToAgentQueue
from schemas import ChatMessage, SessionStatus
from utils.log import Logger, zap
from utils.thread_event import ThreadEvent


class UserThread(threading.Thread):
    def __init__(
        self,
        user_to_agent_queue: UserToAgentQueue,
        agent_to_user_queue: AgentToUserQueue,
        stop_event: ThreadEvent,
        stop_callback: Callable[[str | None], None],
        logger: Logger,
    ) -> None:
        super().__init__(name="UserThread", daemon=False)
        self._user_to_agent_queue = user_to_agent_queue
        self._agent_to_user_queue = agent_to_user_queue
        self._stop_event = stop_event
        self._stop_callback = stop_callback
        self._logger = logger
        self._ui_session_status = SessionStatus.NEW_TASK
        self._last_prompt_status: SessionStatus | None = None
        self._last_progress_notice_at = 0.0

    def stop(self) -> None:
        self._stop_callback(self.name)

    def release_resources(self) -> None:
        return None

    def run(self) -> None:
        try:
            while self._is_running():
                self._print_prompt_if_needed()
                displayed_any_message = self._drain_agent_messages()
                if not self._is_running():
                    break

                user_input = self._poll_user_input(timeout=0.5)
                if self._handle_user_input(user_input):
                    break

                self._print_progress_notice_if_needed(displayed_any_message)
        except Exception as exc:
            self._logger.error("User thread crashed", zap.any("error", exc))
        finally:
            self.release_resources()
            self.stop()

    def _print_prompt_if_needed(self) -> None:
        status = self._ui_session_status
        if status == self._last_prompt_status:
            return
        if status == SessionStatus.NEW_TASK:
            print("Can I Help You ?")
        else:
            print("To better solve the problem, you can provide the AI with solution prompts")
        self._last_prompt_status = status

    def _drain_agent_messages(self) -> bool:
        displayed_any_message = False
        while self._can_wait_for_agent_message():
            message = self._agent_to_user_queue.get_agent_message(timeout=0.01)
            if message is None:
                break
            self._sync_session_status_from_agent_message(message)
            if self._is_control_message(message):
                continue
            print(self._format_agent_message(message))
            displayed_any_message = True
            self._last_progress_notice_at = time.monotonic()
        return displayed_any_message

    def _poll_user_input(self, timeout: float) -> str | None:
        if not self._is_running():
            return None
        readable, _, _ = select.select([sys.stdin], [], [], timeout)
        if readable:
            return sys.stdin.readline()
        return None

    def _handle_user_input(self, user_input: str | None) -> bool:
        if user_input is None:
            return False
        stripped = user_input.strip()
        if not stripped:
            return False
        if stripped.lower() in {"exit", "quit"}:
            self._logger.error(
                "User requested exit, stopping user thread",
                zap.any("input", stripped),
            )
            return True
        message = ChatMessage(role="user", content=stripped)
        self._user_to_agent_queue.send_user_message(message)
        self._last_prompt_status = None
        return False

    def _print_progress_notice_if_needed(self, displayed_any_message: bool) -> None:
        if self._ui_session_status != SessionStatus.IN_PROGRESS:
            return
        now = time.monotonic()
        if displayed_any_message:
            return
        if now - self._last_progress_notice_at < 2.0:
            return
        print("Assistant: thinking and solving...")
        self._last_progress_notice_at = now

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

    def _sync_session_status_from_agent_message(self, message: ChatMessage) -> None:
        session_status = message.metadata.get("session_status")
        if session_status == SessionStatus.NEW_TASK:
            self._ui_session_status = SessionStatus.NEW_TASK
            self._last_prompt_status = None
            return
        if session_status == SessionStatus.IN_PROGRESS:
            self._ui_session_status = SessionStatus.IN_PROGRESS
            self._last_prompt_status = None

    def _is_running(self) -> bool:
        return not self._stop_event.is_set() and not self._is_any_queue_closed()

    def _can_wait_for_agent_message(self) -> bool:
        return not self._stop_event.is_set() and not self._agent_to_user_queue.is_closed()

    @staticmethod
    def _is_control_message(message: ChatMessage) -> bool:
        return bool(message.metadata.get("control")) and not message.content.strip()

    def _is_any_queue_closed(self) -> bool:
        return self._user_to_agent_queue.is_closed() or self._agent_to_user_queue.is_closed()
