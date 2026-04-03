from __future__ import annotations

import json
import os
import select
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from config import ConfigValueReader, JsonConfig
from queue.message_queue import AgentToUserQueue, UserToAgentQueue
from schemas import ChatMessage, SessionStatus
from utils.log import Logger, zap
from utils.thread_event import ThreadEvent


class UserThread(threading.Thread):
    def __init__(
        self,
        user_to_agent_queue: UserToAgentQueue,
        agent_to_user_queue: AgentToUserQueue,
        config: JsonConfig,
        stop_event: ThreadEvent,
        stop_callback: Callable[[str | None], None],
        logger: Logger,
    ) -> None:
        super().__init__(name="UserThread", daemon=False)
        self._user_to_agent_queue = user_to_agent_queue
        self._agent_to_user_queue = agent_to_user_queue
        self._config = config
        self._config_value_reader = ConfigValueReader(config)
        self._stop_event = stop_event
        self._stop_callback = stop_callback
        self._logger = logger
        self._new_task_user_input_timeout_seconds = self._config_value_reader.positive_float(
            "agent.latency.new_task_user_input_timeout_seconds",
            60.0,
        )
        self._in_progress_wait_command_timeout_seconds = self._config_value_reader.positive_float(
            "agent.latency.in_progress_wait_command_timeout_seconds",
            5.0,
        )
        self._hint_input_timeout_seconds = self._config_value_reader.positive_float(
            "agent.latency.hint_input_timeout_seconds",
            60.0,
        )
        self._agent_message_poll_timeout_seconds = self._config_value_reader.positive_float(
            "agent.latency.agent_message_poll_timeout_seconds",
            0.1,
        )
        self._progress_notice_interval_seconds = self._config_value_reader.positive_float(
            "agent.latency.user_progress_notice_interval_seconds",
            2.0,
        )
        self._task_name = str(self._config.get("task.name", "external_sorting")).strip() or "external_sorting"
        self._project_root = Path(__file__).resolve().parent.parent
        self._task_source_dir = Path(
            os.environ.get(
                "NANOAGENT_TASK_SOURCE_DIR",
                self._project_root / "testing" / "tasks" / self._task_name,
            )
        )
        self._task_runtime_dir = Path(
            os.environ.get(
                "NANOAGENT_TASK_RUNTIME_DIR",
                self._project_root / "runtime" / self._task_name,
            )
        )
        self._prompt_file_path = Path(
            os.environ.get(
                "NANOAGENT_TASK_PROMPT_FILE",
                self._task_source_dir / "prompt.txt",
            )
        )
        self._ui_session_status = SessionStatus.NEW_TASK
        self._last_prompt_status: SessionStatus | None = None
        self._last_progress_notice_at = 0.0
        self._task_started = False

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
                if self._should_finish():
                    print("Task finished, bye")
                    break

                user_input = self._wait_for_user_input()
                need_quit = self._handle_user_input(user_input)
                if need_quit:
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
            print(f"Assistant: loading task `{self._task_name}` from {self._prompt_file_path}")
        else:
            print("Assistant: task is in progress, you can input a hint at any time")
        self._last_prompt_status = status

    def _drain_agent_messages(self) -> bool:
        displayed_any_message = False
        while self._is_running():
            message = self._agent_to_user_queue.get_agent_message(
                timeout=self._agent_message_poll_timeout_seconds
            )
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

    def _wait_for_user_input(self) -> str | None:
        if self._ui_session_status == SessionStatus.NEW_TASK:
            return self._load_question_from_file()
        return self._wait_for_hint_command()

    def _load_question_from_file(self) -> str | None:
        try:
            content = self._prompt_file_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read task prompt file: {self._prompt_file_path}"
            ) from exc

        if not content:
            raise ValueError(
                f"Task prompt file is empty: {self._prompt_file_path}"
            )
        runtime_dir = self._task_runtime_dir.resolve()
        source_dir = self._task_source_dir.resolve()
        result_path = runtime_dir / "result.txt"
        return (
            f"{content}\n\n"
            "Runtime constraints:\n"
            f"- Current task name: {self._task_name}\n"
            f"- Read-only task input directory: {source_dir}\n"
            f"- Writable runtime directory for all generated files: {runtime_dir}\n"
            f"- Final result file must be written to: {result_path}\n"
            "- All intermediate files, temporary files, and generated outputs must stay under the writable runtime directory.\n"
            "- Do not write any generated file back into the testing/tasks directory. Treat that directory as read-only input.\n"
            "- These runtime constraints override any earlier output path mentioned in the task description.\n"
        )

    def _wait_for_hint_command(self) -> str | None:
        user_input = self._poll_user_input(
            timeout=self._in_progress_wait_command_timeout_seconds,
        )
        if user_input is None:
            return None

        stripped = user_input.strip()
        if stripped.lower() != "wait":
            return user_input

        print("Assistant: hint mode enabled, please input your hint within 60 seconds")
        return self._poll_user_input(timeout=self._hint_input_timeout_seconds)

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
        if self._ui_session_status == SessionStatus.NEW_TASK:
            self._task_started = True
        self._user_to_agent_queue.send_user_message(message)
        return False

    def _print_progress_notice_if_needed(self, displayed_any_message: bool) -> None:
        if self._ui_session_status != SessionStatus.IN_PROGRESS:
            return
        now = time.monotonic()
        if displayed_any_message:
            return
        if now - self._last_progress_notice_at < self._progress_notice_interval_seconds:
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
        if session_status != self._ui_session_status:
            self._ui_session_status = session_status

    def _should_finish(self) -> bool:
        return self._task_started and self._ui_session_status == SessionStatus.NEW_TASK

    def _is_running(self) -> bool:
        return not self._stop_event.is_set() and not self._is_any_queue_closed()

    @staticmethod
    def _is_control_message(message: ChatMessage) -> bool:
        return bool(message.metadata.get("control")) and not message.content.strip()

    def _is_any_queue_closed(self) -> bool:
        return self._user_to_agent_queue.is_closed() or self._agent_to_user_queue.is_closed()
