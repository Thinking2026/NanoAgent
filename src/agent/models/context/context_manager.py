from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from utils.time.time import now as _time_now
from uuid import uuid4

from infra.observability.tracing.tracer import Tracer
from llm.llm_gateway import LLMGateway
from schemas import LLMMessage, UnifiedLLMRequest
from schemas.task import KnowledgeEntry, Plan, Task, UserPreferenceEntry
from schemas.types import LLMRole
from tools.tool_registry import ToolRegistry
from utils.log.log import Logger

if TYPE_CHECKING:
    from agent.models.context.estimator.token_estimator import BaseTokenEstimator
    from agent.models.context.truncation.token_truncation import ContextTruncator
    from config import ConfigReader

_CHARS_PER_TOKEN_FALLBACK = 3.5


@dataclass(frozen=True)
class ToolCallEntry:
    """A single tool call within a multi-call assistant turn."""
    tool_call_id: str
    tool_name: str
    tool_arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolUseMetadata:
    """Metadata for assistant messages that contain one or more tool calls."""
    # Primary call — used when there is exactly one tool call in the turn.
    tool_call_id: str
    tool_name: str
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    # Additional calls when the LLM issues multiple tool calls in one turn.
    extra_calls: tuple[ToolCallEntry, ...] = field(default_factory=tuple)

    def all_call_ids(self) -> list[str]:
        ids = [self.tool_call_id]
        ids.extend(entry.tool_call_id for entry in self.extra_calls)
        return ids

    def all_calls(self) -> list[ToolCallEntry]:
        primary = ToolCallEntry(self.tool_call_id, self.tool_name, self.tool_arguments)
        return [primary, *self.extra_calls]


@dataclass(frozen=True)
class ToolResultMetadata:
    """Metadata for tool-role messages that carry tool execution results."""
    tool_call_id: str
    tool_name: str
    success: bool = True


@dataclass(frozen=True)
class SummaryMetadata:
    """Metadata for synthetic summary messages injected by context compression."""
    stage_index: int
    original_message_count: int


@dataclass(frozen=True)
class ContextMessage:
    id: str
    role: LLMRole
    content: str
    timestamp: datetime = field(default_factory=_time_now)
    token_count: int | None = None
    # Typed metadata — exactly one of these is set depending on role/purpose.
    # assistant tool-call turn: tool_use is set
    # tool result turn: tool_result is set
    # compressed summary: summary is set
    # plain user/assistant turns: all None
    tool_use: ToolUseMetadata | None = None
    tool_result: ToolResultMetadata | None = None
    summary: SummaryMetadata | None = None


@dataclass
class StageRecord:
    """Tracks the lifecycle and outcome of a single plan-step execution."""
    stage_index: int
    plan_step_order: int = 0
    # Ordered list of message IDs belonging to this stage (append-only).
    message_ids: list[str] = field(default_factory=list)
    # LLM-generated summary produced after the stage completes successfully.
    summary: str | None = None
    # Terminal state flags — mutually exclusive.
    completed: bool = False
    dropped: bool = False

    @property
    def first_message_id(self) -> str | None:
        return self.message_ids[0] if self.message_ids else None

    @property
    def last_message_id(self) -> str | None:
        return self.message_ids[-1] if self.message_ids else None

    @property
    def message_count(self) -> int:
        return len(self.message_ids)

    def record_message(self, message_id: str) -> None:
        self.message_ids.append(message_id)

    def is_summarized(self) -> bool:
        return self.summary is not None

    def is_active(self) -> bool:
        return not self.completed and not self.dropped


class ContextManager:
    """Single source of truth for all context sent to the LLM.

    Responsibilities:
    - Owns system_prompt, tool_schemas, knowledge_entries, user_preferences, variables
    - Tracks conversation messages in _ctx_window (mutable) and _history (append-only)
    - Tracks stage boundaries by message ID (immune to index shifting)
    - Assembles and optionally truncates LLMRequest via get_context_window()
    """

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        config: ConfigReader | None = None,
        llm_gateway: LLMGateway | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._config = config
        self._llm_gateway = llm_gateway

        self._system_prompt: str = ""

        self._tool_schemas: list[dict] = []
        if tool_registry is not None:
            self._tool_schemas = tool_registry.get_tool_schemas()

        self._knowledge_entries: list[KnowledgeEntry] = []
        self._user_preferences_entries: list[UserPreferenceEntry] = []
        self._variables: dict[str, Any] = {}

        self._ctx_window: list[ContextMessage] = []
        self._history: list[ContextMessage] = []

        self._stage_records: list[StageRecord] = []
        self._message_id_to_stage: dict[str, int] = {}
        self._active_stage_index: int | None = None
        self._last_success_stage_index: int | None = None

        self._current_token_count: int = 0
        self._pressure_callback: Callable[[float], None] | None = None
        self._pressure_threshold: float = 0.85

        self._streaming_buffers: dict[str, list[str]] = {}
        self._streaming_roles: dict[str, LLMRole] = {}
        self._streaming_metadata: dict[str, dict[str, Any]] = {}

        self._token_truncator: ContextTruncator | None = None
        self._task: Task = None
        self._plan: Plan = None

        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Basic getters
    # ------------------------------------------------------------------

    def set_task(self, task: Task) -> None:
        self._task = task

    def get_task(self) -> Task:
        return self._task

    def set_plan(self, plan: Plan) -> None:
        self._plan = plan

    def get_plan(self) -> Plan:
        return self._plan

    # ------------------------------------------------------------------
    # Configuration setters
    # ------------------------------------------------------------------

    def set_system_prompt(self, prompt: str) -> None:
        with self._lock:
            self._system_prompt = prompt

    def get_system_prompt(self) -> str:
        with self._lock:
            return self._system_prompt

    def append_system_prompt(self, text: str) -> None:
        with self._lock:
            self._system_prompt += text

    def append_system_prompt_line(self, text: str) -> None:
        with self._lock:
            self._system_prompt += f"\n{text}"

    def set_tool_schemas(self, schemas: list[dict[str, Any]]) -> None:
        with self._lock:
            self._tool_schemas = list(schemas)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._tool_schemas)

    def set_knowledge_entries(self, entries: list[KnowledgeEntry]) -> None:
        with self._lock:
            self._knowledge_entries = list(entries)

    def set_user_preferences(self, entries: list[UserPreferenceEntry]) -> None:
        with self._lock:
            self._user_preferences_entries = list(entries)

    def set_variables(self, variables: dict[str, Any]) -> None:
        with self._lock:
            self._variables = dict(variables)

    def get_variables(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._variables)

    # ------------------------------------------------------------------
    # Stage lifecycle
    # ------------------------------------------------------------------

    def begin_stage(self, stage_index: int, plan_step_order: int = 0) -> None:
        with self._lock:
            while len(self._stage_records) <= stage_index:
                self._stage_records.append(
                    StageRecord(
                        stage_index=len(self._stage_records),
                        plan_step_order=plan_step_order,
                    )
                )
            self._active_stage_index = stage_index

    def end_stage(self, stage_index: int, success: bool) -> None:
        """Mark the stage as complete. On success, triggers async LLM summarization."""
        with self._lock:
            if stage_index >= len(self._stage_records):
                return
            record = self._stage_records[stage_index]
            record.completed = True
            if self._active_stage_index == stage_index:
                self._active_stage_index = None
            if success:
                self._last_success_stage_index = stage_index

        if success:
            threading.Thread(
                target=self._generate_stage_summary,
                args=(stage_index,),
                daemon=True,
            ).start()

    def drop_stage(self, stage_index: int) -> None:
        """Remove all ctx_window messages for stage_index. History is unchanged."""
        with self._lock:
            if stage_index >= len(self._stage_records):
                return
            stage_msg_ids = self._get_stage_message_ids(stage_index)
            dropped_tokens = sum(
                m.token_count or 0
                for m in self._ctx_window
                if m.id in stage_msg_ids
            )
            self._ctx_window = [m for m in self._ctx_window if m.id not in stage_msg_ids]
            self._current_token_count = max(0, self._current_token_count - dropped_tokens)
            self._stage_records[stage_index].dropped = True

    def summarize_stage(self, stage_index: int, summary: str) -> None:
        """Replace stage messages in ctx_window with a single summary message."""
        with self._lock:
            if stage_index >= len(self._stage_records):
                return
            stage_msg_ids = self._get_stage_message_ids(stage_index)
            if not stage_msg_ids:
                return

            replaced_tokens = sum(
                m.token_count or 0
                for m in self._ctx_window
                if m.id in stage_msg_ids
            )
            summary_token_count = self._estimate_text_tokens(summary)
            original_count = len(stage_msg_ids)
            summary_msg = ContextMessage(
                id=str(uuid4()),
                role="assistant",
                content=summary,
                token_count=summary_token_count,
                summary=SummaryMetadata(
                    stage_index=stage_index,
                    original_message_count=original_count,
                ),
            )
            new_window: list[ContextMessage] = []
            inserted = False
            for m in self._ctx_window:
                if m.id in stage_msg_ids:
                    if not inserted:
                        new_window.append(summary_msg)
                        self._message_id_to_stage[summary_msg.id] = stage_index
                        inserted = True
                else:
                    new_window.append(m)
            self._ctx_window = new_window
            self._current_token_count = max(
                0, self._current_token_count - replaced_tokens + summary_token_count
            )
            self._stage_records[stage_index].summary = summary

    def get_stage_messages(self, stage_index: int) -> list[LLMMessage]:
        """Return ctx_window messages for stage_index as LLMMessages."""
        with self._lock:
            if stage_index >= len(self._stage_records):
                return []
            if self._stage_records[stage_index].dropped:
                return []
            stage_msg_ids = self._get_stage_message_ids(stage_index)
            msgs = [m for m in self._ctx_window if m.id in stage_msg_ids]
            return self._to_llm_messages(msgs)

    # ------------------------------------------------------------------
    # Message management
    # ------------------------------------------------------------------

    def add_message(
        self,
        role: LLMRole,
        content: str,
        tool_use: ToolUseMetadata | None = None,
        tool_result: ToolResultMetadata | None = None,
    ) -> str:
        """Append a message to ctx_window and history. Returns the message UUID."""
        token_count = self._estimate_text_tokens(content)
        with self._lock:
            msg = ContextMessage(
                id=str(uuid4()),
                role=role,
                content=content,
                token_count=token_count,
                tool_use=tool_use,
                tool_result=tool_result,
            )
            self._ctx_window.append(msg)
            self._history.append(msg)
            self._current_token_count += token_count

            if self._active_stage_index is not None:
                idx = self._active_stage_index
                self._message_id_to_stage[msg.id] = idx
                self._stage_records[idx].record_message(msg.id)

            pressure = self._check_pressure()

        if pressure is not None and self._pressure_callback is not None:
            self._pressure_callback(pressure)

        return msg.id

    def get_conversation_history(self) -> list[LLMMessage]:
        """Return the full append-only history as LLMMessages."""
        with self._lock:
            return self._to_llm_messages(list(self._history))

    def replace_conversation_history(self, messages: list[LLMMessage]) -> None:
        """Replace ctx_window and history (used for checkpoint restore)."""
        with self._lock:
            ctx_msgs = [self._from_llm_message(m) for m in messages]
            self._ctx_window = ctx_msgs
            self._history = list(ctx_msgs)
            self._stage_records = []
            self._message_id_to_stage = {}
            self._active_stage_index = None
            self._last_success_stage_index = None
            self._current_token_count = sum(m.token_count or 0 for m in ctx_msgs)

    # ------------------------------------------------------------------
    # Core: build LLMRequest
    # ------------------------------------------------------------------

    def get_context_window(self, provider_name: str) -> UnifiedLLMRequest:
        """Assemble, optionally truncate, and return the LLMRequest for the LLM."""
        with self._lock:
            system_prompt = self._build_system_prompt()
            repaired = self._repair_context(list(self._ctx_window))

            truncator = self._get_truncator()
            if truncator is not None:
                estimator = self._get_estimator(provider_name)
                total_budget = self._get_total_budget(provider_name)
                truncated = truncator.truncate(repaired, total_budget, estimator)
                messages = self._to_llm_messages(truncated)
            else:
                messages = self._to_llm_messages(repaired)

            return UnifiedLLMRequest(
                system_prompt=system_prompt,
                messages=messages,
                tool_schemas=self._tool_schemas if self._tool_schemas else None,
            )

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    def get_token_usage(self, provider_name: str) -> dict[str, int]:
        """Return per-role token counts for the current context window."""
        with self._lock:
            estimator = self._get_estimator(provider_name)
            system_prompt = self._build_system_prompt()
            messages = self._to_llm_messages(list(self._ctx_window))
            request = UnifiedLLMRequest(
                system_prompt=system_prompt,
                messages=messages,
                tool_schemas=self._tool_schemas if self._tool_schemas else None,
            )
        return estimator.estimate(request)

    def set_context_pressure_callback(
        self,
        callback: Callable[[float], None],
        threshold: float = 0.85,
    ) -> None:
        """Register a callback invoked when token usage exceeds threshold (0.0-1.0)."""
        with self._lock:
            self._pressure_callback = callback
            self._pressure_threshold = threshold

    # ------------------------------------------------------------------
    # Streaming message support
    # ------------------------------------------------------------------

    def begin_streaming_message(
        self,
        role: LLMRole,
        tool_use: ToolUseMetadata | None = None,
        tool_result: ToolResultMetadata | None = None,
    ) -> str:
        """Start a streaming message. Returns a stream_id to pass to subsequent calls."""
        stream_id = str(uuid4())
        with self._lock:
            self._streaming_buffers[stream_id] = []
            self._streaming_roles[stream_id] = role
            self._streaming_metadata[stream_id] = {
                "tool_use": tool_use,
                "tool_result": tool_result,
            }
        return stream_id

    def append_streaming_chunk(self, stream_id: str, chunk: str) -> None:
        """Append a text chunk to an in-progress streaming message."""
        with self._lock:
            if stream_id in self._streaming_buffers:
                self._streaming_buffers[stream_id].append(chunk)

    def end_streaming_message(self, stream_id: str) -> str | None:
        """Finalize a streaming message and commit it to ctx_window and history.

        Returns the committed message UUID, or None if stream_id is unknown.
        """
        with self._lock:
            if stream_id not in self._streaming_buffers:
                return None
            content = "".join(self._streaming_buffers.pop(stream_id))
            role = self._streaming_roles.pop(stream_id)
            extra = self._streaming_metadata.pop(stream_id)

        return self.add_message(
            role=role,
            content=content,
            tool_use=extra["tool_use"],
            tool_result=extra["tool_result"],
        )

    # ------------------------------------------------------------------
    # Reset / release
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear ctx_window and stage tracking. Preserves history and config."""
        with self._lock:
            self._ctx_window.clear()
            self._stage_records.clear()
            self._message_id_to_stage.clear()
            self._active_stage_index = None
            self._last_success_stage_index = None
            self._current_token_count = 0
            self._streaming_buffers.clear()
            self._streaming_roles.clear()
            self._streaming_metadata.clear()

    def release(self) -> None:
        """Full teardown: clear everything."""
        with self._lock:
            self._system_prompt = ""
            self._tool_schemas = []
            self._knowledge_entries = []
            self._user_preferences_entries = []
            self._variables = {}
            self._ctx_window.clear()
            self._history.clear()
            self._stage_records.clear()
            self._message_id_to_stage.clear()
            self._active_stage_index = None
            self._last_success_stage_index = None
            self._current_token_count = 0
            self._pressure_callback = None
            self._streaming_buffers.clear()
            self._streaming_roles.clear()
            self._streaming_metadata.clear()
            self._token_truncator = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_stage_summary(self, stage_index: int) -> None:
        """Call LLM to summarize a completed stage and replace its messages."""
        if self._llm_gateway is None:
            return
        stage_messages = self.get_stage_messages(stage_index)
        if not stage_messages:
            return
        try:
            summary_provider = (
                self._config.get("llm.summary_provider", "deepseek")
                if self._config else "deepseek"
            )
            history_text = "\n".join(
                f"[{m.role}] {m.content}" for m in stage_messages
            )
            request = UnifiedLLMRequest(
                system_prompt=(
                    "You are a context compressor for an AI agent. "
                    "Summarize the following stage execution into a concise paragraph. "
                    "Preserve: key decisions made, tools used and their outcomes, "
                    "important findings, and the final result of the stage. "
                    "Output only the summary text, no preamble or labels."
                ),
                messages=[LLMMessage(role="user", content=history_text)],
                tool_schemas=[],
            )
            response = self._llm_gateway.generate(request, summary_provider)
            self.summarize_stage(stage_index, response.assistant_message.content)
        except Exception:
            pass

    def _build_system_prompt(self) -> str:
        """Assemble the full system prompt from base + task context + knowledge + preferences."""
        parts: list[str] = []

        if self._system_prompt:
            parts.append(self._system_prompt)

        task_lines: list[str] = ["## Task Context"]
        task_lines.append(f"**Objective:** {self._task.description}")
        if self._task.intent:
            task_lines.append(f"**User Intent:** {self._task.intent}")
        if self._task.output_constraints:
            task_lines.append(f"**Output Constraints:** {self._task.output_constraints}")
        parts.append("\n".join(task_lines))

        if self._knowledge_entries:
            lines: list[str] = [
                "## Domain Knowledge",
                "The following entries are relevant to this task. "
                "Treat them as authoritative references and avoid redundant searches "
                "for information already covered here.",
            ]
            for entry in self._knowledge_entries:
                tags_str = f" `[{', '.join(entry.tags)}]`" if entry.tags else ""
                lines.append(f"\n### {entry.title}{tags_str}")
                lines.append(entry.content)
            parts.append("\n".join(lines))

        if self._user_preferences_entries:
            lines = [
                "## User Preferences",
                "Strictly follow these preferences throughout the task. "
                "They reflect the user's working style, quality standards, and expectations. "
                "Violating them is considered a task failure.",
            ]
            for pref in self._user_preferences_entries:
                if pref.keywords:
                    keyword_str = ", ".join(pref.keywords)
                    lines.append(f"- **[{keyword_str}]** {pref.content}")
                else:
                    lines.append(f"- {pref.content}")
            parts.append("\n".join(lines))

        return "\n\n".join(p for p in parts if p)

    def _get_stage_message_ids(self, stage_index: int) -> set[str]:
        """Return the set of ctx_window message IDs belonging to stage_index."""
        return set(self._stage_records[stage_index].message_ids)

    def _get_estimator(self, provider_name: str) -> BaseTokenEstimator:
        # Always delegate to factory (factory has its own per-provider cache),
        # so switching providers mid-task gets the correct estimator.
        from agent.models.context.estimator.token_estimator import TokenEstimatorFactory
        return TokenEstimatorFactory.get_estimator(provider_name)

    def _get_truncator(self) -> ContextTruncator | None:
        if self._token_truncator is not None:
            return self._token_truncator
        if self._config is None:
            return None
        strategy_name = self._config.get("context_truncation.strategy", "react")
        from agent.models.context.budget.token_budget_manager import TokenBudgetManagerFactory
        from agent.models.context.truncation.token_truncation import TruncatorFactory
        budget_manager = TokenBudgetManagerFactory.create(strategy_name, self._config)
        self._token_truncator = TruncatorFactory.create(
            strategy_name,
            budget_manager,
            self._logger,
            self._config,
        )
        return self._token_truncator

    def _get_total_budget(self, provider_name: str) -> int:
        if self._config is None:
            return 32000
        return int(
            self._config.get(
                f"llm.provider_settings.{provider_name}.context_window", 32000
            )
        )

    def _check_pressure(self) -> float | None:
        """Return current usage ratio if it exceeds the threshold, else None.

        Must be called while holding self._lock.
        """
        if self._pressure_callback is None:
            return None
        total_budget = self._get_total_budget_no_config()
        if total_budget <= 0:
            return None
        ratio = self._current_token_count / total_budget
        if ratio >= self._pressure_threshold:
            return ratio
        return None

    def _get_total_budget_no_config(self) -> int:
        """Best-effort budget estimate without a provider name."""
        if self._config is None:
            return 32000
        return int(self._config.get("llm.default_context_window", 32000))

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        """Fast character-based token estimate used for add_message bookkeeping."""
        return max(1, int(len(text) / _CHARS_PER_TOKEN_FALLBACK))

    @classmethod
    def _repair_context(cls, messages: list[ContextMessage]) -> list[ContextMessage]:
        """Apply all structural repairs required by mainstream LLM provider APIs.

        Repairs applied in order:
        1. Drop leading tool-result messages (no preceding tool_use).
        2. Ensure the first message is from the user role.
        3. Repair orphaned tool_use / tool_result pairs.
        4. Drop trailing assistant tool_use messages with no following result.
        5. Merge consecutive same-role messages (some providers reject them).
        """
        msgs = list(messages)
        msgs = cls._drop_leading_tool_results(msgs)
        msgs = cls._ensure_first_message_is_user(msgs)
        msgs = cls._repair_tool_pairs(msgs)
        msgs = cls._drop_trailing_tool_use(msgs)
        msgs = cls._merge_consecutive_same_role(msgs)
        return msgs

    @classmethod
    def _drop_leading_tool_results(cls, messages: list[ContextMessage]) -> list[ContextMessage]:
        """Remove tool-result messages that appear before any tool_use message."""
        first_tool_use_idx: int | None = None
        for i, m in enumerate(messages):
            if m.tool_use is not None:
                first_tool_use_idx = i
                break

        result: list[ContextMessage] = []
        for i, m in enumerate(messages):
            if m.tool_result is not None:
                if first_tool_use_idx is None or i < first_tool_use_idx:
                    continue  # orphaned result before any tool_use — drop
            result.append(m)
        return result

    @classmethod
    def _ensure_first_message_is_user(cls, messages: list[ContextMessage]) -> list[ContextMessage]:
        """Drop leading non-user messages until the first user message is reached.

        All major providers (OpenAI, Anthropic, Gemini) require the conversation
        to start with a user turn.
        """
        for i, m in enumerate(messages):
            if m.role == "user":
                return messages[i:]
        return []

    @classmethod
    def _repair_tool_pairs(cls, messages: list[ContextMessage]) -> list[ContextMessage]:
        """Ensure every tool_use has a matching tool_result and vice-versa.

        OpenAI and Anthropic both enforce:
        - Every assistant message with tool_calls must be followed (eventually,
          before the next assistant turn) by a tool message for each call_id.
        - Every tool message must reference a call_id that appeared in the
          immediately preceding assistant tool_use block.

        Strategy:
        - Walk forward; collect the set of pending call_ids from each assistant
          tool_use message.
        - For each tool-result message, check its call_id is in the pending set.
          If not, drop it (orphaned result).
        - After processing all results for a tool_use block, if any call_ids
          remain unresolved, inject a synthetic error result for each so the
          provider never sees an open tool_use.
        """
        result: list[ContextMessage] = []
        pending: set[str] = set()

        for msg in messages:
            if msg.tool_use is not None:
                # New tool_use block — flush any still-pending from previous block.
                for call_id in list(pending):
                    result.append(cls._synthetic_tool_error(call_id))
                pending.clear()
                result.append(msg)
                for call_id in msg.tool_use.all_call_ids():
                    pending.add(call_id)

            elif msg.tool_result is not None:
                call_id = msg.tool_result.tool_call_id
                if call_id not in pending:
                    # Orphaned result — no matching tool_use in current block.
                    continue
                result.append(msg)
                pending.discard(call_id)

            else:
                # Plain user or assistant message — flush any open tool_use first.
                if pending and msg.role == "user":
                    for call_id in list(pending):
                        result.append(cls._synthetic_tool_error(call_id))
                    pending.clear()
                result.append(msg)

        # Flush remaining open tool_use at end of message list.
        for call_id in list(pending):
            result.append(cls._synthetic_tool_error(call_id))

        return result

    @classmethod
    def _drop_trailing_tool_use(cls, messages: list[ContextMessage]) -> list[ContextMessage]:
        """Remove assistant tool_use messages at the tail that have no results.

        After _repair_tool_pairs these should already be covered by synthetic
        errors, but this is a safety net for callers that skip the full pipeline.
        """
        while messages and messages[-1].tool_use is not None:
            messages = messages[:-1]
        return messages

    @classmethod
    def _merge_consecutive_same_role(cls, messages: list[ContextMessage]) -> list[ContextMessage]:
        """Merge back-to-back messages with the same role into one.

        Anthropic's API rejects consecutive user or assistant messages.
        Tool-result messages are exempt — they must stay as individual turns.
        """
        if not messages:
            return messages
        result: list[ContextMessage] = [messages[0]]
        for msg in messages[1:]:
            prev = result[-1]
            # Never merge tool-result or tool-use turns — they carry structured metadata.
            if (
                prev.role == msg.role
                and prev.tool_use is None
                and prev.tool_result is None
                and msg.tool_use is None
                and msg.tool_result is None
            ):
                merged = ContextMessage(
                    id=prev.id,
                    role=prev.role,
                    content=prev.content + "\n\n" + msg.content,
                    timestamp=prev.timestamp,
                    token_count=(prev.token_count or 0) + (msg.token_count or 0),
                )
                result[-1] = merged
            else:
                result.append(msg)
        return result

    @classmethod
    def _synthetic_tool_error(cls, call_id: str) -> ContextMessage:
        """Create a placeholder tool-result for an unresolved tool_use call_id."""
        return ContextMessage(
            id=str(uuid4()),
            role="tool",
            content="Tool execution did not complete.",
            token_count=8,
            tool_result=ToolResultMetadata(
                tool_call_id=call_id,
                tool_name="unknown",
                success=False,
            ),
        )

    @classmethod
    def _to_llm_messages(cls, messages: list[ContextMessage]) -> list[LLMMessage]:
        result: list[LLMMessage] = []
        for m in messages:
            metadata: dict[str, Any] = {}
            if m.tool_use is not None:
                # Emit the same format the provider serializers consume:
                # metadata["tool_calls"] = [{"name": ..., "llm_raw_tool_call_id": ..., "arguments": ...}]
                all_calls = [
                    {
                        "name": entry.tool_name,
                        "llm_raw_tool_call_id": entry.tool_call_id,
                        "arguments": entry.tool_arguments,
                    }
                    for entry in m.tool_use.all_calls()
                ]
                metadata["tool_calls"] = all_calls
                metadata["tool_calls_count"] = len(all_calls)
            elif m.tool_result is not None:
                # Emit the key the provider serializers look for.
                metadata["llm_raw_tool_call_id"] = m.tool_result.tool_call_id
                metadata["tool_name"] = m.tool_result.tool_name
                metadata["success"] = m.tool_result.success
            elif m.summary is not None:
                metadata["summarized"] = True
                metadata["stage_index"] = m.summary.stage_index
            result.append(LLMMessage(role=m.role, content=m.content, metadata=metadata))
        return result

    @staticmethod
    def _from_llm_message(message: LLMMessage) -> ContextMessage:
        token_count = max(1, int(len(message.content) / _CHARS_PER_TOKEN_FALLBACK))
        tool_use: ToolUseMetadata | None = None
        tool_result: ToolResultMetadata | None = None
        if message.role == "assistant":
            tool_calls: list[dict] = message.metadata.get("tool_calls", [])
            if tool_calls:
                primary = tool_calls[0]
                tool_use = ToolUseMetadata(
                    tool_call_id=primary["llm_raw_tool_call_id"],
                    tool_name=primary["name"],
                    tool_arguments=dict(primary.get("arguments", {})),
                    extra_calls=tuple(
                        ToolCallEntry(
                            tool_call_id=c["llm_raw_tool_call_id"],
                            tool_name=c["name"],
                            tool_arguments=dict(c.get("arguments", {})),
                        )
                        for c in tool_calls[1:]
                    ),
                )
        elif message.role == "tool":
            tool_result = ToolResultMetadata(
                tool_call_id=message.metadata.get("llm_raw_tool_call_id") or "",
                tool_name=message.metadata.get("tool_name", ""),
                success=message.metadata.get("success", True),
            )
        return ContextMessage(
            id=str(uuid4()),
            role=message.role,
            content=message.content,
            token_count=token_count,
            tool_use=tool_use,
            tool_result=tool_result,
        )
