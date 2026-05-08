from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from infra.observability.tracing.tracer import Tracer
from llm.llm_gateway import LLMGateway
from schemas import LLMMessage, UnifiedLLMRequest, LLMResponse
from schemas.task import KnowledgeEntry, Plan, Task, UserPreferenceEntry
from schemas.types import LLMRole
from tools.tool_registry import ToolRegistry
from utils.log.log import Logger

if TYPE_CHECKING:
    from agent.models.context.estimator.token_estimator import BaseTokenEstimator
    from agent.models.context.truncation.token_truncation import ContextTruncator
    from config import ConfigReader


@dataclass(frozen=True)
class ContextMessage:
    id: str
    role: LLMRole
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class StageRecord:
    stage_index: int
    first_message_id: str | None = None
    last_message_id: str | None = None
    summary: str | None = None
    dropped: bool = False


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
        self._knowledge_entries: list[KnowledgeEntry] = []
        self._user_preferences_entries: list[UserPreferenceEntry] = []
        self._variables: dict[str, Any] = {}

        self._ctx_window: list[ContextMessage] = []
        self._history: list[ContextMessage] = []

        self._stage_records: list[StageRecord] = []
        self._message_id_to_stage: dict[str, int] = {}
        self._active_stage_index: int | None = None
        self._last_success_stage_index: int | None = None

        self._token_estimator: BaseTokenEstimator = None
        self._token_truncator: ContextTruncator | None = None

        self._lock = threading.RLock()

        self._tool_schemas: list[dict] = []
        if tool_registry is not None:
            self._tool_schemas = tool_registry.get_tool_schemas()


    # ------------------------------------------------------------------
    # Basic getters
    # ------------------------------------------------------------------
    def set_task(self, task:Task) -> None:
        self._task = task

    def get_task(self) -> Task:
        return self._task

    def set_plan(self, plan:Plan) -> None:
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

    def begin_stage(self, stage_index: int) -> None:
        """Record the start of a new stage. The next add_message call will
        set first_message_id for this stage."""
        with self._lock:
            while len(self._stage_records) <= stage_index:
                self._stage_records.append(
                    StageRecord(stage_index=len(self._stage_records))
                )
            self._active_stage_index = stage_index

    def end_stage(self, stage_index: int, success: bool) -> None:
        """Mark the stage as complete. On success, triggers async LLM summarization."""
        with self._lock:
            if stage_index >= len(self._stage_records):
                return
            record = self._stage_records[stage_index]
            last_id = self._ctx_window[-1].id if self._ctx_window else None
            self._stage_records[stage_index] = StageRecord(
                stage_index=record.stage_index,
                first_message_id=record.first_message_id,
                last_message_id=last_id,
                summary=record.summary,
                dropped=record.dropped,
            )
            if self._active_stage_index == stage_index:
                self._active_stage_index = None
            if success:
                self._last_success_stage_index = stage_index

        # Generate summary outside the lock to avoid blocking during LLM call
        if success:
            self._generate_stage_summary(stage_index)

    def drop_stage(self, stage_index: int) -> None:
        """Remove all ctx_window messages for stage_index. History is unchanged."""
        with self._lock:
            if stage_index >= len(self._stage_records):
                return
            stage_msg_ids = self._get_stage_message_ids(stage_index)
            self._ctx_window = [m for m in self._ctx_window if m.id not in stage_msg_ids]
            record = self._stage_records[stage_index]
            self._stage_records[stage_index] = StageRecord(
                stage_index=record.stage_index,
                first_message_id=record.first_message_id,
                last_message_id=record.last_message_id,
                summary=record.summary,
                dropped=True,
            )

    def summarize_stage(self, stage_index: int, summary: str) -> None:
        """Replace stage messages in ctx_window with a single summary message."""
        with self._lock:
            if stage_index >= len(self._stage_records):
                return
            stage_msg_ids = self._get_stage_message_ids(stage_index)
            if not stage_msg_ids:
                return

            summary_msg = ContextMessage(
                id=str(uuid4()),
                role="assistant",
                content=summary,
                metadata={"summarized": True, "stage_index": stage_index},
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

            record = self._stage_records[stage_index]
            self._stage_records[stage_index] = StageRecord(
                stage_index=record.stage_index,
                first_message_id=record.first_message_id,
                last_message_id=record.last_message_id,
                summary=summary,
                dropped=record.dropped,
            )

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
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Append a message to ctx_window and history. Returns the message UUID."""
        with self._lock:
            msg = ContextMessage(
                id=str(uuid4()),
                role=role,
                content=content,
                metadata=dict(metadata) if metadata else {},
            )
            self._ctx_window.append(msg)
            self._history.append(msg)

            if self._active_stage_index is not None:
                idx = self._active_stage_index
                self._message_id_to_stage[msg.id] = idx
                record = self._stage_records[idx]
                if record.first_message_id is None:
                    self._stage_records[idx] = StageRecord(
                        stage_index=record.stage_index,
                        first_message_id=msg.id,
                        last_message_id=record.last_message_id,
                        summary=record.summary,
                        dropped=record.dropped,
                    )
            return msg.id

    def add_llm_response(self, response: LLMResponse) -> None:
        """Append the assistant message from an LLMResponse to context."""
        msg = response.assistant_message
        self.add_message(
            role=msg.role,
            content=msg.content,
            metadata=dict(msg.metadata),
        )

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

    # ------------------------------------------------------------------
    # Core: build LLMRequest
    # ------------------------------------------------------------------

    def get_context_window(self, provider_name: str) -> UnifiedLLMRequest:
        """Assemble, optionally truncate, and return the LLMRequest for the LLM."""
        with self._lock:
            system_prompt = self._build_system_prompt()
            repaired = self._repair_tool_pairs(list(self._ctx_window))

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
    # Reset / release
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear ctx_window and stage tracking. Preserves history and config."""
        with self._lock:
            self._ctx_window.clear()
            self._stage_records.clear()
            self._message_id_to_stage.clear()
            self._active_stage_index = None

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
            self._token_estimator = None
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
            # Summary is best-effort; a failure must not affect stage completion
            pass

    def _build_system_prompt(self) -> str:
        """Assemble the full system prompt from base + task context + knowledge + preferences."""
        parts: list[str] = []

        if self._system_prompt:
            parts.append(self._system_prompt)

        # Task and plan overview so the agent always knows the big picture
        task_lines: list[str] = ["## Task Context"]
        task_lines.append(f"**Objective:** {self._task.description}")
        if self._task.intent:
            task_lines.append(f"**User Intent:** {self._task.intent}")
        if self._task.output_constraints:
            task_lines.append(f"**Output Constraints:** {self._task.output_constraints}")
        parts.append("\n".join(task_lines))

        # Domain knowledge — authoritative references the agent should consult
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

        # User preferences — behavioral constraints that must be respected
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
        return {
            msg_id
            for msg_id, idx in self._message_id_to_stage.items()
            if idx == stage_index
        }

    def _get_estimator(self, provider_name: str) -> BaseTokenEstimator:
        if self._token_estimator is not None:
            return self._token_estimator
        from agent.models.context.estimator.token_estimator import TokenEstimatorFactory
        self._token_estimator = TokenEstimatorFactory.get_estimator(provider_name)
        return self._token_estimator

    def _get_truncator(self) -> ContextTruncator | None:
        if self._token_truncator is not None:
            return self._token_truncator
        if self._config is None:
            return None
        strategy_name = self._config.get("context_truncation.strategy", "react")
        from agent.models.context.budget.token_budget_manager import TokenBudgetManagerFactory
        from agent.models.context.truncation.token_truncation import TruncatorFactory
        budget_manager = TokenBudgetManagerFactory.create(strategy_name, self._config)
        logger = Logger.get_instance()
        self._token_truncator = TruncatorFactory.create(
            strategy_name,
            budget_manager,
            logger,
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

    @classmethod
    def _repair_tool_pairs(cls, messages: list[ContextMessage]) -> list[ContextMessage]:
        """Remove trailing assistant messages whose tool calls have no matching tool results."""
        repaired = list(messages)
        while repaired:
            last = repaired[-1]
            if last.role != "assistant" or not last.metadata.get("tool_calls"):
                break
            tool_call_ids = {
                tc.get("llm_raw_tool_call_id")
                for tc in last.metadata.get("tool_calls", [])
                if isinstance(tc, dict)
            }
            following_tool_ids = {
                m.metadata.get("llm_raw_tool_call_id")
                for m in repaired
                if m.role == "tool"
            }
            if tool_call_ids and not tool_call_ids.issubset(following_tool_ids):
                repaired.pop()
                continue
            break
        return repaired

    @classmethod
    def _to_llm_messages(cls, messages: list[ContextMessage]) -> list[LLMMessage]:
        return [
            LLMMessage(
                role=message.role,
                content=message.content,
                metadata=dict(message.metadata),
            )
            for message in messages
        ]

    @staticmethod
    def _from_llm_message(message: LLMMessage) -> ContextMessage:
        return ContextMessage(
            id=str(uuid4()),
            role=message.role,
            content=message.content,
            metadata=dict(message.metadata),
        )
