from __future__ import annotations

"""Tool call dispatch for the reasoning loop.

Pre-check → execute → format observation → inject into context → publish events.
Split out of StageExecutor so the reasoning loop only has to call dispatch().
"""

from typing import TYPE_CHECKING

from agent.events.events import ToolCallResultProduced, ToolCallStarted
from agent.models.context.context_manager import (
    ToolCallEntry,
    ToolResultMetadata,
    ToolUseMetadata,
)
from schemas.errors import (
    TOOL_ARGUMENT_ERROR,
    TOOL_NOT_FOUND,
    build_pipeline_error,
)
from schemas.event_bus import EventBus
from schemas.types import LLMMessage, Stage, ToolCall, ToolResult
from tools.tool_registry import ToolRegistry
from utils.log.log import Logger

if TYPE_CHECKING:
    from agent.models.context.context_manager import ContextManager
    from agent.models.reasoning.reasoning_manager import ReasoningManager


class ToolDispatcher:
    """Executes the tool calls of one TOOL_CALL decision."""

    def __init__(
        self,
        logger: Logger,
        tool_registry: ToolRegistry,
        reasoning_manager: ReasoningManager,
        context_manager: ContextManager,
        event_bus: EventBus,
        forbidden_tools: frozenset[str] = frozenset(),
    ) -> None:
        self._logger = logger
        self._tool_registry = tool_registry
        self._reasoning_manager = reasoning_manager
        self._context_manager = context_manager
        self._event_bus = event_bus
        self._forbidden_tools = forbidden_tools

    def dispatch(self, stage: Stage, tool_calls: list[ToolCall]) -> None:
        for tool_call in tool_calls:
            self._logger.info("Dispatching tool call",
                task_id=stage.task_id, stage_id=stage.id,
                iteration=stage.iteration_count, tool_name=tool_call.name,
                argument_keys=list(tool_call.arguments.keys()))
            self._event_bus.publish(
                ToolCallStarted.with_meta(
                    task_id=stage.task_id,
                    tool_name=tool_call.name,
                    arguments=dict(tool_call.arguments),
                    step_order=stage.order,
                )
            )

            rejection = self._check_tool_call(tool_call)
            if rejection is not None:
                self._record_rejection(stage, tool_call, rejection)
                continue

            result: ToolResult = self._tool_registry.execute(tool_call)
            self._record_result(stage, tool_call, result)

    # ------------------------------------------------------------------
    # Result recording
    # ------------------------------------------------------------------

    def _record_rejection(self, stage: Stage, tool_call: ToolCall, rejection: ToolResult) -> None:
        self._logger.warning("Tool call rejected before execution",
            task_id=stage.task_id, stage_id=stage.id,
            tool_name=tool_call.name,
            error_code=None if rejection.error is None else rejection.error.code,
            error_message=None if rejection.error is None else rejection.error.message)
        self._inject_observation(tool_call, rejection)
        self._event_bus.publish(
            ToolCallResultProduced.with_meta(
                task_id=stage.task_id,
                tool_name=tool_call.name,
                step_order=stage.order,
                success=False,
                result=f"← {tool_call.name}: ✗ pre-check failed",
            )
        )

    def _record_result(self, stage: Stage, tool_call: ToolCall, result: ToolResult) -> None:
        self._inject_observation(tool_call, self._tool_result_for_observation(result))
        self._event_bus.publish(
            ToolCallResultProduced.with_meta(
                task_id=stage.task_id,
                tool_name=tool_call.name,
                step_order=stage.order,
                success=result.success,
                result=f"← {tool_call.name}: {'✓' if result.success else '✗'} {(result.output or '')[:100]}",
            )
        )
        self._logger.info("Tool call result recorded",
            task_id=stage.task_id, stage_id=stage.id,
            tool_name=tool_call.name, success=result.success,
            error_code=None if result.error is None else result.error.code,
            output_length=len(result.output or ""))

    def _inject_observation(self, tool_call: ToolCall, result: ToolResult) -> None:
        observation = self._reasoning_manager.format_tool_observation(
            tool_call=tool_call,
            result=result,
        )
        self._context_manager.add_message(
            observation.role,
            observation.content,
            tool_result=build_tool_result_metadata(observation.metadata),
        )

    # ------------------------------------------------------------------
    # Pre-execution checks
    # ------------------------------------------------------------------

    def _check_tool_call(self, tool_call: ToolCall) -> ToolResult | None:
        if not self._tool_registry.has_tool(tool_call.name):
            available = ", ".join(s["name"] for s in self._tool_registry.get_tool_schemas())
            return self._reject(
                tool_call, TOOL_NOT_FOUND,
                f"Tool '{tool_call.name}' does not exist. Available: {available}.",
            )

        if self._forbidden_tools and tool_call.name in self._forbidden_tools:
            return self._reject(
                tool_call, TOOL_NOT_FOUND,
                f"Tool '{tool_call.name}' is forbidden.",
            )

        missing = self._tool_registry.validate_arguments(tool_call)
        if missing:
            return self._reject(
                tool_call, TOOL_ARGUMENT_ERROR,
                f"Tool '{tool_call.name}' missing required args: {', '.join(missing)}.",
            )

        return None

    @staticmethod
    def _reject(tool_call: ToolCall, code: str, message: str) -> ToolResult:
        return ToolResult(
            output="",
            llm_raw_tool_call_id=tool_call.llm_raw_tool_call_id,
            success=False,
            error=build_pipeline_error(code, message),
        )

    @staticmethod
    def _tool_result_for_observation(result: ToolResult) -> ToolResult:
        if result.success or result.output or result.error is None:
            return result
        return ToolResult(
            output=f"Tool call failed: [{result.error.code}] {result.error.message}",
            llm_raw_tool_call_id=result.llm_raw_tool_call_id,
            success=False,
            error=result.error,
        )


# ── Metadata conversion helpers ───────────────────────────────────────────────

_TOOL_METADATA_KEYS = frozenset({"tool_calls", "tool_calls_count"})


def build_tool_use_metadata(metadata: dict) -> ToolUseMetadata | None:
    """Convert LLM response metadata into a typed ToolUseMetadata."""
    calls: list[dict] = metadata.get("tool_calls", [])
    if not calls:
        return None
    primary = calls[0]
    extra = tuple(
        ToolCallEntry(
            tool_call_id=c["llm_raw_tool_call_id"],
            tool_name=c["name"],
            tool_arguments=dict(c.get("arguments", {})),
        )
        for c in calls[1:]
    )
    return ToolUseMetadata(
        tool_call_id=primary["llm_raw_tool_call_id"],
        tool_name=primary["name"],
        tool_arguments=dict(primary.get("arguments", {})),
        extra_calls=extra,
    )


def build_tool_result_metadata(observation_metadata: dict) -> ToolResultMetadata:
    """Convert format_tool_observation metadata into a typed ToolResultMetadata."""
    return ToolResultMetadata(
        tool_call_id=observation_metadata.get("llm_raw_tool_call_id") or "",
        tool_name=observation_metadata.get("tool_name", ""),
        success=observation_metadata.get("success", True),
    )


def non_tool_metadata(msg: LLMMessage | None) -> dict:
    """Return metadata from an assistant LLMMessage, excluding tool-call keys."""
    if not msg or not msg.metadata:
        return {}
    return {k: v for k, v in msg.metadata.items() if k not in _TOOL_METADATA_KEYS}
