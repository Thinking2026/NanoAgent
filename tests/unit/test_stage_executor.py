from __future__ import annotations

from types import SimpleNamespace

from agent.models.context.context_manager import ContextManager
from agent.models.executor.stage_executor import StageExecutor, StageStatus
from agent.models.reasoning.decision import NextDecision, NextDecisionType
from schemas.errors import TOOL_EXECUTION_ERROR, build_pipeline_error
from schemas.ids import PlanStepId, TaskId
from schemas.types import LLMMessage, ToolCall, ToolResult


class FakeStrategy:
    def format_tool_observation(
        self,
        tool_call: ToolCall,
        result: ToolResult,
    ) -> LLMMessage:
        return LLMMessage(
            role="tool",
            content=result.output,
            metadata={
                "tool_name": tool_call.name,
                "success": result.success,
                "llm_raw_tool_call_id": result.llm_raw_tool_call_id,
            },
        )


class FakeReasoningManager:
    def __init__(self, decisions: list[NextDecision]) -> None:
        self._decisions = list(decisions)
        self._strategy = FakeStrategy()

    def reason_once(
        self,
        context_manager: ContextManager,
        tool_registry: object,
        selected_tool_names: list[str] | None = None,
        provider_name: str | None = None,
    ) -> NextDecision:
        return self._decisions.pop(0)

    def set_llm_gateway(self, llm_gateway: object) -> None:
        return None

    def format_tool_observation(
        self,
        tool_call: ToolCall,
        result: ToolResult,
    ) -> LLMMessage:
        return self._strategy.format_tool_observation(tool_call, result)


class FakeToolRegistry:
    def __init__(self, result: ToolResult) -> None:
        self.result = result
        self.calls: list[ToolCall] = []

    def execute(self, tool_call: ToolCall) -> ToolResult:
        self.calls.append(tool_call)
        return self.result

    def get_tool_schemas(self) -> list[dict]:
        return []

    def has_tool(self, name: str) -> bool:
        return True

    def validate_arguments(self, tool_call: ToolCall) -> list[str]:
        return []


class FakeKnowledgeLoader:
    def load(self, query: str) -> list[object]:
        return [SimpleNamespace(content=f"knowledge for {query}")]


def make_executor(
    decisions: list[NextDecision],
    tool_result: ToolResult | None = None,
) -> StageExecutor:
    return StageExecutor(
        reasoning_manager=FakeReasoningManager(decisions),
        context_manager=ContextManager(),
        tool_registry=FakeToolRegistry(tool_result or ToolResult(output="ok")),
        quality_evaluator=object(),
        knowledge_loader=FakeKnowledgeLoader(),
        max_iterations=5,
    )