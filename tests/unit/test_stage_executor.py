from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.models.context.context_manager import ContextManager
from agent.models.executor.stage_executor import StageExecutor, _StageOutcome, _StageResult
from agent.models.model_routing.circuit_breaker import CircuitState
from agent.models.reasoning.decision import NextDecision, NextDecisionType
from schemas.errors import LLMNormalizedError, LLMNormalizedErrorCode, CallerAction
from schemas.ids import PlanStepId, TaskId
from schemas.types import ContextWindow, LLMMessage, ToolCall, ToolResult


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

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
        context_window: ContextWindow,
        provider_name: str,
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

    def get_tool_schemas_for(self, names: list[str]) -> list[dict]:
        return []

    def has_tool(self, name: str) -> bool:
        return True

    def validate_arguments(self, tool_call: ToolCall) -> list[str]:
        return []


class FakeKnowledgeLoader:
    def load(self, query: str) -> list[object]:
        return [SimpleNamespace(content=f"knowledge for {query}", tags=[])]


class FakeModelSelector:
    def __init__(self) -> None:
        self.failure_calls: list[tuple[str, object]] = []
        self.success_calls: list[str] = []
        self._next_provider: str | None = None
        self._recovered_provider: str | None = None
        self._current_provider = "p1"

    def get_current_provider(self) -> str:
        return self._current_provider

    def advance_provider(self, error: object) -> str:
        self.failure_calls.append((self._current_provider, error))
        self._current_provider = self._next_provider or "p2"
        return self._current_provider

    def confirm_provider_success(self) -> None:
        self.success_calls.append(self._current_provider)

    def get_next_available_provider(self, chain: list[str], current: str) -> str | None:
        return self._next_provider

    def get_best_recovered_provider(self, chain: list[str], current: str) -> str | None:
        return self._recovered_provider


def _make_context_manager() -> ContextManager:
    cm = MagicMock(spec=ContextManager)
    cm.get_context_window.return_value = ContextWindow(messages=[], tool_schemas=None)
    return cm


def make_executor(
    decisions: list[NextDecision],
    tool_result: ToolResult | None = None,
    model_selector: FakeModelSelector | None = None,
) -> StageExecutor:
    config = MagicMock()
    config.get = lambda key, default=None: {
        "agent.max_attempt_iterations": 5,
        "agent.max_replan_stage_retries": 3,
        "agent.max_stage_retries": 2,
        "tools.forbidden_tools": [],
    }.get(key, default)
    config.positive_float = lambda key, default: default

    tracer = MagicMock()
    tracer.start_span.return_value.__enter__ = MagicMock(return_value=MagicMock())
    tracer.start_span.return_value.__exit__ = MagicMock(return_value=False)

    executor = StageExecutor(
        config=config,
        logger=MagicMock(),
        tracer=tracer,
        reasoning_manager=FakeReasoningManager(decisions),
        context_manager=_make_context_manager(),
        quality_evaluator=MagicMock(),
        knowledge_loader=FakeKnowledgeLoader(),
        planner=MagicMock(),
        llm_gateway=MagicMock(),
        event_bus=MagicMock(),
        model_selector=model_selector or FakeModelSelector(),
        tool_registry=FakeToolRegistry(tool_result or ToolResult(output="ok", llm_raw_tool_call_id="id1")),
        renderer=MagicMock(),
    )
    executor._driver = MagicMock()
    executor._driver.loop_user_messages.return_value = None
    return executor


# ---------------------------------------------------------------------------
# _execute_stage returns _StageResult
# ---------------------------------------------------------------------------

class TestStageResultEncapsulation:

    def _make_stage(self, executor: StageExecutor):
        from agent.models.executor.stage_executor import Stage
        from schemas.ids import StageId
        import uuid
        return Stage(
            id=StageId(str(uuid.uuid4())),
            task_id=TaskId("task-1"),
            plan_step_id=PlanStepId("step-1"),
            order=1,
            goal="test goal",
            description="test description",
        )

    def test_returns_stage_result_on_success(self):
        executor = make_executor([
            NextDecision(decision_type=NextDecisionType.FINAL_ANSWER, answer="done", tool_calls=[]),
        ])
        stage = self._make_stage(executor)
        result = executor._execute_stage(stage, "p1")
        assert isinstance(result, _StageResult)
        assert result.outcome == _StageOutcome.SUCCESS
        assert result.llm_error is None

    def test_returns_stage_result_with_llm_error_on_switch_model(self):
        exc = LLMNormalizedError(LLMNormalizedErrorCode.HTTP_5XX, "server error")
        reasoning = MagicMock()
        reasoning.reason_once.side_effect = exc
        reasoning.format_tool_observation = FakeStrategy().format_tool_observation

        config = MagicMock()
        config.get = lambda key, default=None: {
            "agent.max_attempt_iterations": 5,
            "agent.max_replan_stage_retries": 3,
            "agent.max_stage_retries": 2,
            "tools.forbidden_tools": [],
        }.get(key, default)
        config.positive_float = lambda key, default: default
        tracer = MagicMock()
        tracer.start_span.return_value.__enter__ = MagicMock(return_value=MagicMock())
        tracer.start_span.return_value.__exit__ = MagicMock(return_value=False)

        executor = StageExecutor(
            config=config,
            logger=MagicMock(),
            tracer=tracer,
            reasoning_manager=reasoning,
            context_manager=_make_context_manager(),
            quality_evaluator=MagicMock(),
            knowledge_loader=FakeKnowledgeLoader(),
            planner=MagicMock(),
            llm_gateway=MagicMock(),
            event_bus=MagicMock(),
            model_selector=FakeModelSelector(),
            tool_registry=FakeToolRegistry(ToolResult(output="ok", llm_raw_tool_call_id="id1")),
            renderer=MagicMock(),
        )
        executor._driver = MagicMock()
        executor._driver.loop_user_messages.return_value = None

        stage = self._make_stage(executor)
        result = executor._execute_stage(stage, "p1")
        assert result.outcome == _StageOutcome.SWITCH_MODEL
        assert result.llm_error is exc

    def test_returns_fatal_on_fatal_llm_error(self):
        exc = LLMNormalizedError(LLMNormalizedErrorCode.CONFIG_ERROR, "bad config")
        reasoning = MagicMock()
        reasoning.reason_once.side_effect = exc
        reasoning.format_tool_observation = FakeStrategy().format_tool_observation

        config = MagicMock()
        config.get = lambda key, default=None: {
            "agent.max_attempt_iterations": 5,
            "agent.max_replan_stage_retries": 3,
            "agent.max_stage_retries": 2,
            "tools.forbidden_tools": [],
        }.get(key, default)
        config.positive_float = lambda key, default: default
        tracer = MagicMock()
        tracer.start_span.return_value.__enter__ = MagicMock(return_value=MagicMock())
        tracer.start_span.return_value.__exit__ = MagicMock(return_value=False)

        executor = StageExecutor(
            config=config,
            logger=MagicMock(),
            tracer=tracer,
            reasoning_manager=reasoning,
            context_manager=_make_context_manager(),
            quality_evaluator=MagicMock(),
            knowledge_loader=FakeKnowledgeLoader(),
            planner=MagicMock(),
            llm_gateway=MagicMock(),
            event_bus=MagicMock(),
            model_selector=FakeModelSelector(),
            tool_registry=FakeToolRegistry(ToolResult(output="ok", llm_raw_tool_call_id="id1")),
            renderer=MagicMock(),
        )
        executor._driver = MagicMock()
        executor._driver.loop_user_messages.return_value = None

        stage = self._make_stage(executor)
        result = executor._execute_stage(stage, "p1")
        assert result.outcome == _StageOutcome.FATAL
        assert result.llm_error is None


# ---------------------------------------------------------------------------
# ModelSelector integration in execute()
# ---------------------------------------------------------------------------

class TestExecuteModelSelectorIntegration:

    def _make_plan(self, executor: StageExecutor):
        from agent.models.executor.stage_executor import _replace_step
        from schemas.task import Plan, PlanStep
        from schemas.ids import PlanId
        import uuid
        step = PlanStep(
            id=PlanStepId("step-1"),
            order=1,
            goal="test goal",
            description="test description",
            key_results=[],
            inputs=[],
            required_tools=[],
            action_constraints=[],
            risks=[],
            dependencies=[],
            execution_notes="",
            output_constraints="",
        )
        return Plan(
            id=PlanId(str(uuid.uuid4())),
            task_id=TaskId("task-1"),
            step_list=[step],
        )

    def test_success_calls_record_provider_success(self):
        sel = FakeModelSelector()
        executor = make_executor(
            decisions=[
                NextDecision(decision_type=NextDecisionType.FINAL_ANSWER, answer="done", tool_calls=[]),
            ],
            model_selector=sel,
        )
        executor._quality_evaluator.evaluate_stage_result.return_value = MagicMock(
            passed=True, recovery_action=None, feedback=""
        )
        plan = self._make_plan(executor)
        executor.execute(plan=plan)
        assert "p1" in sel.success_calls

    def test_switch_model_calls_record_provider_failure(self):
        sel = FakeModelSelector()
        sel._next_provider = "p2"
        exc = LLMNormalizedError(LLMNormalizedErrorCode.HTTP_5XX, "error")
        reasoning = MagicMock()
        # first call raises, second succeeds
        reasoning.reason_once.side_effect = [
            exc,
            NextDecision(decision_type=NextDecisionType.FINAL_ANSWER, answer="done", tool_calls=[]),
        ]
        reasoning.format_tool_observation = FakeStrategy().format_tool_observation

        config = MagicMock()
        config.get = lambda key, default=None: {
            "agent.max_attempt_iterations": 5,
            "agent.max_replan_stage_retries": 3,
            "agent.max_stage_retries": 2,
            "tools.forbidden_tools": [],
        }.get(key, default)
        config.positive_float = lambda key, default: default
        tracer = MagicMock()
        tracer.start_span.return_value.__enter__ = MagicMock(return_value=MagicMock())
        tracer.start_span.return_value.__exit__ = MagicMock(return_value=False)

        executor = StageExecutor(
            config=config,
            logger=MagicMock(),
            tracer=tracer,
            reasoning_manager=reasoning,
            context_manager=_make_context_manager(),
            quality_evaluator=MagicMock(),
            knowledge_loader=FakeKnowledgeLoader(),
            planner=MagicMock(),
            llm_gateway=MagicMock(),
            event_bus=MagicMock(),
            model_selector=sel,
            tool_registry=FakeToolRegistry(ToolResult(output="ok", llm_raw_tool_call_id="id1")),
            renderer=MagicMock(),
        )
        executor._driver = MagicMock()
        executor._driver.loop_user_messages.return_value = None
        executor._quality_evaluator.evaluate_stage_result.return_value = MagicMock(
            passed=True, recovery_action=None, feedback=""
        )

        plan = self._make_plan(executor)
        executor.execute(plan=plan)
        assert len(sel.failure_calls) == 1
        assert sel.failure_calls[0][0] == "p1"
        assert sel.failure_calls[0][1] is exc
