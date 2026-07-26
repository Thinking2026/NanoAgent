from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.models.context.context_manager import ContextManager
from agent.models.executor.loop_state import StageStartReason
from agent.models.executor.stage_executor import StageExecutor, _StageOutcome, _StageResult
from agent.models.model_routing.circuit_breaker import CircuitState
from agent.models.reasoning.decision import NextDecision, NextDecisionType
from schemas.errors import LLMNormalizedError, LLMNormalizedErrorCode, CallerAction, PipelineError
from schemas.ids import PlanStepId, TaskId
from schemas.task import StageRecoveryAction
from schemas.types import ContextWindow, LLMMessage, ToolCall, ToolResult, UserCommandType


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
        self.calls: list[dict[str, object]] = []

    def reason_once(
        self,
        context_window: ContextWindow,
        provider_name: str,
        json_mode: bool = False,
        json_required_keys: list[str] | None = None,
    ) -> NextDecision:
        self.calls.append({
            "provider_name": provider_name,
            "json_mode": json_mode,
            "json_required_keys": json_required_keys,
        })
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

    reasoning_manager = FakeReasoningManager(decisions)
    executor = StageExecutor(
        config=config,
        logger=MagicMock(),
        tracer=tracer,
        reasoning_manager=reasoning_manager,
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
    executor._fake_reasoning_manager = reasoning_manager
    executor._driver = MagicMock()
    executor._driver.loop_user_messages.return_value = None
    return executor


# ---------------------------------------------------------------------------
# _normalize_stage_output
# ---------------------------------------------------------------------------

class TestNormalizeStageOutput:

    def _make_executor(self) -> StageExecutor:
        return object.__new__(StageExecutor)

    def test_unwraps_single_json_code_fence(self):
        executor = self._make_executor()
        result = executor._normalize_stage_output("""```json
{"items": [1, 2]}
```""")
        assert result == '{"items": [1, 2]}'

    def test_preserves_markdown_with_embedded_json_example(self):
        executor = self._make_executor()
        markdown = """# Report

Use this payload:

```json
{"items": [1, 2]}
```

Then continue with the explanation.
"""
        assert executor._normalize_stage_output(markdown) == markdown.strip()

    def test_preserves_markdown_header(self):
        executor = self._make_executor()
        markdown = """# Final Result

The answer is complete.
"""
        assert executor._normalize_stage_output(markdown) == markdown.strip()


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

    def test_reasoning_uses_json_mode(self):
        executor = make_executor([
            NextDecision(decision_type=NextDecisionType.FINAL_ANSWER, answer='{"result": "done"}', tool_calls=[]),
        ])
        stage = self._make_stage(executor)
        result = executor._execute_stage(stage, "p1")
        assert result.outcome == _StageOutcome.SUCCESS
        assert executor._fake_reasoning_manager.calls[0]["json_mode"] is True

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


# ---------------------------------------------------------------------------
# Stage level loop control flow
# ---------------------------------------------------------------------------

def _plan_with_steps(count: int):
    from schemas.task import Plan, PlanStep
    from schemas.ids import PlanId
    import uuid
    steps = [
        PlanStep(
            id=PlanStepId(f"step-{i + 1}"),
            order=i + 1,
            goal=f"goal {i + 1}",
            description="d",
        )
        for i in range(count)
    ]
    return Plan(id=PlanId(str(uuid.uuid4())), task_id=TaskId("task-1"), step_list=steps)


def _final_answer(text: str = "done") -> NextDecision:
    return NextDecision(decision_type=NextDecisionType.FINAL_ANSWER, answer=text, tool_calls=[])


def _eval(passed: bool, action=None, feedback: str = "fb"):
    return MagicMock(passed=passed, recovery_action=action, feedback=feedback)


class TestStageLevelLoopFlow:

    def test_last_step_result_is_returned_without_evaluation(self):
        # The final deliverable is reviewed by Pipeline, not by the stage evaluator.
        executor = make_executor([_final_answer("final")])
        result = executor.execute(plan=_plan_with_steps(1))
        assert result == "final"
        executor._quality_evaluator.evaluate_stage_result.assert_not_called()

    def test_walks_every_step_in_order(self):
        executor = make_executor([_final_answer("s1"), _final_answer("s2"), _final_answer("s3")])
        executor._quality_evaluator.evaluate_stage_result.return_value = _eval(True)
        assert executor.execute(plan=_plan_with_steps(3)) == "s3"
        # Steps 1 and 2 are evaluated; the last one is not.
        assert executor._quality_evaluator.evaluate_stage_result.call_count == 2

    def test_start_step_index_skips_completed_steps(self):
        executor = make_executor([_final_answer("s2")])
        result = executor.execute(plan=_plan_with_steps(2), start_step_index=1)
        assert result == "s2"

    def test_checkpoint_callback_fires_per_passed_non_final_stage(self):
        executor = make_executor([_final_answer("s1"), _final_answer("s2")])
        executor._quality_evaluator.evaluate_stage_result.return_value = _eval(True)
        saved: list[int] = []
        executor.set_stage_success_callback(saved.append)
        executor.execute(plan=_plan_with_steps(2))
        assert saved == [0]

    def test_cancel_returns_none_and_publishes_failure(self):
        executor = make_executor([_final_answer()])
        cancel = SimpleNamespace(type=UserCommandType.CANCEL, content="")
        executor._driver.loop_user_messages.return_value = cancel

        assert executor.execute(plan=_plan_with_steps(1)) is None
        published = [type(c.args[0]).__name__ for c in executor._event_bus.publish.call_args_list]
        assert "StageExecutionFailed" in published

    def test_retry_same_step_reruns_the_same_index(self):
        # step 1 fails eval → retried → passes → step 2 (last) delivers.
        executor = make_executor([
            _final_answer("bad"), _final_answer("retried"), _final_answer("last"),
        ])
        executor._quality_evaluator.evaluate_stage_result.side_effect = [
            _eval(False, StageRecoveryAction.RETRY_SAME_STEP),
            _eval(True),
        ]
        assert executor.execute(plan=_plan_with_steps(2)) == "last"
        executor._context_manager.drop_latest_stage_context.assert_called()
        assert executor._planner.renew_plan_step.call_count == 0

    def test_replan_this_step_asks_planner_for_a_new_step(self):
        executor = make_executor([
            _final_answer("bad"), _final_answer("retried"), _final_answer("last"),
        ])
        executor._quality_evaluator.evaluate_stage_result.side_effect = [
            _eval(False, StageRecoveryAction.REPLAN_THIS_STEP),
            _eval(True),
        ]
        from schemas.task import PlanStep
        executor._planner.renew_plan_step.return_value = PlanStep(
            id=PlanStepId("step-1b"), order=1, goal="revised", description="d")
        assert executor.execute(plan=_plan_with_steps(2)) == "last"
        executor._planner.renew_plan_step.assert_called_once()
        executor._context_manager.set_plan.assert_called()

    def test_per_step_replan_limit_raises(self):
        executor = make_executor([_final_answer("bad")] * 10)
        executor._max_replan_stage_retries = 1
        executor._quality_evaluator.evaluate_stage_result.return_value = _eval(
            False, StageRecoveryAction.REPLAN_THIS_STEP)
        from schemas.task import PlanStep
        executor._planner.renew_plan_step.return_value = PlanStep(
            id=PlanStepId("step-1b"), order=1, goal="revised", description="d")

        with pytest.raises(PipelineError):
            executor.execute(plan=_plan_with_steps(2))

    def test_exhausting_the_plan_raises_max_iterations(self):
        # An empty plan has no step to produce a result from.
        executor = make_executor([])
        with pytest.raises(PipelineError):
            executor.execute(plan=_plan_with_steps(0))


class TestReasoningLoopHandlers:

    def _stage(self):
        from schemas.types import Stage
        from schemas.ids import StageId
        import uuid
        return Stage(
            id=StageId(str(uuid.uuid4())),
            task_id=TaskId("task-1"),
            plan_step_id=PlanStepId("step-1"),
            order=1,
            goal="g",
            description="d",
        )

    def test_continue_then_final_answer(self):
        executor = make_executor([
            NextDecision(decision_type=NextDecisionType.CONTINUE, message="thinking", tool_calls=[]),
            _final_answer("done"),
        ])
        stage = self._stage()
        result = executor._execute_stage(stage, "p1")
        assert result.outcome == _StageOutcome.SUCCESS
        assert stage.iteration_count == 2

    def test_tool_call_is_dispatched_then_loop_continues(self):
        executor = make_executor([
            NextDecision(
                decision_type=NextDecisionType.TOOL_CALL,
                tool_calls=[ToolCall(name="shell", arguments={}, llm_raw_tool_call_id="c1")],
            ),
            _final_answer("done"),
        ])
        stage = self._stage()
        result = executor._execute_stage(stage, "p1")
        assert result.outcome == _StageOutcome.SUCCESS
        assert executor._tool_registry.calls[0].name == "shell"

    def test_consecutive_tool_call_nudge_is_injected(self):
        executor = make_executor(
            [NextDecision(
                decision_type=NextDecisionType.TOOL_CALL,
                tool_calls=[ToolCall(name="shell", arguments={}, llm_raw_tool_call_id="c1")],
            )] * 3 + [_final_answer("done")]
        )
        executor._max_tool_consecutive_count = 2
        stage = self._stage()
        executor._execute_stage(stage, "p1")

        nudges = [
            c for c in executor._context_manager.add_message.call_args_list
            if c.args[0] == "user" and "consecutive tool calls" in str(c.args[1])
        ]
        assert len(nudges) == 1

    def test_guidance_is_injected_without_ending_the_stage(self):
        executor = make_executor([_final_answer("done")])
        guidance = SimpleNamespace(type=UserCommandType.GUIDANCE, content=" focus on X ")
        executor._driver.loop_user_messages.side_effect = [guidance, None]

        result = executor._execute_stage(self._stage(), "p1")
        assert result.outcome == _StageOutcome.SUCCESS
        injected = [
            c.args[1] for c in executor._context_manager.add_message.call_args_list
            if c.args[0] == "user"
        ]
        assert "focus on X" in injected

    def test_clarification_falls_back_to_default_when_user_is_silent(self):
        from schemas.types import DEFAULT_CLARIFICATION
        executor = make_executor([
            NextDecision(decision_type=NextDecisionType.CLARIFICATION_NEEDED,
                         message="which one?", tool_calls=[]),
            _final_answer("done"),
        ])
        result = executor._execute_stage(self._stage(), "p1")
        assert result.outcome == _StageOutcome.SUCCESS
        injected = [str(c.args[1]) for c in executor._context_manager.add_message.call_args_list]
        assert any(DEFAULT_CLARIFICATION in text for text in injected)

    def test_max_iterations_requests_a_model_switch(self):
        executor = make_executor(
            [NextDecision(decision_type=NextDecisionType.CONTINUE, message="loop", tool_calls=[])] * 5
        )
        stage = self._stage()
        result = executor._execute_stage(stage, "p1")
        assert result.outcome == _StageOutcome.SWITCH_MODEL
        assert result.llm_error is None
        assert stage.iteration_count == 5

    def test_unknown_decision_type_does_not_crash_the_loop(self):
        # A decision type without a handler is skipped; the loop still terminates
        # on the iteration ceiling rather than raising.
        executor = make_executor([SimpleNamespace(
            decision_type="SOMETHING_NEW",
            tool_calls=[],
            assistant_message=None,
            message="",
            answer="",
        )] * 20)
        stage = self._stage()
        stage.iteration_count = 4  # one iteration left
        result = executor._execute_stage(stage, "p1")
        assert result.outcome == _StageOutcome.SWITCH_MODEL
        # The fallback handler consumed the iteration instead of spinning.
        assert stage.iteration_count == 5
