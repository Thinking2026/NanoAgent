from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.models.context.task_context_seeder import TaskContextSeeder
from agent.models.executor.loop_state import (
    ReactState,
    StageLoopState,
    StageOutcome,
    StageRecoveryResult,
    StageResult,
    StageStartReason,
    escalate_recovery_action,
)
from agent.models.executor.tool_dispatcher import (
    ToolDispatcher,
    build_tool_result_metadata,
    build_tool_use_metadata,
    non_tool_metadata,
)
from schemas.ids import PlanId, PlanStepId, TaskId
from schemas.task import (
    EvaluationReport,
    EvaluationTarget,
    Plan,
    PlanStep,
    StageRecoveryAction,
    Task,
    TaskStatus,
    ToolMatch,
)
from schemas.types import LLMMessage, StageId, Stage, ToolCall, ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _report(passed: bool, action: StageRecoveryAction | None = None,
            feedback: str = "fb") -> EvaluationReport:
    return EvaluationReport(
        target_type=EvaluationTarget.STAGE_RESULT,
        target_id="stage-1",
        passed=passed,
        feedback=feedback,
        evaluated_at=datetime(2026, 1, 1),
        recovery_action=action,
    )


def _step(order: int = 1) -> PlanStep:
    return PlanStep(
        id=PlanStepId(f"step-{order}"),
        order=order,
        goal=f"goal {order}",
        description="d",
    )


def _plan(step_count: int = 2) -> Plan:
    return Plan(
        id=PlanId("plan-1"),
        task_id=TaskId("task-1"),
        step_list=[_step(i + 1) for i in range(step_count)],
    )


def _task(matches: list[ToolMatch] | None = None) -> Task:
    return Task(
        id=TaskId("task-1"),
        user_id="user-1",
        description="do it",
        created_at=datetime(2026, 1, 1),
        status=TaskStatus.CREATED,
        tool_matches=matches or [],
    )


def _stage() -> Stage:
    return Stage(
        id=StageId("stage-1"),
        task_id=TaskId("task-1"),
        plan_step_id=PlanStepId("step-1"),
        order=1,
        goal="g",
        description="d",
    )


# ---------------------------------------------------------------------------
# escalate_recovery_action
# ---------------------------------------------------------------------------

class TestEscalation:

    @pytest.mark.parametrize("given,expected", [
        (StageRecoveryAction.RETRY_SAME_STEP, StageRecoveryAction.REPLAN_THIS_STEP),
        (StageRecoveryAction.REPLAN_THIS_STEP, StageRecoveryAction.REPLAN_FROM_HERE),
        (StageRecoveryAction.REPLAN_FROM_HERE, StageRecoveryAction.REPLAN_ALL),
        (StageRecoveryAction.REPLAN_ALL, StageRecoveryAction.REPLAN_ALL),
    ])
    def test_escalates_one_level(self, given, expected):
        assert escalate_recovery_action(given) == expected


# ---------------------------------------------------------------------------
# StageLoopState
# ---------------------------------------------------------------------------

class TestStageLoopStateActionResolution:

    def test_defaults_to_replan_this_step_when_no_action(self):
        state = StageLoopState()
        assert state.resolve_action(_report(False)) == StageRecoveryAction.REPLAN_THIS_STEP

    def test_first_retry_same_step_is_not_escalated(self):
        state = StageLoopState()
        action = state.resolve_action(_report(False, StageRecoveryAction.RETRY_SAME_STEP))
        assert action == StageRecoveryAction.RETRY_SAME_STEP
        assert state.same_failure_count == 1

    def test_second_consecutive_retry_same_step_escalates(self):
        state = StageLoopState()
        state.resolve_action(_report(False, StageRecoveryAction.RETRY_SAME_STEP))
        action = state.resolve_action(_report(False, StageRecoveryAction.RETRY_SAME_STEP))
        assert action == StageRecoveryAction.REPLAN_THIS_STEP

    def test_non_retry_action_resets_same_failure_count(self):
        state = StageLoopState()
        state.resolve_action(_report(False, StageRecoveryAction.RETRY_SAME_STEP))
        state.resolve_action(_report(False, StageRecoveryAction.REPLAN_FROM_HERE))
        assert state.same_failure_count == 0

    def test_escalation_keeps_counter_so_it_stays_escalated(self):
        # Once escalated the counter is intentionally not cleared, so a third
        # RETRY_SAME_STEP suggestion escalates again rather than falling back.
        state = StageLoopState()
        state.resolve_action(_report(False, StageRecoveryAction.RETRY_SAME_STEP))
        state.resolve_action(_report(False, StageRecoveryAction.RETRY_SAME_STEP))
        action = state.resolve_action(_report(False, StageRecoveryAction.RETRY_SAME_STEP))
        assert state.same_failure_count == 3
        assert action == StageRecoveryAction.REPLAN_THIS_STEP


class TestStageLoopStateCounters:

    def test_replan_all_resets_step_attempts_and_bumps_total(self):
        state = StageLoopState(replan_attempts=2)
        state.correction_feedback.append("old")
        state.on_recovery(StageRecoveryAction.REPLAN_ALL, "fresh")
        assert state.replan_attempts == 0
        assert state.total_replan_count == 1
        assert state.correction_feedback == []

    def test_cheaper_action_accumulates_attempts_and_feedback(self):
        state = StageLoopState()
        state.on_recovery(StageRecoveryAction.RETRY_SAME_STEP, "fb-1")
        state.on_recovery(StageRecoveryAction.REPLAN_THIS_STEP, "fb-2")
        assert state.replan_attempts == 2
        assert state.total_replan_count == 0
        assert state.correction_feedback == ["fb-1", "fb-2"]

    def test_stage_passed_clears_per_step_state(self):
        state = StageLoopState(replan_attempts=3, same_failure_count=2)
        state.correction_feedback.append("fb")
        state.on_stage_passed()
        assert state.replan_attempts == 0
        assert state.same_failure_count == 0
        assert state.correction_feedback == []

    def test_advance_moves_to_next_step_as_new(self):
        state = StageLoopState(step_index=1, start_reason=StageStartReason.EVAL_RETRY)
        state.advance()
        assert state.step_index == 2
        assert state.start_reason is StageStartReason.NEW

    def test_apply_recovery_adopts_plan_step_and_reason(self):
        state = StageLoopState(step_index=3)
        new_plan = _plan(1)
        returned = state.apply_recovery(
            StageRecoveryResult(new_plan, 0, StageStartReason.REPLAN_ALL, True)
        )
        assert returned is new_plan
        assert state.step_index == 0
        assert state.start_reason is StageStartReason.REPLAN_ALL

    def test_correction_feedback_list_is_shared_with_caller(self):
        # StageExecutor passes its own list in and reads it back after recovery.
        shared: list[str] = []
        state = StageLoopState(correction_feedback=shared)
        state.on_recovery(StageRecoveryAction.RETRY_SAME_STEP, "fb")
        assert shared == ["fb"]


# ---------------------------------------------------------------------------
# StageResult / ReactState
# ---------------------------------------------------------------------------

class TestStageResult:

    def test_defaults_to_no_llm_error(self):
        assert StageResult(outcome=StageOutcome.SUCCESS).llm_error is None

    def test_react_state_starts_at_zero(self):
        assert ReactState().tool_consecutive_count == 0


# ---------------------------------------------------------------------------
# ToolDispatcher
# ---------------------------------------------------------------------------

class _FakeRegistry:
    def __init__(self, *, has_tool: bool = True, missing: list[str] | None = None,
                 result: ToolResult | None = None) -> None:
        self._has_tool = has_tool
        self._missing = missing or []
        self._result = result or ToolResult(output="ok", llm_raw_tool_call_id="c1")
        self.executed: list[ToolCall] = []

    def has_tool(self, name: str) -> bool:
        return self._has_tool

    def validate_arguments(self, tool_call: ToolCall) -> list[str]:
        return self._missing

    def get_tool_schemas(self) -> list[dict]:
        return [{"name": "shell"}]

    def execute(self, tool_call: ToolCall) -> ToolResult:
        self.executed.append(tool_call)
        return self._result


class _FakeReasoningManager:
    def format_tool_observation(self, tool_call: ToolCall, result: ToolResult) -> LLMMessage:
        return LLMMessage(
            role="tool",
            content=result.output,
            metadata={
                "tool_name": tool_call.name,
                "success": result.success,
                "llm_raw_tool_call_id": result.llm_raw_tool_call_id,
            },
        )


def _dispatcher(registry: _FakeRegistry, forbidden: frozenset[str] = frozenset()):
    context_manager = MagicMock()
    event_bus = MagicMock()
    dispatcher = ToolDispatcher(
        logger=MagicMock(),
        tool_registry=registry,
        reasoning_manager=_FakeReasoningManager(),
        context_manager=context_manager,
        event_bus=event_bus,
        forbidden_tools=forbidden,
    )
    return dispatcher, context_manager, event_bus


class TestToolDispatcher:

    def test_executes_and_injects_observation(self):
        registry = _FakeRegistry()
        dispatcher, ctx, bus = _dispatcher(registry)
        call = ToolCall(name="shell", arguments={"cmd": "ls"}, llm_raw_tool_call_id="c1")

        dispatcher.dispatch(_stage(), [call])

        assert registry.executed == [call]
        assert ctx.add_message.call_count == 1
        # ToolCallStarted + ToolCallResultProduced
        assert bus.publish.call_count == 2

    def test_unknown_tool_is_rejected_without_execution(self):
        registry = _FakeRegistry(has_tool=False)
        dispatcher, ctx, _ = _dispatcher(registry)

        dispatcher.dispatch(_stage(), [
            ToolCall(name="nope", arguments={}, llm_raw_tool_call_id="c1")
        ])

        assert registry.executed == []
        # The rejection is still fed back to the model as an observation.
        assert ctx.add_message.call_count == 1

    def test_forbidden_tool_is_rejected(self):
        registry = _FakeRegistry()
        dispatcher, _, _ = _dispatcher(registry, forbidden=frozenset({"shell"}))

        dispatcher.dispatch(_stage(), [
            ToolCall(name="shell", arguments={}, llm_raw_tool_call_id="c1")
        ])

        assert registry.executed == []

    def test_missing_required_arguments_is_rejected(self):
        registry = _FakeRegistry(missing=["cmd"])
        dispatcher, _, _ = _dispatcher(registry)

        dispatcher.dispatch(_stage(), [
            ToolCall(name="shell", arguments={}, llm_raw_tool_call_id="c1")
        ])

        assert registry.executed == []

    def test_failed_result_without_output_gets_error_text(self):
        from schemas.errors import build_pipeline_error
        failed = ToolResult(
            output="",
            llm_raw_tool_call_id="c1",
            success=False,
            error=build_pipeline_error("TOOL_EXEC_ERROR", "boom"),
        )
        registry = _FakeRegistry(result=failed)
        dispatcher, ctx, _ = _dispatcher(registry)

        dispatcher.dispatch(_stage(), [
            ToolCall(name="shell", arguments={}, llm_raw_tool_call_id="c1")
        ])

        injected = ctx.add_message.call_args[0][1]
        assert "boom" in injected

    def test_every_call_in_a_batch_is_dispatched(self):
        registry = _FakeRegistry()
        dispatcher, _, _ = _dispatcher(registry)

        dispatcher.dispatch(_stage(), [
            ToolCall(name="a", arguments={}, llm_raw_tool_call_id="c1"),
            ToolCall(name="b", arguments={}, llm_raw_tool_call_id="c2"),
        ])

        assert [c.name for c in registry.executed] == ["a", "b"]


class TestMetadataHelpers:

    def test_build_tool_use_metadata_splits_primary_and_extras(self):
        meta = build_tool_use_metadata({"tool_calls": [
            {"llm_raw_tool_call_id": "c1", "name": "a", "arguments": {"x": 1}},
            {"llm_raw_tool_call_id": "c2", "name": "b", "arguments": {}},
        ]})
        assert meta.tool_call_id == "c1"
        assert meta.tool_name == "a"
        assert len(meta.extra_calls) == 1
        assert meta.extra_calls[0].tool_name == "b"

    def test_build_tool_use_metadata_returns_none_without_calls(self):
        assert build_tool_use_metadata({}) is None

    def test_build_tool_result_metadata_defaults_success_true(self):
        meta = build_tool_result_metadata({"tool_name": "shell"})
        assert meta.tool_call_id == ""
        assert meta.success is True

    def test_non_tool_metadata_strips_tool_keys(self):
        msg = LLMMessage(role="assistant", content="x", metadata={
            "tool_calls": [{"name": "a"}],
            "tool_calls_count": 1,
            "reasoning": "keep me",
        })
        assert non_tool_metadata(msg) == {"reasoning": "keep me"}

    def test_non_tool_metadata_handles_none(self):
        assert non_tool_metadata(None) == {}


# ---------------------------------------------------------------------------
# TaskContextSeeder
# ---------------------------------------------------------------------------

def _seeder(threshold: float = 0.65):
    config = MagicMock()
    config.get = lambda key, default=None: (
        threshold if key == "planner.tool_score_filter_threshold" else default
    )
    context_manager = MagicMock()
    tool_registry = MagicMock()
    tool_registry.get_tool_schemas_for.side_effect = lambda names: [
        {"name": n} for n in names
    ]
    renderer = MagicMock()
    renderer.render.return_value = "rendered\n"
    seeder = TaskContextSeeder(
        config=config,
        logger=MagicMock(),
        renderer=renderer,
        context_manager=context_manager,
        tool_registry=tool_registry,
    )
    return seeder, context_manager, tool_registry


class TestTaskContextSeederToolFilter:

    def test_keeps_tools_at_or_above_threshold(self):
        seeder, ctx, registry = _seeder()
        task = _task([
            ToolMatch(tool_name="keep", match_score=0.9),
            ToolMatch(tool_name="drop", match_score=0.2),
        ])
        assert seeder.apply_tool_filter(task) == ["keep"]
        registry.get_tool_schemas_for.assert_called_once_with(["keep"])
        ctx.set_tool_schemas.assert_called_once_with([{"name": "keep"}])

    def test_planner_score_can_rescue_a_low_analyzer_score(self):
        seeder, _, _ = _seeder()
        task = _task([ToolMatch(tool_name="t", match_score=0.1, planner_score=0.8)])
        assert seeder.apply_tool_filter(task) == ["t"]

    def test_boundary_score_is_kept(self):
        seeder, _, _ = _seeder(threshold=0.65)
        task = _task([ToolMatch(tool_name="t", match_score=0.65)])
        assert seeder.apply_tool_filter(task) == ["t"]

    def test_leaves_schemas_untouched_when_nothing_qualifies(self):
        # Never strip the agent down to zero tools.
        seeder, ctx, _ = _seeder()
        task = _task([ToolMatch(tool_name="t", match_score=0.1)])
        assert seeder.apply_tool_filter(task) == []
        ctx.set_tool_schemas.assert_not_called()

    def test_no_matches_at_all(self):
        seeder, ctx, _ = _seeder()
        assert seeder.apply_tool_filter(_task([])) == []
        ctx.set_tool_schemas.assert_not_called()


class TestTaskContextSeederSeeding:

    def test_seed_injects_task_message_and_plan_tool_pair(self):
        seeder, ctx, _ = _seeder()
        seeder.seed(_task(), _plan(), "do it")

        roles = [call.args[0] for call in ctx.add_message.call_args_list]
        assert roles == ["user", "assistant", "tool"]

    def test_plan_tool_call_ids_are_paired(self):
        seeder, ctx, _ = _seeder()
        seeder.add_plan_messages(_plan(), "do it")

        assistant_call, tool_call = ctx.add_message.call_args_list
        use_meta = assistant_call.kwargs["tool_use"]
        result_meta = tool_call.kwargs["tool_result"]
        assert use_meta.tool_call_id == result_meta.tool_call_id
        assert use_meta.tool_name == result_meta.tool_name == "make_plan"
        assert result_meta.success is True

    def test_each_seeding_uses_a_fresh_tool_call_id(self):
        seeder, ctx, _ = _seeder()
        seeder.add_plan_messages(_plan(), "do it")
        seeder.add_plan_messages(_plan(), "do it")

        ids = [
            c.kwargs["tool_use"].tool_call_id
            for c in ctx.add_message.call_args_list
            if "tool_use" in c.kwargs
        ]
        assert len(set(ids)) == 2

    def test_rendered_content_is_right_stripped(self):
        seeder, ctx, _ = _seeder()
        seeder.add_rewritten_task_message(_task(), "do it")
        assert ctx.add_message.call_args.args[1] == "rendered"
