from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Callable
from uuid import uuid4

from agent.events.events import (
    NextDecisionMade,
    RePlanStarted,
    RePlanSucceed,
    StageExecutionFailed,
    StageExecutionStarted,
    StageExecutionSucceed,
    TaskCancelled,
    TaskPaused,
    UserClarificationRequested,
    UserGuidanceReceived,
)
from config import ConfigReader
from infra.observability.tracing import Tracer
from infra.rendering_engine import Jinja2PromptRenderer, PromptRenderer
from schemas.errors import (
    AGENT_MAX_ITERATIONS_EXCEEDED,
    CallerAction,
    LLMNormalizedError,
    PipelineError,
)
from schemas.event_bus import EventBus
from schemas.ids import StageId
from schemas.task import (
    EvaluationReport,
    NextDecision,
    NextDecisionType,
    Plan,
    PlanStep,
    StageRecoveryAction,
)
from schemas.types import (
    DEFAULT_CLARIFICATION,
    Stage,
    StageStatus,
    UserCommandType,
)
from tools.tool_registry import ToolRegistry
from utils.log.log import Logger, zap

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
    build_tool_use_metadata,
    non_tool_metadata,
)
from agent.models.model_routing.provider_router import ModelSelector

if TYPE_CHECKING:
    from agent.application.driver import PipelineDriver
    from agent.models.context.context_manager import ContextManager
    from agent.models.evaluate.quality_evaluator import QualityEvaluator
    from agent.models.knowledge.knowledge_loader import KnowledgeLoader
    from agent.models.plan.planner import Planner
    from agent.models.reasoning.reasoning_manager import ReasoningManager
    from llm.llm_gateway import LLMGateway

# These names moved to loop_state.py; re-exported under their old private names
# so existing imports of `from ...stage_executor import _StageOutcome` keep working.
_StartReason = StageStartReason
_StageOutcome = StageOutcome
_StageResult = StageResult
_StageRecoveryResult = StageRecoveryResult
_escalate_recovery_action = escalate_recovery_action

__all__ = [
    "StageExecutor",
    "StageStartReason",
    "StageOutcome",
    "StageResult",
    "StageRecoveryResult",
    "escalate_recovery_action",
    "_StartReason",
    "_StageOutcome",
    "_StageResult",
    "_StageRecoveryResult",
    "_escalate_recovery_action",
]


class StageExecutor:
    """Drives the Stage level loop and the Stage internal reasoning loop.

    Stage level (`execute`): iterates over plan steps, handles eval / recovery /
    model switch. One step per iteration; every branch either returns, retries
    the same index, or advances.

    Reasoning level (`_execute_stage`): ReAct loop — poll user, reason, act on
    the decision. Each decision type has its own handler in `_decision_handlers`;
    a handler returns None to keep looping or a StageResult to exit.
    """

    def __init__(
        self,
        config: ConfigReader,
        logger: Logger,
        tracer: Tracer,
        reasoning_manager: ReasoningManager,
        context_manager: ContextManager,
        quality_evaluator: QualityEvaluator,
        knowledge_loader: KnowledgeLoader,
        planner: Planner,
        llm_gateway: LLMGateway,
        event_bus: EventBus,
        model_selector: ModelSelector,
        tool_registry: ToolRegistry,
        renderer: PromptRenderer | None = None,
        context_seeder: TaskContextSeeder | None = None,
    ) -> None:
        self._config = config
        self._logger = logger
        self._tracer = tracer
        self._reasoning_manager = reasoning_manager
        self._context_manager = context_manager
        self._quality_evaluator = quality_evaluator
        self._knowledge_loader = knowledge_loader  # kept for DI compatibility
        self._planner = planner
        self._llm_gateway = llm_gateway
        self._event_bus = event_bus
        self._model_selector = model_selector
        self._tool_registry = tool_registry
        self._renderer: PromptRenderer = renderer or Jinja2PromptRenderer()

        self._max_iterations = int(self._config.get("agent.max_attempt_iterations", 60))
        self._max_replan_stage_retries = int(self._config.get("agent.max_replan_stage_retries", 3))
        self._max_total_replan_count = int(self._config.get("agent.max_total_replan_count", 8))
        self._max_tool_consecutive_count = int(self._config.get("agent.max_tool_consecutive_count", 15))
        forbidden_tools: frozenset[str] = frozenset(self._config.get("tools.forbidden_tools", []))

        self._context_seeder = context_seeder or TaskContextSeeder(
            config=config,
            logger=logger,
            renderer=self._renderer,
            context_manager=context_manager,
            tool_registry=tool_registry,
        )
        self._tool_dispatcher = ToolDispatcher(
            logger=logger,
            tool_registry=tool_registry,
            reasoning_manager=reasoning_manager,
            context_manager=context_manager,
            event_bus=event_bus,
            forbidden_tools=forbidden_tools,
        )
        self._decision_handlers: dict[
            NextDecisionType,
            Callable[[Stage, NextDecision, ReactState], StageResult | None],
        ] = {
            NextDecisionType.FINAL_ANSWER:         self._on_final_answer,
            NextDecisionType.CONTINUE:             self._on_continue,
            NextDecisionType.TOOL_CALL:            self._on_tool_call,
            NextDecisionType.CLARIFICATION_NEEDED: self._on_clarification,
            NextDecisionType.PAUSED:               self._on_paused,
        }

        self._current_stage: Stage | None = None
        self._current_stage_index: int = 0
        self._correction_feedback: list[str] = []
        self._task_recovery_feedback: str = ""
        self._task_description: str = ""
        self._task_output_constraints: str = ""
        self._task_goal: str = ""
        self._task_intent: str = ""
        self._agent_poll_timeout = self._config.positive_float(
            "agent.latency.agent_message_poll_timeout_seconds", 0.1
        )
        self._driver: PipelineDriver | None = None
        self._on_stage_success: Callable[[int], None] | None = None

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def set_driver(self, driver: PipelineDriver) -> None:
        self._driver = driver

    def set_stage_success_callback(self, cb: Callable[[int], None]) -> None:
        self._on_stage_success = cb

    def set_task_description(self, task_description: str) -> None:
        self._task_description = task_description

    def set_task_output_constraints(self, output_constraints: str) -> None:
        self._task_output_constraints = output_constraints or ""

    def set_task_goal(self, task_goal: str) -> None:
        self._task_goal = task_goal or ""

    def set_task_intent(self, task_intent: str) -> None:
        self._task_intent = task_intent or ""

    def set_task_recovery_feedback(self, feedback: str) -> None:
        self._task_recovery_feedback = feedback

    def set_llm_gateway(self, llm_gateway: LLMGateway) -> None:
        self._llm_gateway = llm_gateway
        self._reasoning_manager.set_llm_gateway(self._llm_gateway)

    def get_task_output_constraints(self) -> str:
        return self._task_output_constraints

    def get_task_goal(self) -> str:
        return self._task_goal

    def get_task_intent(self) -> str:
        return self._task_intent

    def get_task_recovery_feedback(self) -> str:
        return self._task_recovery_feedback

    def get_conversation_history(self):
        return self._context_manager.get_conversation_history()

    def reset(self) -> None:
        self._context_manager.reset()
        self._correction_feedback = []
        self._task_recovery_feedback = ""

    # ------------------------------------------------------------------
    # Stage level loop
    # ------------------------------------------------------------------

    def execute(self, plan: Plan, start_step_index: int = 0) -> str | None:
        """Execute all stages in *plan* and return the final result string.

        Returns None when execution is cancelled or fatally failed.
        """
        state = StageLoopState(step_index=start_step_index,
                               correction_feedback=self._correction_feedback)
        self._logger.info("Enter Stage execution", plan_id=plan.id, task_id=plan.task_id)

        while state.step_index < len(plan.step_list):
            stage = self._start_stage(plan, state)
            result = self._run_reasoning_loop(plan, stage, state)

            if result.outcome is StageOutcome.FATAL:
                self._report_stage_failure(plan, stage, state)
                return None

            if result.outcome is StageOutcome.SWITCH_MODEL:
                self._switch_provider(plan, stage, result)
                state.start_reason = StageStartReason.MODEL_SWITCH
                continue

            self._settle_stage_output(stage)
            if self._is_last_step(plan, state):
                self._logger.info("The last stage execution succeed",
                    task_id=plan.task_id, plan_id=plan.id, final_step_order=stage.order)
                return stage.result

            review = self._evaluate_stage(plan, stage, state)
            if not review.passed:
                plan = self._recover_stage(plan, stage, state, review)
                continue

            self._commit_stage(plan, stage, state)

        raise PipelineError(AGENT_MAX_ITERATIONS_EXCEEDED, "reach max iterations")

    # ── Stage level steps ─────────────────────────────────────────────

    def _start_stage(self, plan: Plan, state: StageLoopState) -> Stage:
        """Build the Stage for the current step, announce it, and open its context."""
        step = plan.step_list[state.step_index]
        self._current_stage_index = state.step_index
        provider = self._model_selector.get_current_provider()

        self._event_bus.publish(
            StageExecutionStarted.with_meta(
                task_id=plan.task_id,
                step_order=state.step_index + 1,
                total_steps=len(plan.step_list),
                stage_goal=step.goal,
                stage_description=step.description,
                required_stage_tools=step.required_tools,
                start_reason=state.start_reason.value,
                provider=provider,
            )
        )

        stage = Stage(
            id=StageId(str(uuid4())),
            task_id=plan.task_id,
            plan_step_id=step.id,
            order=state.step_index + 1,
            goal=step.goal,
            description=step.description,
            key_results=step.key_results,
            inputs=step.inputs,
            required_tools=step.required_tools,
            action_constraints=step.action_constraints,
            risks=step.risks,
            dependencies=step.dependencies,
            execution_notes=step.execution_notes,
            output_constraints=step.output_constraints,
        )
        self._current_stage = stage
        self._logger.info("Stage started",
            task_id=plan.task_id, step_order=stage.order, goal=step.goal,
            start_reason=state.start_reason.value, provider=provider)
        self._context_manager.begin_stage(stage)
        return stage

    def _run_reasoning_loop(
        self, plan: Plan, stage: Stage, state: StageLoopState
    ) -> StageResult:
        """Run the inner reasoning loop for *stage* inside its own span."""
        provider = self._model_selector.get_current_provider()
        is_last = self._is_last_step(plan, state)
        with self._tracer.start_span("stage.execute", "stage",
            task_id=plan.task_id, plan_id=plan.id,
            stage_id=stage.id, step_index=state.step_index,
            goal=stage.goal, provider=provider,
            start_reason=state.start_reason.value) as span:

            result = self._execute_stage(
                stage, provider,
                total_steps=len(plan.step_list),
                correction_feedback=state.correction_feedback,
                task_recovery_feedback=self._task_recovery_feedback,
                is_last=is_last,
                task_output_constraints=self._task_output_constraints if is_last else "",
            )
            span.add_attributes({
                "outcome": result.outcome.name,
                "iterations": stage.iteration_count,
                "result_length": len(stage.result),
            })

        self._logger.info("Stage execution outcome",
            task_id=plan.task_id, step_order=stage.order,
            outcome=result.outcome.name, use_iterations=stage.iteration_count)
        return result

    def _report_stage_failure(self, plan: Plan, stage: Stage, state: StageLoopState) -> None:
        self._logger.error("Stage execution ended fatally",
            task_id=plan.task_id, step_order=stage.order, reason=stage.result)
        self._event_bus.publish(
            StageExecutionFailed.with_meta(
                task_id=plan.task_id,
                step_order=state.step_index + 1,
                total_steps=len(plan.step_list),
                stage_goal=stage.goal,
                reason=stage.result,
            )
        )

    def _switch_provider(self, plan: Plan, stage: Stage, result: StageResult) -> None:
        next_provider = self._model_selector.advance_provider(result.llm_error)
        self._context_manager.drop_latest_stage_context()
        self._logger.info("Stage need switch provider",
            task_id=plan.task_id, step_order=stage.order, next_provider=next_provider)

    def _settle_stage_output(self, stage: Stage) -> None:
        self._model_selector.confirm_provider_success()
        stage.result = self._normalize_stage_output(stage.result)

    def _evaluate_stage(
        self, plan: Plan, stage: Stage, state: StageLoopState
    ) -> EvaluationReport:
        step = plan.step_list[state.step_index]
        report = self._quality_evaluator.evaluate_stage_result(
            step,
            stage.result,
            self._llm_gateway,
            task_goal=self._task_goal,
            task_intent=self._task_intent,
        )
        self._logger.info("Stage result evaluation complete",
            task_id=plan.task_id, step_order=stage.order, passed=report.passed,
            recovery_action=None if report.recovery_action is None else report.recovery_action.value,
            feedback=report.feedback)
        return report

    def _recover_stage(
        self, plan: Plan, stage: Stage, state: StageLoopState, review: EvaluationReport
    ) -> Plan:
        """Apply the eval-failure recovery action and return the plan to retry with."""
        step = plan.step_list[state.step_index]
        self._guard_replan_limits(state, step)

        action = state.resolve_action(review)
        if action != (review.recovery_action or StageRecoveryAction.REPLAN_THIS_STEP):
            self._logger.warning("RETRY_SAME_STEP repeated, escalating recovery",
                task_id=plan.task_id, step_order=stage.order,
                same_failure_count=state.same_failure_count,
                original_action=(review.recovery_action or StageRecoveryAction.REPLAN_THIS_STEP).value,
                escalated_action=action.value)

        with self._tracer.start_span("stage.recovery", "stage",
            task_id=plan.task_id, plan_id=plan.id,
            step_index=state.step_index, action=action.value,
            attempt=state.replan_attempts):
            self._event_bus.publish(
                RePlanStarted.with_meta(
                    task_id=plan.task_id,
                    step_order=state.step_index + 1,
                    total_steps=len(plan.step_list),
                    stage_goal=step.goal,
                    feedback=review.feedback,
                    recovery_action=action.value,
                    retry=state.replan_attempts + 1,
                    max_retries=self._max_replan_stage_retries,
                )
            )
            recovery = self._apply_stage_recovery(action, plan, state.step_index, review.feedback)
            self._logger.info("Stage begin to rerun",
                task_id=plan.task_id, step_order=stage.order)

        plan = state.apply_recovery(recovery)
        self._event_bus.publish(
            RePlanSucceed.with_meta(
                task_id=plan.task_id,
                step_order=state.step_index + 1,
                total_steps=len(plan.step_list),
                recovery_action=action.value,
                start_reason=state.start_reason.value,
            )
        )
        state.on_recovery(action, review.feedback)
        self._correction_feedback = state.correction_feedback
        return plan

    def _guard_replan_limits(self, state: StageLoopState, step: PlanStep) -> None:
        if state.replan_attempts >= self._max_replan_stage_retries:
            raise PipelineError(
                "LLM_REPLAN_LIMIT_EXCEEDED",
                f"Per-step replan limit ({self._max_replan_stage_retries}) exceeded "
                f"at stage {state.step_index + 1}: {step.goal}",
            )
        if state.total_replan_count > self._max_total_replan_count:
            raise PipelineError(
                "LLM_REPLAN_LIMIT_EXCEEDED",
                f"Total stage replan-all limit ({self._max_total_replan_count}) exceeded "
                f"at stage {state.step_index + 1}: {step.goal}",
            )

    def _commit_stage(self, plan: Plan, stage: Stage, state: StageLoopState) -> None:
        """Eval passed: record the result, checkpoint, and advance to the next step."""
        state.on_stage_passed()
        self._correction_feedback = state.correction_feedback

        summary = f"## Step {state.step_index + 1}'s result\n\n{stage.result}"
        self._context_manager.add_message("assistant", summary)
        # Summarise and update context (async LLM summarisation inside end_stage)
        self._context_manager.end_stage(stage, success=True)
        self._logger.info("Stage completed",
            task_id=plan.task_id, step_order=stage.order, is_last=False,
            result_length=len(stage.result))

        # Async checkpoint save after each non-final stage eval passes
        if self._on_stage_success is not None:
            self._on_stage_success(state.step_index)

        self._event_bus.publish(
            StageExecutionSucceed.with_meta(
                task_id=plan.task_id,
                step_order=state.step_index + 1,
                total_steps=len(plan.step_list),
                stage_goal=stage.goal,
                result=summary,
            )
        )
        state.advance()

    @staticmethod
    def _is_last_step(plan: Plan, state: StageLoopState) -> bool:
        return state.step_index == len(plan.step_list) - 1

    # ------------------------------------------------------------------
    # Stage internal reasoning loop
    # ------------------------------------------------------------------

    def _execute_stage(
        self, stage: Stage, provider_name: str, total_steps: int = 0,
        correction_feedback: list[str] | None = None,
        task_recovery_feedback: str = "",
        is_last: bool = False,
        task_output_constraints: str = "",
    ) -> StageResult:
        """ReAct reasoning loop for a single stage.

        Per iteration: poll user commands, build context, ask the LLM, then let
        the handler for the returned decision type act. A handler returns None to
        continue looping or a StageResult to end the stage.
        """
        self._context_manager.add_message("user", self._renderer.render(
            "stage_executor/stage_prompt.j2", {
                "stage": stage,
                "total_steps": total_steps,
                "correction_feedback": correction_feedback,
                "task_recovery_feedback": task_recovery_feedback,
                "is_last": is_last,
                "task_output_constraints": task_output_constraints,
            }))
        react = ReactState()

        while stage.iteration_count < self._max_iterations:
            self._logger.info("Stage inner process started",
                task_id=stage.task_id, step_order=stage.order,
                current_iteration=stage.iteration_count, provider=provider_name)

            interrupt = self._poll_user_commands(stage)
            if interrupt is not None:
                return interrupt

            try:
                decision = self._reason_once(stage, provider_name)
            except (LLMNormalizedError, PipelineError) as exc:
                return self._on_reasoning_error(stage, provider_name, exc)

            self._publish_decision(stage, decision)

            handler = self._decision_handlers.get(
                decision.decision_type, self._on_unknown_decision)
            result = handler(stage, decision, react)
            if result is not None:
                return result

        stage.fail(f"Max iterations ({self._max_iterations}) exceeded")
        self._logger.error("Stage exceeded max iterations",
            task_id=stage.task_id, step_order=stage.order, max_iterations=self._max_iterations)
        return StageResult(outcome=StageOutcome.SWITCH_MODEL)  # llm_error=None → default cooloff

    # ── Reasoning loop steps ──────────────────────────────────────────

    def _poll_user_commands(self, stage: Stage) -> StageResult | None:
        """Non-blocking poll for user co-operation signals.

        CANCEL ends the stage; GUIDANCE is injected into context and the loop
        carries on.
        """
        user_cmd = self._driver.loop_user_messages(self._agent_poll_timeout)
        if user_cmd is None:
            return None

        self._logger.info("User command received during stage",
            task_id=stage.task_id, step_order=stage.order,
            command_type=_enum_value(user_cmd.type))

        if user_cmd.type == UserCommandType.CANCEL:
            self._event_bus.publish(
                TaskCancelled.with_meta(task_id=stage.task_id, message="Task cancelled by user.")
            )
            stage.fail("Cancelled by user.")
            self._logger.info("User cancel the task",
                task_id=stage.task_id, step_order=stage.order)
            return StageResult(outcome=StageOutcome.FATAL)

        if user_cmd.type == UserCommandType.GUIDANCE:
            guidance = (user_cmd.content or "").strip()
            self._logger.info("Receive guidance from use for this task",
                task_id=stage.task_id, step_order=stage.order)
            self._event_bus.publish(
                UserGuidanceReceived.with_meta(
                    task_id=stage.task_id,
                    step_order=stage.order,
                    message=guidance,
                )
            )
            self._context_manager.add_message("user", guidance)
        return None

    def _reason_once(self, stage: Stage, provider_name: str) -> NextDecision:
        context_window = self._context_manager.get_context_window(provider_name)
        with self._tracer.start_span("stage.reason_once", "reasoning",
            task_id=stage.task_id, stage_id=stage.id,
            iteration=stage.iteration_count, provider=provider_name,
            message_count=len(context_window.messages),
            tool_schema_count=len(context_window.tool_schemas or [])) as span:
            decision = self._reasoning_manager.reason_once(
                context_window,
                provider_name,
                json_mode=True,
                json_required_keys=["final_answer"],
            )
            span.add_attributes({
                "decision_type": _enum_value(decision.decision_type),
                "tool_call_count": len(decision.tool_calls),
                "has_assistant_message": decision.assistant_message is not None,
            })
        return decision

    def _on_reasoning_error(
        self, stage: Stage, provider_name: str, exc: LLMNormalizedError | PipelineError
    ) -> StageResult:
        """Map a reasoning failure onto a stage outcome.

        Non-fatal LLM errors ask the Stage level loop for a provider switch;
        everything else is fatal.
        """
        if isinstance(exc, LLMNormalizedError):
            self._logger.error("LLM error happened during stage reasoning",
                task_id=stage.task_id, step_order=stage.order,
                iteration=stage.iteration_count, provider=provider_name,
                error_code=_enum_value(exc.code),
                caller_action=_enum_value(exc.caller_action),
                message=exc.message)
            if exc.caller_action == CallerAction.FATAL:
                stage.fail(f"Fatal LLM error: {exc.message}")
                return StageResult(outcome=StageOutcome.FATAL)
            stage.fail(f"LLM error: {exc.message}")
            return StageResult(outcome=StageOutcome.SWITCH_MODEL, llm_error=exc)

        self._logger.error("Pipeline error happened during stage reasoning",
            task_id=stage.task_id, step_order=stage.order,
            iteration=stage.iteration_count,
            error_code=exc.code, message=exc.message)
        stage.fail(f"Agent error: {exc.message}")
        return StageResult(outcome=StageOutcome.FATAL)

    def _publish_decision(self, stage: Stage, decision: NextDecision) -> None:
        self._event_bus.publish(
            NextDecisionMade.with_meta(
                task_id=stage.task_id,
                step_order=stage.order,
                iteration=stage.iteration_count + 1,
                decision=_enum_value(decision.decision_type),
                tool_name=", ".join(call.name for call in decision.tool_calls),
            )
        )

    # ── Decision handlers ─────────────────────────────────────────────
    # Each returns None to continue the loop, or a StageResult to end the stage.

    def _on_final_answer(
        self, stage: Stage, decision: NextDecision, react: ReactState
    ) -> StageResult:
        answer = self._extract_final_answer(decision.answer)
        stage.increment_iteration()
        stage.complete(answer)
        self._logger.info("Stage final answer produced",
            task_id=stage.task_id, step_order=stage.order,
            used_iteration=stage.iteration_count, answer_length=len(answer))
        return StageResult(outcome=StageOutcome.SUCCESS)

    def _on_continue(
        self, stage: Stage, decision: NextDecision, react: ReactState
    ) -> None:
        react.tool_consecutive_count = 0
        content = decision.message or (
            decision.assistant_message.content if decision.assistant_message else ""
        )
        self._context_manager.add_message(
            "assistant", content,
            extra_metadata=non_tool_metadata(decision.assistant_message),
        )
        stage.increment_iteration()
        self._logger.info("Stage continue decision recorded",
            task_id=stage.task_id, step_order=stage.order,
            used_iteration=stage.iteration_count, content_length=len(content))
        return None

    def _on_tool_call(
        self, stage: Stage, decision: NextDecision, react: ReactState
    ) -> None:
        react.tool_consecutive_count += 1
        if decision.assistant_message:
            self._context_manager.add_message(
                decision.assistant_message.role,
                decision.assistant_message.content,
                tool_use=build_tool_use_metadata(decision.assistant_message.metadata),
                extra_metadata=non_tool_metadata(decision.assistant_message),
            )
        self._tool_dispatcher.dispatch(stage, decision.tool_calls)
        stage.increment_iteration()
        self._logger.info("Stage tool-call decision processed",
            task_id=stage.task_id, step_order=stage.order,
            used_iteration=stage.iteration_count, tool_call_count=len(decision.tool_calls))

        if react.tool_consecutive_count > self._max_tool_consecutive_count:
            self._context_manager.add_message(
                "user",
                f"The number of consecutive tool calls has reached "
                f"{self._max_tool_consecutive_count}. Now, please complete the current "
                f"step more efficiently",
            )
            react.tool_consecutive_count = 0
        return None

    def _on_clarification(
        self, stage: Stage, decision: NextDecision, react: ReactState
    ) -> None:
        react.tool_consecutive_count = 0
        question = decision.message or "Please provide clarification."
        if decision.assistant_message:
            self._context_manager.add_message(
                "assistant", decision.assistant_message.content,
                extra_metadata=non_tool_metadata(decision.assistant_message),
            )
        else:
            self._context_manager.add_message("assistant", question)

        self._event_bus.publish(
            UserClarificationRequested.with_meta(
                task_id=stage.task_id,
                question=question,
            )
        )
        user_cmd = self._driver.loop_user_messages(timeout=1)
        has_clarification = (user_cmd is not None) and (user_cmd.type == UserCommandType.CLARIFICATION)
        clarification = user_cmd.content.strip() if has_clarification else DEFAULT_CLARIFICATION
        self._logger.info(
            "Receive user's clarification",
            zap.any("task_id", stage.task_id),
            zap.any("has_clarification", bool(has_clarification)),
            zap.any("user_clarification", clarification),
        )
        self._context_manager.add_message("user", f"Clarification: {clarification}")
        stage.increment_iteration()
        self._logger.info("Stage clarification handled",
            task_id=stage.task_id, step_order=stage.order, used_iteration=stage.iteration_count)
        return None

    def _on_paused(
        self, stage: Stage, decision: NextDecision, react: ReactState
    ) -> None:
        """Agent-requested pause.

        Waits for RESUME. While no RESUME arrives the iteration counter is left
        untouched on purpose, so the loop keeps waiting instead of burning the
        iteration budget — the stage stays parked until the user resumes.
        """
        react.tool_consecutive_count = 0
        reason = decision.message or "Task paused."
        if decision.assistant_message:
            self._context_manager.add_message(
                "assistant", decision.assistant_message.content,
                extra_metadata=non_tool_metadata(decision.assistant_message),
            )

        self._event_bus.publish(TaskPaused.with_meta(task_id=stage.task_id, reason=reason))
        stage.pause()
        resume_cmd = self._driver.loop_user_messages(timeout=1)
        if resume_cmd is not None and resume_cmd.type == UserCommandType.RESUME:
            stage.status = StageStatus.RUNNING
            stage.increment_iteration()
            self._logger.info("Stage paused by agent but now resumed",
                task_id=stage.task_id, step_order=stage.order,
                used_iteration=stage.iteration_count)
        return None

    def _on_unknown_decision(
        self, stage: Stage, decision: NextDecision, react: ReactState
    ) -> None:
        """Fallback for a decision type with no registered handler.

        Counts the iteration so an unhandled type can't spin the loop forever;
        the stage then ends on the iteration ceiling like any stuck stage.
        """
        stage.increment_iteration()
        self._logger.warning("Unhandled decision type in reasoning loop",
            task_id=stage.task_id, step_order=stage.order,
            decision=_enum_value(decision.decision_type),
            used_iteration=stage.iteration_count)
        return None

    # ------------------------------------------------------------------
    # Stage recovery
    # ------------------------------------------------------------------

    def _apply_stage_recovery(
        self,
        action: StageRecoveryAction,
        plan: Plan,
        step_index: int,
        feedback: str,
    ) -> StageRecoveryResult:
        """根据 LLM 建议的恢复模式清理上下文并更新计划。代价从低到高。"""
        self._logger.info("Applying stage recovery",
            plan_id=plan.id, step_index=step_index, action=action.value, feedback=feedback)

        if action == StageRecoveryAction.RETRY_SAME_STEP:
            self._context_manager.drop_latest_stage_context()
            return StageRecoveryResult(plan, step_index, StageStartReason.EVAL_RETRY, False)

        if action == StageRecoveryAction.REPLAN_THIS_STEP:
            self._context_manager.drop_latest_stage_context()
            step = self._replan_step(plan.step_list[step_index], feedback)
            plan = _replace_step(plan, step_index, step)
            self._context_manager.set_plan(plan)
            return StageRecoveryResult(plan, step_index, StageStartReason.EVAL_RETRY, False)

        if action == StageRecoveryAction.REPLAN_FROM_HERE:
            self._context_manager.drop_stages_from(step_index)
            task = self._context_manager.get_task()
            plan = self._planner.renew_plan_from_step(
                task, plan, step_index, feedback, self._llm_gateway)
            self._context_manager.set_plan(plan)
            return StageRecoveryResult(plan, step_index, StageStartReason.REPLAN_FROM, False)

        # REPLAN_ALL: 代价最高，清空全部上下文，从 step 0 重新开始
        self._context_manager.reset()
        task = self._context_manager.get_task()
        plan, task = self._planner.renew_plan(
            task=task, feedback=feedback, llm_api=self._llm_gateway)
        self._context_manager.set_task(task)
        self._context_manager.set_plan(plan)
        self._context_seeder.apply_tool_filter(task, reason="stage_replan_all")
        self._context_seeder.seed(task, plan, self._task_description)
        return StageRecoveryResult(plan, 0, StageStartReason.REPLAN_ALL, True)

    def _replan_step(self, step: PlanStep, feedback: str) -> PlanStep:
        task = self._context_manager.get_task()
        plan = self._context_manager.get_plan()
        return self._planner.renew_plan_step(task, step, feedback, self._llm_gateway, plan)

    # ------------------------------------------------------------------
    # Output normalisation
    # ------------------------------------------------------------------

    def _extract_final_answer(self, answer: str) -> str:
        """Unwrap the JSON-mode protocol envelope from the business artifact."""
        stripped = answer.strip()
        if not stripped:
            return answer

        try:
            data = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return stripped

        if isinstance(data, dict) and isinstance(data.get("final_answer"), str):
            return data["final_answer"].strip()
        return stripped

    def _normalize_stage_output(self, result: str) -> str:
        """Normalize only unambiguous JSON wrapping in stage output.

        Stage results may be plain markdown/text or JSON depending on the step.
        Be conservative here: silently extracting an embedded JSON fragment can
        truncate a valid markdown deliverable, so only unwrap when the whole
        output is JSON or a single JSON code fence.
        """
        stripped = result.strip()
        if not stripped:
            return result

        try:
            json.loads(stripped)
            return stripped
        except (json.JSONDecodeError, ValueError):
            pass

        fence_match = re.fullmatch(r"```(?:json|JSON)?\s*\n?([\s\S]*?)\n?```", stripped)
        if fence_match:
            candidate = fence_match.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except (json.JSONDecodeError, ValueError):
                pass

        return stripped


# ── Module-level helpers ──────────────────────────────────────────────────────

def _enum_value(value) -> str:
    """Enum member → its value; anything else → str(). Used for log/span fields."""
    return value.value if hasattr(value, "value") else str(value)


def _replace_step(plan: Plan, index: int, new_step: PlanStep) -> Plan:
    """Return a new Plan with step at *index* replaced by *new_step*."""
    new_steps = list(plan.step_list)
    new_steps[index] = new_step
    return Plan(
        id=plan.id,
        task_id=plan.task_id,
        step_list=new_steps,
        created_at=plan.created_at,
    )
