from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from tools.tool_registry import ToolRegistry
import threading
from typing import TYPE_CHECKING
from uuid import uuid4

from agent.events.events import (
    NextDecisionMade,
    StageExecutionFailed,
    StageExecutionStarted,
    StageExecutionSucceed,
    TaskCancelled,
    TaskPaused,
    ToolCallResultProduced,
    ToolCallStarted,
    UserClarificationRequested,
)
from config import ConfigReader
from infra.observability.tracing import Tracer
from infra.rendering_engine import Jinja2PromptRenderer, PromptRenderer
from schemas.errors import (
    AGENT_MAX_ITERATIONS_EXCEEDED,
    PipelineError,
    CallerAction,
    LLMNormalizedError,
    TOOL_NOT_FOUND,
    TOOL_ARGUMENT_ERROR,
    build_pipeline_error,
)
from schemas.event_bus import EventBus
from schemas.ids import StageId
from schemas.task import (
    NextDecisionType,
    Plan,
    PlanStep,
    StageRecoveryAction,
)
from schemas.types import DEFAULT_CLARIFICATION, Stage, StageStatus, ToolCall, ToolResult, UserCommandType, LLMMessage
from utils.log.log import Logger, zap

if TYPE_CHECKING:
    from agent.application.driver import PipelineDriver
    from agent.models.context.context_manager import ContextManager
    from agent.models.evaluate.quality_evaluator import QualityEvaluator
    from agent.models.knowledge.knowledge_loader import KnowledgeLoader
    from agent.models.plan.planner import Planner
    from agent.models.reasoning.reasoning_manager import ReasoningManager
    from llm.llm_gateway import LLMGateway

from agent.models.context.context_manager import ToolCallEntry, ToolResultMetadata, ToolUseMetadata
from agent.models.model_routing.provider_router import ModelSelector


# ── Stage start reason labels (shown to user) ─────────────────────────────────

class _StartReason(str, Enum):
    NEW          = "A. New stage execution"
    EVAL_RETRY   = "B. Eval failed — step updated, retrying"
    MODEL_SWITCH = "C. Model switched, retrying"
    REPLAN       = "D. Execution failed — plan updated, retrying"
    REPLAN_FROM  = "E. Eval failed — replanned from current step"
    REPLAN_ALL   = "F. Eval failed — full replan, restarting from step 1"


# ── Internal outcome codes from _execute_stage ────────────────────────────────

class _StageOutcome(Enum):
    SUCCESS                    = auto()  # stage.complete() was called
    SWITCH_MODEL               = auto()  # LLMError that warrants a provider switch
    FATAL                      = auto()  # cancelled / unrecoverable error


@dataclass
class _StageRecoveryResult:
    """Returned by _apply_stage_recovery; carries updated loop variables."""
    plan: Plan
    step_index: int
    start_reason: _StartReason
    reset_replan_counter: bool  # True only for REPLAN_ALL (restarts from step 0)


@dataclass
class _StageResult:
    """Returned by _execute_stage; carries outcome, optional guidance, and the raw LLM error."""
    outcome: _StageOutcome
    llm_error: LLMNormalizedError | None = None


class StageExecutor:
    """Drives the Stage Level loop and the Stage internal reasoning loop.

    Stage Level (execute): iterates over plan steps, handles eval/retry/model-switch.
    Reasoning loop (_execute_stage): ReAct loop — reason → tool → inject → repeat.

    Interrupt/pause/resume are signalled via threading.Event so Pipeline can call
    interrupt() / pause() / resume() from a different thread while the loop runs.
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
    ) -> None:
        self._config = config
        self._logger = logger
        self._tracer = tracer
        self._reasoning_manager = reasoning_manager
        self._context_manager = context_manager
        self._quality_evaluator = quality_evaluator
        self._knowledge_loader = knowledge_loader
        self._planner = planner
        self._llm_gateway = llm_gateway
        self._event_bus = event_bus
        self._model_selector = model_selector
        self._tool_registry = tool_registry
        self._renderer: PromptRenderer = renderer or Jinja2PromptRenderer()

        self._max_iterations = int(self._config.get("agent.max_attempt_iterations", 60))
        self._max_replan_stage_retries = int(self._config.get("agent.max_replan_stage_retries", 3))
        self._max_stage_eval_retries = int(self._config.get("agent.max_stage_retries", 2))
        self._forbidden_tools: frozenset[str] = frozenset(self._config.get("tools.forbidden_tools", []))

        self._current_stage: Stage | None = None
        self._current_stage_index: int = 0
        self._cancelled = threading.Event()
        self._agent_poll_timeout = self._config.positive_float(
            "agent.latency.agent_message_poll_timeout_seconds", 0.1
        )
        self._loop_msg_timeout = self._config.positive_float(
            "agent.latency.loop_user_message_timeout_seconds", 300.0
        )
        self._driver: PipelineDriver | None = None

    def set_driver(self, driver: PipelineDriver) -> None:
        self._driver = driver

    def cancel(self) -> None:
        self._cancelled.set()

    # ------------------------------------------------------------------
    # Stage Level loop
    # ------------------------------------------------------------------

    def execute(
        self,
        plan: Plan,
    ) -> str | None:
        """Execute all stages in *plan* and return the final result string.

        Returns None if execution is cancelled, interrupted, or fatally failed.

        Flow per step:
          1.0  Publish StageExecutionStarted with start-reason label.
          1.1  (Handled inside model_selector.advance_provider) Consider switching
               back to the highest-priority recovered provider.
          1.2  Run the internal reasoning loop (_execute_stage).
          1.2.1  SUCCESS → evaluate result.
            1.2.1.1  Eval passed → summarise, checkpoint, advance (or deliver).
            1.2.1.2  Eval failed → reset ctx, replan step, retry from 1.
          1.2.2  SWITCH_MODEL → reset ctx, switch provider, retry from 1.
          1.2.3  NEED_REPLAN → reset ctx, replan step, retry from 1.
          1.2.4  FATAL → return None.
        """
        step_index: int = 0
        start_reason: _StartReason = _StartReason.NEW
        current_replan_stage_attempts = 0
        self._logger.info("Enter Stage execution", plan_id=plan.id, task_id=plan.task_id)

        while step_index < len(plan.step_list):
            self._event_bus.publish(StageExecutionStarted.with_meta(task_id=plan.task_id, step_order=step_index+1))

            step = plan.step_list[step_index]
            total = len(plan.step_list)
            self._current_stage_index = step_index

            self._current_stage = Stage( 
                id=StageId(str(uuid4())),
                task_id=plan.task_id,
                plan_step_id=step.id,
                order=step_index+1,
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
            _provider = self._model_selector.get_current_provider()
            self._logger.info("Stage started",
                task_id=plan.task_id, step_order=self._current_stage.order, goal=step.goal, start_reason=start_reason.value, provider=_provider)

            # ── 1.2 Run reasoning loop ─────────────────────────────────────
            self._context_manager.begin_stage(self._current_stage)
            with self._tracer.start_span("stage.execute", "stage",
                task_id=plan.task_id, plan_id=plan.id,
                stage_id=self._current_stage.id, step_index=step_index,
                goal=step.goal, provider=_provider,
                start_reason=start_reason.value) as span:
                stage_result = self._execute_stage(
                    self._current_stage, _provider,
                    total_steps=len(plan.step_list),
                )
                span.add_attributes({
                    "outcome": stage_result.outcome.name,
                    "iterations": self._current_stage.iteration_count,
                    "result_length": len(self._current_stage.result),
                })
            self._logger.info("Stage execution outcome",
                task_id=plan.task_id, step_order=self._current_stage.order,
                outcome=stage_result.outcome.name,
                use_iterations=self._current_stage.iteration_count)
            outcome = stage_result.outcome

            # ── 1.2.4 Fatal (cancel / unrecoverable) ──────────────────────
            if outcome == _StageOutcome.FATAL:
                self._logger.error("Stage execution ended fatally",
                    task_id=plan.task_id, step_order=self._current_stage.order,
                    reason=self._current_stage.result)

                self._event_bus.publish(StageExecutionFailed.with_meta(task_id=plan.task_id, step_order=step_index+1, reason=self._current_stage.result))
                return None

            # ── 1.2.2 Switch model ─────────────────────────────────────────
            if outcome == _StageOutcome.SWITCH_MODEL:
                next_provider = self._model_selector.advance_provider(stage_result.llm_error)
                self._context_manager.drop_latest_stage_context()
                start_reason = _StartReason.MODEL_SWITCH
                self._logger.info("Stage need switch provider",
                    task_id=plan.task_id, step_order=self._current_stage.order, next_provider=next_provider)
                continue  # retry same step_index

            # ── 1.2.1 Stage succeeded — evaluate result ────────────────────
            assert outcome == _StageOutcome.SUCCESS
            self._model_selector.confirm_provider_success()

            is_last = (step_index == (len(plan.step_list) - 1))
            if is_last:
                # 1.2.1.1.4 All stages done — deliver final result
                self._logger.info("The last stage execution succeed",
                    task_id=plan.task_id, plan_id=plan.id, final_step_order=self._current_stage.order)
                return self._current_stage.result

            eval_report = self._quality_evaluator.evaluate_stage_result(
                step,
                self._current_stage.result,
                self._llm_gateway,
            )
            self._logger.info("Stage result evaluation complete",
                task_id=plan.task_id, step_order=self._current_stage.order, passed=eval_report.passed,
                recovery_action=None if eval_report.recovery_action is None else eval_report.recovery_action.value,
                feedback=eval_report.feedback)

            if not eval_report.passed:
                current_replan_stage_attempts += 1
                if current_replan_stage_attempts > self._max_replan_stage_retries:
                    raise PipelineError(
                        "LLM_REPLAN_LIMIT_EXCEEDED",
                        f"Max replan attempts exceeded at stage {step_index + 1}: {step.goal}",
                    )
                action = eval_report.recovery_action or StageRecoveryAction.REPLAN_THIS_STEP
                with self._tracer.start_span("stage.recovery", "stage",
                    task_id=plan.task_id, plan_id=plan.id,
                    step_index=step_index, action=action.value,
                    attempt=current_replan_stage_attempts):
                    recovery = self._apply_stage_recovery(action, plan, step_index, eval_report.feedback)

                    self._logger.info("Stage begin to rerun", task_id=plan.task_id, step_order=self._current_stage.order)

                plan, step_index, start_reason = recovery.plan, recovery.step_index, recovery.start_reason
                if recovery.reset_replan_counter:
                    current_replan_stage_attempts = 0
                continue

            # ── 1.2.1.1 Eval passed ────────────────────────────────────────
            stage_summary = (
                f"## Step {step_index + 1}'s result\n\n"
                f"{self._current_stage.result}"
            )
            self._context_manager.add_message("assistant", stage_summary)
            # Summarise and update context (async LLM summarisation inside end_stage)
            self._context_manager.end_stage(self._current_stage, success=True)
            self._logger.info("Stage completed",
                task_id=plan.task_id, step_order=self._current_stage.order, is_last=is_last,
                result_length=len(self._current_stage.result))

            # 1.2.1.1.3 Advance to next stage
            self._event_bus.publish(StageExecutionSucceed.with_meta(task_id=plan.task_id, step_order=step_index+1, result=stage_summary))
            step_index += 1
            current_replan_stage_attempts = 0
            start_reason = _StartReason.NEW

        raise PipelineError(AGENT_MAX_ITERATIONS_EXCEEDED, "reach max iterations")

    # ------------------------------------------------------------------
    # Stage internal reasoning loop
    # ------------------------------------------------------------------

    def _execute_stage(
        self, stage: Stage, provider_name: str, total_steps: int = 0
    ) -> _StageResult:
        """ReAct reasoning loop for a single stage.

        Returns (outcome, guidance_or_feedback) where guidance is non-empty only
        for NEED_REPLAN outcomes.

        Flow per iteration:
          3.   Poll async user commands (cancel / guidance).
          1.   get_context_window — truncation handled inside ContextManager.
          2.   Call LLM → Decision.
          2.0  Publish LLMResponseGenerated event.
          2.1  FINAL_ANSWER  → complete stage, return SUCCESS.
          2.2  CONTINUE      → inject assistant message, loop.
          2.3  TOOL_CALL     → dispatch tools (with events), loop.
          2.4  CLARIFICATION → publish event, block, inject reply, loop.
          2.5  PAUSED        → publish event, block, resume, loop.
        """
        stage_prompt = self._renderer.render("stage_executor/stage_prompt.j2", {
            "stage": stage,
            "total_steps": total_steps,
        })
        self._context_manager.add_message("user", stage_prompt)
        tool_consecutive_count: int = 0

        while stage.iteration_count < self._max_iterations:
            self._logger.info("Stage inner process started",
                task_id=stage.task_id, step_order=stage.order,
                current_iteration=stage.iteration_count, provider=provider_name)

            # ── 3. Poll async user commands ────────────────────────────────
            user_cmd = self._driver.loop_user_messages(self._agent_poll_timeout)
            if user_cmd is not None:
                self._logger.info("User command received during stage",
                    task_id=stage.task_id, step_order=stage.order,
                    command_type=user_cmd.type.value if hasattr(user_cmd.type, "value") else str(user_cmd.type))
                if user_cmd.type == UserCommandType.CANCEL:
                    self._event_bus.publish(
                        TaskCancelled.with_meta(task_id=stage.task_id, message="Task cancelled by user.")
                    )
                    stage.fail("Cancelled by user.")

                    self._logger.info("User cancel the task", task_id=stage.task_id, step_order=stage.order)
                    return _StageResult(outcome=_StageOutcome.FATAL)
                if user_cmd.type == UserCommandType.GUIDANCE:
                    self._logger.info("Receive guidance from use for this task", task_id=stage.task_id, step_order=stage.order)
                    self._context_manager.add_message("user", user_cmd.content.strip())

            # ── 1. Get context window ──────────────────────────────────────
            try:
                context_window = self._context_manager.get_context_window(provider_name)
                # ── 2. Call LLM ────────────────────────────────────────────
                with self._tracer.start_span("stage.reason_once", "reasoning",
                    task_id=stage.task_id, stage_id=stage.id,
                    iteration=stage.iteration_count, provider=provider_name,
                    message_count=len(context_window.messages),
                    tool_schema_count=len(context_window.tool_schemas or [])) as span:
                    decision = self._reasoning_manager.reason_once(context_window, provider_name)
                    span.add_attributes({
                        "decision_type": decision.decision_type.value if hasattr(decision.decision_type, "value") else str(decision.decision_type),
                        "tool_call_count": len(decision.tool_calls),
                        "has_assistant_message": decision.assistant_message is not None,
                    })
            except LLMNormalizedError as exc:
                self._logger.error("LLM error happened during stage reasoning",
                    task_id=stage.task_id, step_order=stage.order,
                    iteration=stage.iteration_count, provider=provider_name,
                    error_code=exc.code.value if hasattr(exc.code, "value") else str(exc.code),
                    caller_action=exc.caller_action.value if hasattr(exc.caller_action, "value") else str(exc.caller_action),
                    message=exc.message)
                if exc.caller_action == CallerAction.FATAL:
                    stage.fail(f"Fatal LLM error: {exc.message}")
                    return _StageResult(outcome=_StageOutcome.FATAL)
                stage.fail(f"LLM error: {exc.message}")
                return _StageResult(outcome=_StageOutcome.SWITCH_MODEL, llm_error=exc)
            except PipelineError as exc:
                self._logger.error("Pipeline error happened during stage reasoning",
                    task_id=stage.task_id, step_order=stage.order,
                    iteration=stage.iteration_count,
                    error_code=exc.code, message=exc.message)
                stage.fail(f"Agent error: {exc.message}")
                return _StageResult(outcome=_StageOutcome.FATAL)

            # 2.0 publish "LLM reply generated" event
            self._event_bus.publish(
                NextDecisionMade.with_meta(
                    task_id=stage.task_id,
                    decision=decision.decision_type,
                )
            )

            # ── 2.1 Final answer ───────────────────────────────────────────
            if decision.decision_type == NextDecisionType.FINAL_ANSWER:
                stage.increment_iteration()
                stage.complete(decision.answer)
                self._logger.info("Stage final answer produced",
                    task_id=stage.task_id, step_order=stage.order,
                    used_iteration=stage.iteration_count, answer_length=len(decision.answer))
                return _StageResult(outcome=_StageOutcome.SUCCESS)

            # ── 2.2 Continue reasoning ─────────────────────────────────────
            if decision.decision_type == NextDecisionType.CONTINUE:
                tool_consecutive_count = 0
                content = decision.message or (
                    decision.assistant_message.content if decision.assistant_message else ""
                )
                self._context_manager.add_message(
                    "assistant", content,
                    extra_metadata=_non_tool_metadata(decision.assistant_message),
                )
                stage.increment_iteration()
                self._logger.info("Stage continue decision recorded",
                    task_id=stage.task_id, step_order=stage.order,
                    used_iteration=stage.iteration_count, content_length=len(content))
                continue

            # ── 2.3 Tool call ──────────────────────────────────────────────
            if decision.decision_type == NextDecisionType.TOOL_CALL:
                tool_consecutive_count += 1
                if decision.assistant_message:
                    tool_use = _build_tool_use_metadata(decision.assistant_message.metadata)
                    self._context_manager.add_message(
                        decision.assistant_message.role,
                        decision.assistant_message.content,
                        tool_use=tool_use,
                        extra_metadata=_non_tool_metadata(decision.assistant_message),
                    )
                self._dispatch_tool_calls(stage, decision.tool_calls)
                stage.increment_iteration()
                self._logger.info("Stage tool-call decision processed",
                    task_id=stage.task_id, step_order=stage.order,
                    used_iteration=stage.iteration_count, tool_call_count=len(decision.tool_calls))

                if tool_consecutive_count > self._config.get("agent.max_tool_consecutive_count", 15):
                    self._context_manager.add_message(
                        "user",
                        "The number of consecutive tool calls has reached 15. Now, please complete the current step more efficiently",
                    )
                    tool_consecutive_count = 0 

                continue

            # ── 2.4 Clarification needed ───────────────────────────────────
            if decision.decision_type == NextDecisionType.CLARIFICATION_NEEDED:
                tool_consecutive_count = 0
                question = decision.message or "Please provide clarification."
                if decision.assistant_message:
                    self._context_manager.add_message(
                        "assistant", decision.assistant_message.content,
                        extra_metadata=_non_tool_metadata(decision.assistant_message),
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
                has_clarification = (user_cmd is not None) and (user_cmd.type ==UserCommandType.CLARIFICATION)
                clarification = user_cmd.content.strip() if has_clarification else DEFAULT_CLARIFICATION
                self._logger.info(
                    "Receive user's clarification",
                    zap.any("task_id", stage.task_id),
                    zap.any("has_clarification", bool(has_clarification)),
                    zap.any("user_clarification", clarification),
                )
                self._context_manager.add_message(
                    "user", f"Clarification: {clarification}"
                )
                stage.increment_iteration()
                self._logger.info("Stage clarification handled",
                    task_id=stage.task_id, step_order=stage.order, used_iteration=stage.iteration_count)
                continue

            # ── 2.5 Paused ────────────────────────────────────────────────
            if decision.decision_type == NextDecisionType.PAUSED:
                tool_consecutive_count = 0
                reason = decision.message or "Task paused."
                if decision.assistant_message:
                    self._context_manager.add_message(
                        "assistant", decision.assistant_message.content,
                        extra_metadata=_non_tool_metadata(decision.assistant_message),
                    )

                self._event_bus.publish(
                    TaskPaused.with_meta(task_id=stage.task_id, reason=reason)
                )
                stage.pause()
                resume_cmd = self._driver.loop_user_messages(timeout=1)
                if resume_cmd is not None and resume_cmd.type == UserCommandType.RESUME:
                    stage.status = StageStatus.RUNNING
                    stage.increment_iteration()
                    self._logger.info("Stage paused by agent but now resumed",
                        task_id=stage.task_id, step_order=stage.order, used_iteration=stage.iteration_count)
                    continue

        stage.fail(f"Max iterations ({self._max_iterations}) exceeded")
        self._logger.error("Stage exceeded max iterations",
            task_id=stage.task_id, step_order=stage.order, max_iterations=self._max_iterations)
        return _StageResult(outcome=_StageOutcome.SWITCH_MODEL)  # llm_error=None → default cooloff

    def reset(self) -> None:
        self._context_manager.reset()

    # ------------------------------------------------------------------
    # Public helpers used by Pipeline
    # ------------------------------------------------------------------

    def get_current_stage(self) -> Stage | None:
        return self._current_stage

    def archive_current_stage_context(self) -> None:
        """Reset context window for a full task retry, preserving history."""
        self._context_manager.reset()

    def get_conversation_history(self):
        return self._context_manager.get_conversation_history()

    def set_llm_gateway(self, llm_gateway: LLMGateway) -> None:
        self._llm_gateway = llm_gateway
        self._reasoning_manager.set_llm_gateway(self._llm_gateway)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _replan_step(self, step: PlanStep, feedback: str) -> PlanStep:
        task = self._context_manager.get_task()
        plan = self._context_manager.get_plan()
        return self._planner.renew_plan_step(
            task, step, feedback, self._llm_gateway, plan
        )

    def _apply_stage_recovery(
        self,
        action: StageRecoveryAction,
        plan: Plan,
        step_index: int,
        feedback: str,
    ) -> _StageRecoveryResult:
        """根据 LLM 建议的恢复模式清理上下文并更新计划。代价从低到高。"""
        self._logger.info("Applying stage recovery",
            plan_id=plan.id, step_index=step_index, action=action.value, feedback=feedback)
        if action == StageRecoveryAction.RETRY_SAME_STEP:
            self._context_manager.drop_latest_stage_context()
            return _StageRecoveryResult(plan, step_index, _StartReason.EVAL_RETRY, False)

        if action == StageRecoveryAction.REPLAN_THIS_STEP:
            self._context_manager.drop_latest_stage_context()
            step = self._replan_step(plan.step_list[step_index], feedback)
            plan = _replace_step(plan, step_index, step)
            self._context_manager.set_plan(plan)
            return _StageRecoveryResult(plan, step_index, _StartReason.EVAL_RETRY, False)

        if action == StageRecoveryAction.REPLAN_FROM_HERE:
            self._context_manager.drop_stages_from(step_index)
            task = self._context_manager.get_task()
            plan = self._planner.renew_plan_from_step(task, plan, step_index, feedback, self._llm_gateway)
            self._context_manager.set_plan(plan)
            return _StageRecoveryResult(plan, step_index, _StartReason.REPLAN_FROM, False)

        # REPLAN_ALL: 代价最高，清空全部上下文，从 step 0 重新开始
        self._context_manager.reset()
        task = self._context_manager.get_task()
        plan, task = self._planner.renew_plan(task=task, feedback=feedback, llm_api=self._llm_gateway)
        self._context_manager.set_task(task)
        self._context_manager.set_plan(plan)
        self._update_tool_schemas(task)
        return _StageRecoveryResult(plan, 0, _StartReason.REPLAN_ALL, True)

    def _update_tool_schemas(self, task) -> None:
        """Re-filter context_manager tool schemas using updated planner scores."""
        threshold: float = float(self._config.get("planner.tool_score_filter_threshold", 0.65))
        score_map = {m.tool_name: m for m in task.tool_matches}
        filtered_names: list[str] = [
            name for name, m in score_map.items()
            if max(m.match_score, m.planner_score) >= threshold
        ]
        if filtered_names:
            filtered_schemas = self._tool_registry.get_tool_schemas_for(filtered_names)
            self._context_manager.set_tool_schemas(filtered_schemas)
            self._logger.info("Tool schemas updated after replan",
                filtered_count=len(filtered_names), kept_tools=filtered_names)

    def _dispatch_tool_calls(self, stage: Stage, tool_calls: list[ToolCall]) -> None:
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
                )
            )

            rejection = self._check_tool_call(tool_call)
            if rejection is not None:
                self._logger.warning("Tool call rejected before execution",
                    task_id=stage.task_id, stage_id=stage.id,
                    tool_name=tool_call.name,
                    error_code=None if rejection.error is None else rejection.error.code,
                    error_message=None if rejection.error is None else rejection.error.message)
                observation = self._reasoning_manager.format_tool_observation(
                    tool_call=tool_call,
                    result=rejection,
                )
                self._context_manager.add_message(
                    observation.role,
                    observation.content,
                    tool_result=_build_tool_result_metadata(observation.metadata),
                )
                self._event_bus.publish(
                    ToolCallResultProduced.with_meta(
                        task_id=stage.task_id,
                        tool_name=tool_call.name,
                        result=f"← {tool_call.name}: ✗ pre-check failed",
                    )
                )
                continue

            result: ToolResult = self._tool_registry.execute(tool_call)
            observation = self._reasoning_manager.format_tool_observation(
                tool_call=tool_call,
                result=self._tool_result_for_observation(result),
            )
            self._context_manager.add_message(
                observation.role,
                observation.content,
                tool_result=_build_tool_result_metadata(observation.metadata),
            )
            self._event_bus.publish(
                ToolCallResultProduced.with_meta(
                    task_id=stage.task_id,
                    tool_name=tool_call.name,
                    result=f"← {tool_call.name}: {'✓' if result.success else '✗'} {(result.output or '')[:100]}",
                )
            )
            self._logger.info("Tool call result recorded",
                task_id=stage.task_id, stage_id=stage.id,
                tool_name=tool_call.name, success=result.success,
                error_code=None if result.error is None else result.error.code,
                output_length=len(result.output or ""))

    def _check_tool_call(self, tool_call: ToolCall) -> ToolResult | None:
        if not self._tool_registry.has_tool(tool_call.name):
            available = ", ".join(s["name"] for s in self._tool_registry.get_tool_schemas())
            return ToolResult(
                output="",
                llm_raw_tool_call_id=tool_call.llm_raw_tool_call_id,
                success=False,
                error=build_pipeline_error(
                    TOOL_NOT_FOUND,
                    f"Tool '{tool_call.name}' does not exist. Available: {available}.",
                ),
            )

        if self._forbidden_tools and tool_call.name in self._forbidden_tools:
            return ToolResult(
                output="",
                llm_raw_tool_call_id=tool_call.llm_raw_tool_call_id,
                success=False,
                error=build_pipeline_error(
                    TOOL_NOT_FOUND,
                    f"Tool '{tool_call.name}' is forbidden.",
                ),
            )

        missing = self._tool_registry.validate_arguments(tool_call)
        if missing:
            return ToolResult(
                output="",
                llm_raw_tool_call_id=tool_call.llm_raw_tool_call_id,
                success=False,
                error=build_pipeline_error(
                    TOOL_ARGUMENT_ERROR,
                    f"Tool '{tool_call.name}' missing required args: {', '.join(missing)}.",
                ),
            )

        return None

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


# ── Module-level helper ────────────────────────────────────────────────────────

def _fmt_args(args: dict) -> str:
    """Compact argument summary for event content."""
    parts = [f"{k}={str(v)[:25]}" for k, v in list(args.items())[:3]]
    suffix = ", ..." if len(args) > 3 else ""
    return ", ".join(parts) + suffix

def _build_tool_use_metadata(metadata: dict) -> ToolUseMetadata | None:
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


_TOOL_METADATA_KEYS = frozenset({"tool_calls", "tool_calls_count"})


def _non_tool_metadata(msg: LLMMessage | None) -> dict:
    """Return metadata from an assistant LLMMessage, excluding tool-call keys."""
    if not msg or not msg.metadata:
        return {}
    return {k: v for k, v in msg.metadata.items() if k not in _TOOL_METADATA_KEYS}


def _build_tool_result_metadata(observation_metadata: dict) -> ToolResultMetadata:
    """Convert format_tool_observation metadata into a typed ToolResultMetadata."""
    return ToolResultMetadata(
        tool_call_id=observation_metadata.get("llm_raw_tool_call_id") or "",
        tool_name=observation_metadata.get("tool_name", ""),
        success=observation_metadata.get("success", True),
    )


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
