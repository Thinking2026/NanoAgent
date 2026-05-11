from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

from utils.time.time import now as _time_now
import threading
from typing import TYPE_CHECKING 
from uuid import uuid4

from agent.events.events import (
    LLMResponseGenerated,
    StageExecutionStarted,
    StageResultProduced,
    TaskCancelled,
    TaskPaused,
    ToolCallResultProduced,
    ToolCallStarted,
    UserClarificationRequested,
)
from config import ConfigReader
from infra.observability.tracing import Tracer
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
from schemas.ids import PlanStepId, StageId, TaskId
from schemas.task import (
    NextDecisionType,
    Plan,
    PlanStep,
    StageRecoveryAction,
    StageStatus,
)
from schemas.types import LLMMessage, ToolCall, ToolResult, UserCommandType
from utils.log.log import Logger, zap

if TYPE_CHECKING:
    from agent.application.driver import PipelineDriver
    from agent.models.context.context_manager import ContextManager
    from agent.models.evaluate.quality_evaluator import QualityEvaluator
    from agent.models.knowledge.knowledge_loader import KnowledgeLoader
    from agent.models.plan.planner import Planner
    from agent.models.reasoning.reasoning_manager import ReasoningManager
    from llm.llm_gateway import LLMGateway
    from tools.tool_registry import ToolRegistry

from agent.models.context.context_manager import ToolCallEntry, ToolResultMetadata, ToolUseMetadata


# ── Stage start reason labels (shown to user) ─────────────────────────────────

class _StartReason(str, Enum):
    NEW          = "A. 新Stage执行"
    EVAL_RETRY   = "B. Stage执行结果评审不通过，更新Step后重新执行"
    MODEL_SWITCH = "C. 切换模型后重新执行"
    REPLAN       = "D. 执行失败，更新计划后重新执行"
    REPLAN_FROM  = "E. 评审不通过，从当前步骤重新规划后执行"
    REPLAN_ALL   = "F. 评审不通过，重新规划全部步骤后从头执行"


# ── Internal outcome codes from _execute_stage ────────────────────────────────

class _StageOutcome(Enum):
    SUCCESS        = auto()  # stage.complete() was called
    NEED_REPLAN    = auto()  # LLM signalled replan (INTERRUPTED with guidance)
    SWITCH_MODEL   = auto()  # LLMError that warrants a provider switch
    FATAL          = auto()  # cancelled / unrecoverable error


@dataclass
class _StageRecoveryResult:
    """Returned by _apply_stage_recovery; carries updated loop variables."""
    plan: Plan
    step_index: int
    start_reason: _StartReason
    reset_replan_counter: bool  # True only for REPLAN_ALL (restarts from step 0)


@dataclass
class Stage:
    id: StageId
    task_id: TaskId
    plan_step_id: PlanStepId
    plan_step_goal: str
    plan_step_description: str
    plan_step_key_results: list[str] = field(default_factory=list)
    plan_step_inputs: list[str] = field(default_factory=list)
    plan_step_required_tools: list[str] = field(default_factory=list)
    plan_step_constraints: list[str] = field(default_factory=list)
    plan_step_risks: list[str] = field(default_factory=list)
    plan_step_dependencies: list[int] = field(default_factory=list)
    plan_step_execution_notes: str = ""
    plan_step_expected_output: str = ""
    status: StageStatus = StageStatus.RUNNING
    result: str = ""
    iteration_count: int = 0
    started_at: datetime = field(default_factory=_time_now)
    completed_at: datetime | None = None

    def increment_iteration(self) -> None:
        self.iteration_count += 1

    def complete(self, result: str) -> None:
        self.status = StageStatus.COMPLETED
        self.result = result
        self.completed_at = _time_now()

    def fail(self, reason: str = "") -> None:
        self.status = StageStatus.FAILED
        self.result = reason
        self.completed_at = _time_now()

    def pause(self, reason: str = "") -> None:
        self.status = StageStatus.PAUSED
        self.result = reason


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

        self._max_iterations = int(self._config.get("agent.max_attempt_iterations", 60))
        self._max_replan_stage_retries = int(self._config.get("agent.max_replan_stage_retries", 3))
        self._max_stage_eval_retries = int(self._config.get("agent.max_stage_retries", 2))
        self._forbidden_tools: frozenset[str] = frozenset(self._config.get("tools.forbidden_tools", []))

        self._current_stage: Stage | None = None
        self._current_stage_index: int = 0
        self._cancelled = threading.Event()

    def set_driver(self, driver: PipelineDriver) -> None:
        self._driver = driver

    def set_event_bus(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

    def cancel(self) -> None:
        self._cancelled.set()

    # ------------------------------------------------------------------
    # Stage Level loop
    # ------------------------------------------------------------------

    def execute(
        self,
        plan: Plan,
        provider_chain: list[str],
    ) -> str | None:
        """Execute all stages in *plan* and return the final result string.

        Returns None if execution is cancelled, interrupted, or fatally failed.

        Flow per step:
          1.0  Publish StageExecutionStarted with start-reason label.
          1.1  Consider switching back to the highest-priority provider.
          1.2  Run the internal reasoning loop (_execute_stage).
          1.2.1  SUCCESS → evaluate result.
            1.2.1.1  Eval passed → summarise, checkpoint, advance (or deliver).
            1.2.1.2  Eval failed → reset ctx, replan step, retry from 1.
          1.2.2  SWITCH_MODEL → reset ctx, switch provider, retry from 1.
          1.2.3  NEED_REPLAN → reset ctx, replan step, retry from 1.
          1.2.4  FATAL → return None.
        """
        provider_index: int = 0
        step_index: int = 0
        start_reason: _StartReason = _StartReason.NEW
        current_replan_stage_attempts = 0
        self._logger.info(
            "Plan execution started",
            zap.any("plan_id", plan.id),
            zap.any("task_id", plan.task_id),
            zap.any("step_count", len(plan.step_list)),
            zap.any("provider_chain", provider_chain),
        )

        while step_index < len(plan.step_list):
            step = plan.step_list[step_index]

            # ── 1.0 Publish stage-start event ─────────────────────────────
            self._current_stage_index = step_index
            self._current_stage = Stage(
                id=StageId(str(uuid4())),
                task_id=plan.task_id,
                plan_step_id=step.id,
                plan_step_goal=step.goal,
                plan_step_description=step.description,
                plan_step_key_results=step.key_results,
                plan_step_inputs=step.inputs,
                plan_step_required_tools=step.required_tools,
                plan_step_constraints=step.constraints,
                plan_step_risks=step.risks,
                plan_step_dependencies=step.dependencies,
                plan_step_execution_notes=step.execution_notes,
                plan_step_expected_output=step.expected_output,
            )
            self._event_bus.publish(
                StageExecutionStarted(
                    task_id=plan.task_id,
                    order=str(step_index),
                    content=(
                        f"Stage {step_index + 1} 执行开始 [{start_reason.value}]: {step.goal}"
                    ),
                )
            )
            self._logger.info(
                "Stage started",
                zap.any("task_id", plan.task_id),
                zap.any("stage_id", self._current_stage.id),
                zap.any("step_index", step_index),
                zap.any("step_order", step.order),
                zap.any("step_id", step.id),
                zap.any("goal", step.goal),
                zap.any("start_reason", start_reason.value),
                zap.any("provider", provider_chain[provider_index]),
            )

            # ── 1.1 Consider switching back to highest-priority provider ───
            if provider_index > 0 and self._should_use_primary_provider():
                provider_index = 0
                self._logger.info(
                    "Switching back to primary provider",
                    zap.any("provider", provider_chain[provider_index]),
                    zap.any("step_index", step_index),
                )

            # ── 1.2 Run reasoning loop ─────────────────────────────────────
            self._context_manager.begin_stage(step_index, plan_step_order=step.order)
            with self._tracer.start_span(
                "stage.execute",
                "stage",
                {
                    "task_id": plan.task_id,
                    "plan_id": plan.id,
                    "stage_id": self._current_stage.id,
                    "step_index": step_index,
                    "step_order": step.order,
                    "step_id": step.id,
                    "goal": step.goal,
                    "provider": provider_chain[provider_index],
                    "start_reason": start_reason.value,
                },
            ) as span:
                outcome, guidance = self._execute_stage(
                    self._current_stage, provider_chain[provider_index],
                    total_steps=len(plan.step_list),
                )
                span.add_attributes(
                    {
                        "outcome": outcome.name,
                        "iterations": self._current_stage.iteration_count,
                        "result_length": len(self._current_stage.result),
                    }
                )
            self._logger.info(
                "Stage execution outcome",
                zap.any("task_id", plan.task_id),
                zap.any("stage_id", self._current_stage.id),
                zap.any("step_index", step_index),
                zap.any("outcome", outcome.name),
                zap.any("iterations", self._current_stage.iteration_count),
            )

            # ── 1.2.4 Fatal (cancel / unrecoverable) ──────────────────────
            if outcome == _StageOutcome.FATAL:
                self._logger.error(
                    "Stage execution ended fatally",
                    zap.any("task_id", plan.task_id),
                    zap.any("stage_id", self._current_stage.id),
                    zap.any("reason", self._current_stage.result),
                )
                return None

            # ── 1.2.2 Switch model ─────────────────────────────────────────
            if outcome == _StageOutcome.SWITCH_MODEL:
                next_index = provider_index + 1
                if next_index >= len(provider_chain):
                    # 1.2.4 No more providers — unrecoverable
                    raise PipelineError(
                        "LLM_ALL_PROVIDERS_FAILED",
                        f"All providers exhausted at stage {step_index + 1}: {step.goal}",
                    )
                self._context_manager.drop_latest_stage_context()
                provider_index = next_index
                start_reason = _StartReason.MODEL_SWITCH
                self._logger.warning(
                    "Switching provider after stage outcome",
                    zap.any("task_id", plan.task_id),
                    zap.any("step_index", step_index),
                    zap.any("next_provider", provider_chain[provider_index]),
                )
                continue  # retry same step_index

            # ── 1.2.3 Replan step (LLM-signalled) ─────────────────────────
            if outcome == _StageOutcome.NEED_REPLAN:
                self._context_manager.drop_latest_stage_context()
                self._logger.info(
                    "Replanning current step from LLM guidance",
                    zap.any("task_id", plan.task_id),
                    zap.any("step_index", step_index),
                    zap.any("feedback", guidance),
                )
                step = self._replan_step(step, guidance or "")
                plan = _replace_step(plan, step_index, step)
                self._context_manager.set_plan(plan)
                start_reason = _StartReason.REPLAN
                continue  # retry same step_index with updated step

            # ── 1.2.1 Stage succeeded — evaluate result ────────────────────
            assert outcome == _StageOutcome.SUCCESS
            eval_report = self._quality_evaluator.evaluate_stage_result(
                step,
                self._current_stage.result,
                self._llm_gateway,
            )
            self._logger.info(
                "Stage quality evaluation complete",
                zap.any("task_id", plan.task_id),
                zap.any("stage_id", self._current_stage.id),
                zap.any("step_index", step_index),
                zap.any("passed", eval_report.passed),
                zap.any("recovery_action", None if eval_report.recovery_action is None else eval_report.recovery_action.value),
                zap.any("feedback", eval_report.feedback),
            )

            if not eval_report.passed:
                current_replan_stage_attempts += 1
                if current_replan_stage_attempts > self._max_replan_stage_retries:
                    raise PipelineError(
                        "LLM_REPLAN_LIMIT_EXCEEDED",
                        f"Max replan attempts exceeded at stage {step_index + 1}: {step.goal}",
                    )
                action = eval_report.recovery_action or StageRecoveryAction.REPLAN_THIS_STEP
                with self._tracer.start_span(
                    "stage.recovery",
                    "stage",
                    {
                        "task_id": plan.task_id,
                        "plan_id": plan.id,
                        "step_index": step_index,
                        "action": action.value,
                        "attempt": current_replan_stage_attempts,
                    },
                ):
                    recovery = self._apply_stage_recovery(action, plan, step_index, eval_report.feedback)
                plan, step_index, start_reason = recovery.plan, recovery.step_index, recovery.start_reason
                if recovery.reset_replan_counter:
                    current_replan_stage_attempts = 0
                continue

            # ── 1.2.1.1 Eval passed ────────────────────────────────────────
            is_last = step_index == len(plan.step_list) - 1

            stage_summary = (
                f"## 第 {step_index + 1} 步执行结果\n\n"
                f"{self._current_stage.result}"
            )
            self._context_manager.add_message("assistant", stage_summary)

            # Summarise and update context (async LLM summarisation inside end_stage)
            self._context_manager.end_stage(step_index, success=True)
            self._logger.info(
                "Stage completed",
                zap.any("task_id", plan.task_id),
                zap.any("stage_id", self._current_stage.id),
                zap.any("step_index", step_index),
                zap.any("is_last", is_last),
                zap.any("result_length", len(self._current_stage.result)),
            )

            if not is_last:
                self._event_bus.publish(
                    StageResultProduced(
                        task_id=plan.task_id,
                        order=str(step_index),
                        content=(
                            f"Stage {step_index + 1} 执行结果已生成: {self._current_stage.result}"
                        ),
                    )
                )

            # Async checkpoint
            #TODO

            if is_last:
                # 1.2.1.1.4 All stages done — deliver final result
                self._logger.info(
                    "Plan execution completed",
                    zap.any("task_id", plan.task_id),
                    zap.any("plan_id", plan.id),
                    zap.any("final_step_index", step_index),
                )
                return self._current_stage.result

            # 1.2.1.1.3 Advance to next stage
            step_index += 1
            current_replan_stage_attempts = 0
            start_reason = _StartReason.NEW

        raise PipelineError(AGENT_MAX_ITERATIONS_EXCEEDED, "reach max iterations")

    # ------------------------------------------------------------------
    # Stage internal reasoning loop
    # ------------------------------------------------------------------

    def _execute_stage(
        self, stage: Stage, provider_name: str, total_steps: int = 0
    ) -> tuple[_StageOutcome, str]:
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
        stage_prompt_lines = [
            f"## 执行第 {stage.plan_step_id} 步（共 {total_steps} 步）：{stage.plan_step_goal}",
            "",
            f"**目标描述：** {stage.plan_step_description}",
        ]
        if stage.plan_step_key_results:
            stage_prompt_lines.append("")
            stage_prompt_lines.append("**关键产出：**")
            for kr in stage.plan_step_key_results:
                stage_prompt_lines.append(f"- {kr}")
        if stage.plan_step_inputs:
            stage_prompt_lines.append("")
            stage_prompt_lines.append("**必须利用的输入：**")
            for item in stage.plan_step_inputs:
                stage_prompt_lines.append(f"- {item}")
        if stage.plan_step_required_tools:
            stage_prompt_lines.append("")
            stage_prompt_lines.append("**建议/需要使用的工具：**")
            for tool in stage.plan_step_required_tools:
                stage_prompt_lines.append(f"- {tool}")
        if stage.plan_step_constraints:
            stage_prompt_lines.append("")
            stage_prompt_lines.append("**本步骤约束：**")
            for constraint in stage.plan_step_constraints:
                stage_prompt_lines.append(f"- {constraint}")
        if stage.plan_step_risks:
            stage_prompt_lines.append("")
            stage_prompt_lines.append("**风险与检查点：**")
            for risk in stage.plan_step_risks:
                stage_prompt_lines.append(f"- {risk}")
        if stage.plan_step_dependencies:
            stage_prompt_lines.append("")
            stage_prompt_lines.append(
                f"**依赖步骤：** {', '.join(str(i) for i in stage.plan_step_dependencies)}"
            )
        if stage.plan_step_execution_notes:
            stage_prompt_lines.append("")
            stage_prompt_lines.append(f"**执行提示：** {stage.plan_step_execution_notes}")
        stage_prompt_lines.append("")
        stage_prompt_lines.append(
            "请按照上述目标、输入、工具、约束和关键产出完成本步骤。"
        )
        self._context_manager.add_message("user", "\n".join(stage_prompt_lines))
        self._logger.info(
            "Stage prompt added",
            zap.any("task_id", stage.task_id),
            zap.any("stage_id", stage.id),
            zap.any("plan_step_id", stage.plan_step_id),
            zap.any("prompt_length", len("\n".join(stage_prompt_lines))),
        )

        while stage.iteration_count < self._max_iterations:
            self._logger.info(
                "Stage iteration started",
                zap.any("task_id", stage.task_id),
                zap.any("stage_id", stage.id),
                zap.any("iteration", stage.iteration_count),
                zap.any("provider", provider_name),
            )

            # ── 3. Poll async user commands ────────────────────────────────
            user_cmd = self._driver.loop_user_messages(0.1)
            if user_cmd is not None:
                self._logger.info(
                    "User command received during stage",
                    zap.any("task_id", stage.task_id),
                    zap.any("stage_id", stage.id),
                    zap.any("command_type", user_cmd.type.value if hasattr(user_cmd.type, "value") else str(user_cmd.type)),
                )
                if user_cmd.type == UserCommandType.CANCEL:
                    self._cancelled.set()
                    self._event_bus.publish(
                        TaskCancelled(task_id=stage.task_id, content="Task cancelled by user.")
                    )
                    stage.fail("Cancelled by user.")
                    return _StageOutcome.FATAL, ""
                if user_cmd.type == UserCommandType.GUIDANCE:
                    return _StageOutcome.NEED_REPLAN, user_cmd.content or ""

            if self._cancelled.is_set():
                stage.fail("Cancelled.")
                return _StageOutcome.FATAL, ""

            # ── 1. Get context window ──────────────────────────────────────
            try:
                with self._tracer.start_span(
                    "stage.build_context_window",
                    "context",
                    {
                        "task_id": stage.task_id,
                        "stage_id": stage.id,
                        "iteration": stage.iteration_count,
                        "provider": provider_name,
                    },
                ):
                    unified_llm_request = self._context_manager.get_context_window(provider_name)
                # ── 2. Call LLM ────────────────────────────────────────────
                with self._tracer.start_span(
                    "stage.reason_once",
                    "reasoning",
                    {
                        "task_id": stage.task_id,
                        "stage_id": stage.id,
                        "iteration": stage.iteration_count,
                        "provider": provider_name,
                        "message_count": len(unified_llm_request.messages),
                        "tool_schema_count": len(unified_llm_request.tool_schemas or []),
                    },
                ) as span:
                    decision = self._reasoning_manager.reason_once(unified_llm_request, provider_name)
                    span.add_attributes(
                        {
                            "decision_type": decision.decision_type.value if hasattr(decision.decision_type, "value") else str(decision.decision_type),
                            "tool_call_count": len(decision.tool_calls),
                            "has_assistant_message": decision.assistant_message is not None,
                        }
                    )
            except LLMNormalizedError as exc:
                self._logger.error(
                    "LLM error during stage reasoning",
                    zap.any("task_id", stage.task_id),
                    zap.any("stage_id", stage.id),
                    zap.any("iteration", stage.iteration_count),
                    zap.any("provider", provider_name),
                    zap.any("error_code", exc.code.value if hasattr(exc.code, "value") else str(exc.code)),
                    zap.any("caller_action", exc.caller_action.value if hasattr(exc.caller_action, "value") else str(exc.caller_action)),
                    zap.any("message", exc.message),
                )
                if exc.caller_action == CallerAction.FATAL:
                    stage.fail(f"Fatal LLM error: {exc.message}")
                    return _StageOutcome.FATAL, ""
                stage.fail(f"LLM error: {exc.message}")
                return _StageOutcome.SWITCH_MODEL, ""
            except PipelineError as exc:
                self._logger.error(
                    "Pipeline error during stage reasoning",
                    zap.any("task_id", stage.task_id),
                    zap.any("stage_id", stage.id),
                    zap.any("iteration", stage.iteration_count),
                    zap.any("error_code", exc.code),
                    zap.any("message", exc.message),
                )
                stage.fail(f"Agent error: {exc.message}")
                return _StageOutcome.FATAL, ""

            # 2.0 publish "LLM reply generated" event
            self._event_bus.publish(
                LLMResponseGenerated(
                    task_id=stage.task_id,
                    order=str(stage.iteration_count),
                    content=decision.message or (
                        decision.assistant_message.content
                        if decision.assistant_message else ""
                    ),
                )
            )

            # ── 2.1 Final answer ───────────────────────────────────────────
            if decision.decision_type == NextDecisionType.FINAL_ANSWER:
                stage.increment_iteration()
                stage.complete(decision.answer)
                self._logger.info(
                    "Stage final answer produced",
                    zap.any("task_id", stage.task_id),
                    zap.any("stage_id", stage.id),
                    zap.any("iteration", stage.iteration_count),
                    zap.any("answer_length", len(decision.answer)),
                )
                return _StageOutcome.SUCCESS, ""

            # ── 2.2 Continue reasoning ─────────────────────────────────────
            if decision.decision_type == NextDecisionType.CONTINUE:
                content = decision.message or (
                    decision.assistant_message.content if decision.assistant_message else ""
                )
                self._context_manager.add_message("assistant", content)
                stage.increment_iteration()
                self._logger.info(
                    "Stage continue decision recorded",
                    zap.any("task_id", stage.task_id),
                    zap.any("stage_id", stage.id),
                    zap.any("iteration", stage.iteration_count),
                    zap.any("content_length", len(content)),
                )
                continue

            # ── 2.3 Tool call ──────────────────────────────────────────────
            if decision.decision_type == NextDecisionType.TOOL_CALL:
                if decision.assistant_message:
                    tool_use = _build_tool_use_metadata(decision.assistant_message.metadata)
                    self._context_manager.add_message(
                        decision.assistant_message.role,
                        decision.assistant_message.content,
                        tool_use=tool_use,
                    )
                self._dispatch_tool_calls(stage, decision.tool_calls)
                stage.increment_iteration()
                self._logger.info(
                    "Stage tool-call decision processed",
                    zap.any("task_id", stage.task_id),
                    zap.any("stage_id", stage.id),
                    zap.any("iteration", stage.iteration_count),
                    zap.any("tool_call_count", len(decision.tool_calls)),
                )
                continue

            # ── 2.4 Clarification needed ───────────────────────────────────
            if decision.decision_type == NextDecisionType.CLARIFICATION_NEEDED:
                question = decision.message or "Please provide clarification."
                if decision.assistant_message:
                    self._context_manager.add_message(
                        "assistant", decision.assistant_message.content
                    )
                else:
                    self._context_manager.add_message("assistant", question)

                self._event_bus.publish(
                    UserClarificationRequested(
                        task_id=stage.task_id,
                        order=str(stage.iteration_count),
                        question=question,
                        content=question,
                    )
                )
                user_cmd = self._driver.loop_user_messages(timeout=0)
                while user_cmd is None or user_cmd.type != UserCommandType.CLARIFICATION:
                    user_cmd = self._driver.loop_user_messages(timeout=0)
                self._context_manager.add_message(
                    "user", f"Clarification: {user_cmd.content if user_cmd else ''}"
                )
                stage.increment_iteration()
                self._logger.info(
                    "Stage clarification handled",
                    zap.any("task_id", stage.task_id),
                    zap.any("stage_id", stage.id),
                    zap.any("iteration", stage.iteration_count),
                )
                continue

            # ── 2.5 Paused ────────────────────────────────────────────────
            if decision.decision_type == NextDecisionType.PAUSED:
                reason = decision.message or "Task paused."
                if decision.assistant_message:
                    self._context_manager.add_message(
                        "assistant", decision.assistant_message.content
                    )

                self._event_bus.publish(
                    TaskPaused(task_id=stage.task_id, reason=reason, content=reason)
                )
                stage.pause()
                resume_cmd = self._driver.loop_user_messages(timeout=0)
                if resume_cmd is not None and resume_cmd.type == UserCommandType.RESUME:
                    stage.status = StageStatus.RUNNING
                    stage.increment_iteration()
                    self._logger.info(
                        "Stage resumed",
                        zap.any("task_id", stage.task_id),
                        zap.any("stage_id", stage.id),
                        zap.any("iteration", stage.iteration_count),
                    )
                    continue

        stage.fail(f"Max iterations ({self._max_iterations}) exceeded")
        self._logger.error(
            "Stage exceeded max iterations",
            zap.any("task_id", stage.task_id),
            zap.any("stage_id", stage.id),
            zap.any("max_iterations", self._max_iterations),
        )
        return _StageOutcome.SWITCH_MODEL, ""

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

    def _should_use_primary_provider(self) -> bool:
        """Return True when conditions favour switching back to the primary model."""
        return False

    def _replan_step(self, step: PlanStep, feedback: str) -> PlanStep:
        return self._planner.renew_plan_step(
            step, feedback, self._llm_gateway
        )

    def _apply_stage_recovery(
        self,
        action: StageRecoveryAction,
        plan: Plan,
        step_index: int,
        feedback: str,
    ) -> _StageRecoveryResult:
        """根据 LLM 建议的恢复模式清理上下文并更新计划。代价从低到高。"""
        self._logger.info(
            "Applying stage recovery",
            zap.any("plan_id", plan.id),
            zap.any("step_index", step_index),
            zap.any("action", action.value),
            zap.any("feedback", feedback),
        )
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
            plan = self._planner.renew_plan_from_step(plan, step_index, feedback, self._llm_gateway)
            self._context_manager.set_plan(plan)
            return _StageRecoveryResult(plan, step_index, _StartReason.REPLAN_FROM, False)

        # REPLAN_ALL: 代价最高，清空全部上下文，从 step 0 重新开始
        self._context_manager.reset()
        task = self._context_manager.get_task()
        plan = self._planner.renew_plan(task=task, feedback=feedback, llm_api=self._llm_gateway)
        self._context_manager.set_plan(plan)
        return _StageRecoveryResult(plan, 0, _StartReason.REPLAN_ALL, True)

    def _dispatch_tool_calls(self, stage: Stage, tool_calls: list[ToolCall]) -> None:
        for tool_call in tool_calls:
            self._logger.info(
                "Dispatching tool call",
                zap.any("task_id", stage.task_id),
                zap.any("stage_id", stage.id),
                zap.any("iteration", stage.iteration_count),
                zap.any("tool_name", tool_call.name),
                zap.any("argument_keys", list(tool_call.arguments.keys())),
            )
            self._event_bus.publish(
                ToolCallStarted(
                    task_id=stage.task_id,
                    order=str(stage.iteration_count),
                    tool_name=tool_call.name,
                    arguments=dict(tool_call.arguments),
                    content=f"Calling tool: {tool_call.name}",
                )
            )

            rejection = self._check_tool_call(tool_call)
            if rejection is not None:
                self._logger.warning(
                    "Tool call rejected before execution",
                    zap.any("task_id", stage.task_id),
                    zap.any("stage_id", stage.id),
                    zap.any("tool_name", tool_call.name),
                    zap.any("error_code", None if rejection.error is None else rejection.error.code),
                    zap.any("error_message", None if rejection.error is None else rejection.error.message),
                )
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
                    ToolCallResultProduced(
                        task_id=stage.task_id,
                        order=str(stage.iteration_count),
                        tool_name=tool_call.name,
                        content=f"Tool pre-check failed: {tool_call.name}",
                    )
                )
                continue

            result: ToolResult = self._tool_registry.execute(tool_call)

            if not result.success and tool_call.name == "search":
                self._logger.info(
                    "Search tool failed, trying knowledge fallback",
                    zap.any("task_id", stage.task_id),
                    zap.any("stage_id", stage.id),
                    zap.any("query", tool_call.arguments.get("query")),
                )
                fallback = self._knowledge_search_fallback(tool_call)
                if fallback is not None:
                    self._logger.info(
                        "Knowledge fallback produced search result",
                        zap.any("task_id", stage.task_id),
                        zap.any("stage_id", stage.id),
                    )
                    result = fallback

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
                ToolCallResultProduced(
                    task_id=stage.task_id,
                    order=str(stage.iteration_count),
                    tool_name=tool_call.name,
                    content=f"Tool result: {tool_call.name} {'succeeded' if result.success else 'failed'}",
                )
            )
            self._logger.info(
                "Tool call result recorded",
                zap.any("task_id", stage.task_id),
                zap.any("stage_id", stage.id),
                zap.any("tool_name", tool_call.name),
                zap.any("success", result.success),
                zap.any("error_code", None if result.error is None else result.error.code),
                zap.any("output_length", len(result.output or "")),
            )

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

    def _knowledge_search_fallback(self, tool_call: ToolCall) -> ToolResult | None:
        query = str(tool_call.arguments.get("query", "")).strip()
        if not query:
            return None
        try:
            entries = self._knowledge_loader.load(query)
        except Exception as exc:
            self._logger.error(
                "Knowledge fallback failed",
                zap.any("query", query),
                zap.any("error", exc),
            )
            return None
        if not entries:
            return None
        results = [
            {"rank": i + 1, "content": e.content, "tags": list(e.tags)}
            for i, e in enumerate(entries)
        ]
        return ToolResult(
            output=json.dumps(
                {
                    "source": "knowledge_base",
                    "query": query,
                    "result_count": len(results),
                    "results": results,
                },
                ensure_ascii=False,
            ),
            llm_raw_tool_call_id=tool_call.llm_raw_tool_call_id,
            success=True,
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


# ── Module-level helper ────────────────────────────────────────────────────────

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
