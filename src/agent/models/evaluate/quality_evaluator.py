from __future__ import annotations

import json
from typing import TYPE_CHECKING

from agent.events.events import PlanEvaluateFailed, PlanEvaluateStarted, PlanEvaluateSucceed, StageResultEvaluateFailed, StageResultEvaluateStarted, StageResultEvaluateSucceed, TaskResultEvaluateFailed, TaskResultEvaluateStarted, TaskResultEvaluateSucceed
from infra.observability.tracing.tracer import Tracer
from infra.rendering_engine import Jinja2PromptRenderer, PromptRenderer
from schemas.event_bus import EventBus
from schemas.task import EvaluationReport, EvaluationTarget, Plan, PlanStep, StageRecoveryAction, Task, TaskRecoveryAction
from schemas.types import LLMMessage, UnifiedLLMRequest
from utils.time.time import now
from utils.log.log import Logger, zap

if TYPE_CHECKING:
    from config.config import ConfigReader
    from llm.llm_gateway import LLMGateway


class QualityEvaluator:
    """Evaluates task results, stage results, and execution plans via LLM."""
    def __init__(self, config: ConfigReader, logger: Logger, tracer: Tracer, event_bus: EventBus, renderer: PromptRenderer | None = None):
        self._config = config
        self._logger = logger
        self._tracer = tracer
        self._event_bus = event_bus
        self._renderer: PromptRenderer = renderer or Jinja2PromptRenderer()

    def evaluate_plan(
        self,
        task: Task,
        plan: Plan,
        llmgateway: LLMGateway,
    ) -> EvaluationReport:
        prompt = self._renderer.render("quality_evaluator/evaluate_plan.j2", {
            "task": task,
            "plan": plan,
        })
        system_prompt = self._renderer.render("quality_evaluator/system_evaluate_plan.j2", {})
        provider = self._config.get("llm.quality_provider", ["deepseek"])[0] if self._config else "deepseek"
        self._event_bus.publish(PlanEvaluateStarted.with_meta(task_id=task.id))
        try:
            self._logger.info(
                "Call LLM for evaluating plan",
                zap.any("task_id", task.id),
                zap.any("plan_id", plan.id),
                zap.any("step_count", len(plan.step_list)),
                zap.any("provider", provider),
            )
            with self._tracer.start_span(
                "quality.evaluate_plan",
                "evaluation",
                {
                    "task_id": task.id,
                    "plan_id": plan.id,
                    "step_count": len(plan.step_list),
                    "provider": provider,
                },
            ) as span:
                response = llmgateway.generate(
                    UnifiedLLMRequest(
                        messages=[LLMMessage(role="user", content=prompt)],
                        system_prompt=system_prompt,
                        json_mode=True,
                        json_required_keys=["passed", "feedback"],
                    ),
                    provider,
                )
                passed, feedback, need_clarification, clarification_question = _parse_plan_review(
                    response.assistant_message.content
                )
                span.add_attributes(
                    {
                        "passed": passed,
                        "need_user_clarification": need_clarification,
                        "feedback_length": len(feedback),
                    }
                )
        except Exception as exc:
            self._event_bus.publish(PlanEvaluateFailed.with_meta(task_id=task.id, error=exc))
            self._logger.error("Error occurred while evaluating plan", zap.any("error", exc), zap.any("task_id", task.id))
            raise

        self._logger.info(
            "Get evaluation result from LLM",
            zap.any("task_id", task.id),
            zap.any("plan_id", plan.id),
            zap.any("passed", passed),
            zap.any("need_user_clarification", need_clarification),
            zap.any("feedback", feedback),
        )
        self._event_bus.publish(PlanEvaluateSucceed.with_meta(task_id=task.id, passed=passed, feedback=feedback))
        return EvaluationReport(
            target_type=EvaluationTarget.PLAN,
            target_id=str(plan.id),
            passed=passed,
            feedback=feedback,
            evaluated_at=now(),
            need_user_clarification=need_clarification,
            clarification_question=clarification_question,
        )

    def evaluate_task_result(
        self,
        task: Task,
        result: str,
        llmgateway: LLMGateway,
    ) -> EvaluationReport:
        prompt = self._renderer.render("quality_evaluator/evaluate_task_result.j2", {
            "task": task,
            "result": result,
        })
        system_prompt = self._renderer.render("quality_evaluator/system_evaluate_task_result.j2", {})
        provider = self._config.get("llm.quality_provider", ["deepseek"])[0] if self._config else "deepseek"
        self._event_bus.publish(TaskResultEvaluateStarted.with_meta(task_id=task.id))
        try:
            self._logger.info(
                "Evaluating task result",
                zap.any("task_id", task.id),
                zap.any("result_length", len(result)),
                zap.any("provider", provider),
            )
            with self._tracer.start_span(
                "quality.evaluate_task_result",
                "evaluation",
                {"task_id": task.id, "result_length": len(result), "provider": provider},
            ) as span:
                response = llmgateway.generate(
                    UnifiedLLMRequest(
                        messages=[LLMMessage(role="user", content=prompt)],
                        system_prompt=system_prompt,
                        json_mode=True,
                        json_required_keys=["passed", "feedback"],
                    ),
                    provider,
                )
                passed, feedback, task_recovery = _parse_task_evaluation(response.assistant_message.content)
                span.add_attributes(
                    {
                        "passed": passed,
                        "feedback_length": len(feedback),
                        "recovery_action": None if task_recovery is None else task_recovery.value,
                    }
                )
        except Exception as exc:
            self._event_bus.publish(TaskResultEvaluateFailed.with_meta(task_id=task.id, error=exc))
            self._logger.error("Error occurred while evaluating task result", zap.any("error", exc))
            raise

        self._logger.info(
            "Task result evaluation parsed",
            zap.any("task_id", task.id),
            zap.any("passed", passed),
            zap.any("recovery_action", None if task_recovery is None else task_recovery.value),
            zap.any("feedback", feedback),
        )
        self._event_bus.publish(TaskResultEvaluateSucceed.with_meta(task_id=task.id, passed=passed, feedback=feedback))
        return EvaluationReport(
            target_type=EvaluationTarget.TASK_RESULT,
            target_id=str(task.id),
            passed=passed,
            feedback=feedback,
            evaluated_at=now(),
            recovery_action=task_recovery,
        )

    def evaluate_stage_result(
        self,
        step: PlanStep,
        result: str,
        llmgateway: LLMGateway,
        task_goal: str = "",
        task_intent: str = "",
    ) -> EvaluationReport:
        prompt = self._renderer.render("quality_evaluator/evaluate_stage_result.j2", {
            "step": step,
            "result": result,
            "task_goal": task_goal,
            "task_intent": task_intent,
        })
        system_prompt = self._renderer.render("quality_evaluator/system_evaluate_stage_result.j2", {})
        provider = self._config.get("llm.quality_provider", ["deepseek"])[0] if self._config else "deepseek"
        self._event_bus.publish(StageResultEvaluateStarted.with_meta(order=step.order))
        try:
            self._logger.info(
                "Evaluating stage result",
                zap.any("step_id", step.id),
                zap.any("step_order", step.order),
                zap.any("result_length", len(result)),
                zap.any("provider", provider),
            )
            with self._tracer.start_span(
                "quality.evaluate_stage_result",
                "evaluation",
                {
                    "step_id": step.id,
                    "step_order": step.order,
                    "result_length": len(result),
                    "provider": provider,
                },
            ) as span:
                response = llmgateway.generate(
                    UnifiedLLMRequest(
                        messages=[LLMMessage(role="user", content=prompt)],
                        system_prompt=system_prompt,
                        json_mode=True,
                        json_required_keys=["passed", "feedback"],
                    ),
                    provider,
                )
                passed, feedback, stage_recovery = _parse_stage_evaluation(response.assistant_message.content)
                span.add_attributes(
                    {
                        "passed": passed,
                        "feedback_length": len(feedback),
                        "recovery_action": None if stage_recovery is None else stage_recovery.value,
                    }
                )
        except Exception as exc:
            self._event_bus.publish(StageResultEvaluateFailed.with_meta(order=step.order, error=exc))
            self._logger.error("Error occurred while evaluating stage result", zap.any("error", exc))
            raise

        self._logger.info(
            "Stage result evaluation parsed",
            zap.any("step_id", step.id),
            zap.any("passed", passed),
            zap.any("recovery_action", None if stage_recovery is None else stage_recovery.value),
            zap.any("feedback", feedback),
        )
        self._event_bus.publish(StageResultEvaluateSucceed.with_meta(order=step.order, passed=passed, feedback=feedback))
        return EvaluationReport(
            target_type=EvaluationTarget.STAGE_RESULT,
            target_id=str(step.id),
            passed=passed,
            feedback=feedback,
            evaluated_at=now(),
            recovery_action=stage_recovery,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_plan_review(content: str) -> tuple[bool, str, bool, str]:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        inner = lines[1:-1] if lines[-1].startswith("```") else lines[1:]
        content = "\n".join(inner)
    try:
        data = json.loads(content)
        passed = bool(data.get("passed", False))
        feedback = str(data.get("feedback", ""))
        need_clarification = bool(data.get("need_user_clarification", False))
        clarification_question = str(data.get("clarification_question", ""))
        return passed, feedback, need_clarification, clarification_question
    except Exception:
        return True, "", False, ""


def _parse_evaluation(content: str) -> tuple[bool, str]:
    """Used by evaluate_plan — no recovery_action needed."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        inner = lines[1:-1] if lines[-1].startswith("```") else lines[1:]
        content = "\n".join(inner)
    try:
        data = json.loads(content)
        passed = bool(data.get("passed", False))
        feedback = str(data.get("feedback", ""))
        return passed, feedback
    except Exception:
        return True, ""


def _parse_stage_evaluation(content: str) -> tuple[bool, str, StageRecoveryAction | None]:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        inner = lines[1:-1] if lines[-1].startswith("```") else lines[1:]
        content = "\n".join(inner)
    try:
        data = json.loads(content)
        passed = bool(data.get("passed", False))
        feedback = str(data.get("feedback", ""))
        recovery: StageRecoveryAction | None = None
        if not passed:
            try:
                recovery = StageRecoveryAction(data.get("recovery_action"))
            except (ValueError, TypeError):
                recovery = StageRecoveryAction.REPLAN_THIS_STEP
        return passed, feedback, recovery
    except Exception:
        return True, "", None


def _parse_task_evaluation(content: str) -> tuple[bool, str, TaskRecoveryAction | None]:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        inner = lines[1:-1] if lines[-1].startswith("```") else lines[1:]
        content = "\n".join(inner)
    try:
        data = json.loads(content)
        passed = bool(data.get("passed", False))
        feedback = str(data.get("feedback", ""))
        recovery: TaskRecoveryAction | None = None
        if not passed:
            try:
                recovery = TaskRecoveryAction(data.get("recovery_action"))
            except (ValueError, TypeError):
                recovery = TaskRecoveryAction.REPLAN_ALL
        return passed, feedback, recovery
    except Exception:
        return True, "", None
