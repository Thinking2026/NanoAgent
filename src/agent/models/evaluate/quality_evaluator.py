from __future__ import annotations

import json
from typing import TYPE_CHECKING

from infra.observability.tracing.tracer import Tracer
from schemas.task import EvaluationReport, EvaluationTarget, Plan, PlanStep, StageRecoveryAction, Task, TaskRecoveryAction
from schemas.types import LLMMessage, UnifiedLLMRequest
from utils.time.time import now
from utils.log.log import Logger, zap

if TYPE_CHECKING:
    from config.config import ConfigReader
    from llm.llm_gateway import LLMGateway


class QualityEvaluator:
    """Evaluates task results, stage results, and execution plans via LLM."""
    def __init__(self, config:ConfigReader, logger:Logger, tracer: Tracer):
        self._config = config
        self._logger = logger
        self._tracer = tracer

    def evaluate_plan(
        self,
        task: Task,
        plan: Plan,
        llmgateway: LLMGateway,
    ) -> EvaluationReport:
        steps_text = "\n".join(
            f"  Step {s.order}: goal={s.goal}, description={s.description}, "
            f"key_results={s.key_results}, inputs={s.inputs}, tools={s.required_tools}, "
            f"action_constraints={s.action_constraints}, risks={s.risks}, dependencies={s.dependencies}, "
            f"execution_notes={s.execution_notes}"
            for s in plan.step_list
        )
        prompt = (
            f"Review the following execution plan for the given task.\n"
            f"Task: {task.description}\n"
            f"Task goal: {task.task_goal}\n"
            f"Intent: {task.intent}\n"
            f"Task type: {task.task_type}\n"
            f"Required tools: {task.required_tools}\n"
            f"Output constraints: {task.output_constraints}\n"
            f"Known risks: {[r.description for r in task.risks]}\n"
            f"Plan steps:\n{steps_text}\n\n"
            f"Return a JSON object with:\n"
            f"- passed: boolean (true if the plan is feasible and likely to achieve the task)\n"
            f"- feedback: string (issues and suggestions if not passed, empty string if passed)\n"
            f"- need_user_clarification: boolean (true if the plan cannot proceed without additional user input)\n"
            f"- clarification_question: string (the specific question to ask; empty string if not needed)\n\n"
            f"Respond with only valid JSON."
        )
        provider = self._config.get("llm.quality_provider", ["deepseek"])[0] if self._config else "deepseek"
        try:
            self._logger.info(
                "Evaluating plan",
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
                    UnifiedLLMRequest(messages=[LLMMessage(role="user", content=prompt)]),
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
            self._logger.error("Error occurred while evaluating plan", zap.any("error", exc))
            raise

        self._logger.info(
            "Plan evaluation parsed",
            zap.any("task_id", task.id),
            zap.any("plan_id", plan.id),
            zap.any("passed", passed),
            zap.any("need_user_clarification", need_clarification),
            zap.any("feedback", feedback),
        )
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
        prompt = (
            f"Evaluate whether the following result satisfies the task requirements.\n"
            f"Task: {task.description}\n"
            f"Result: {result}\n\n"
            f"Return a JSON object with:\n"
            f"- passed: boolean\n"
            f"- feedback: string (improvement suggestions if not passed, empty string if passed)\n"
            f"- recovery_action: string — only required when passed is false.\n"
            f"  Choose the lowest-cost option that fits the situation:\n"
            f"  RETRY_SAME_PLAN: execution had a transient issue; the plan itself is sound, retry as-is.\n"
            f"  REPLAN_ALL: the plan itself is fundamentally flawed and must be regenerated.\n"
            f"  Prefer the lowest-cost option. Omit or set to null when passed is true.\n\n"
            f"Respond with only valid JSON."
        )
        provider = self._config.get("llm.quality_provider", ["deepseek"])[0] if self._config else "deepseek"
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
                    UnifiedLLMRequest(messages=[LLMMessage(role="user", content=prompt)]),
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
            self._logger.error("Error occurred while evaluating task result", zap.any("error", exc))
            raise

        self._logger.info(
            "Task result evaluation parsed",
            zap.any("task_id", task.id),
            zap.any("passed", passed),
            zap.any("recovery_action", None if task_recovery is None else task_recovery.value),
            zap.any("feedback", feedback),
        )
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
    ) -> EvaluationReport:
        prompt = (
            f"Evaluate whether the following result achieves the step goal.\n"
            f"Step goal: {step.goal}\n"
            f"Step description: {step.description}\n"
            f"Step key results: {step.key_results}\n"
            f"Step inputs: {step.inputs}\n"
            f"Step required tools: {step.required_tools}\n"
            f"Step constraints: {step.action_constraints}\n"
            f"Step risks/checks: {step.risks}\n"
            f"Result: {result}\n\n"
            f"Return a JSON object with:\n"
            f"- passed: boolean\n"
            f"- feedback: string (improvement suggestions if not passed, empty string if passed)\n"
            f"- recovery_action: string — only required when passed is false.\n"
            f"  Choose the lowest-cost option that fits the situation (ordered by cost, lowest first):\n"
            f"  RETRY_SAME_STEP: execution had a transient error; the plan is fine, just retry.\n"
            f"  REPLAN_THIS_STEP: this step's direction is wrong; revise only this step's plan.\n"
            f"  REPLAN_FROM_HERE: this step's failure invalidates subsequent steps' preconditions; replan from here.\n"
            f"  REPLAN_ALL: the overall plan has a fundamental flaw; regenerate the entire plan.\n"
            f"  Prefer the lowest-cost option. Omit or set to null when passed is true.\n\n"
            f"Respond with only valid JSON."
        )
        provider = self._config.get("llm.quality_provider", ["deepseek"])[0] if self._config else "deepseek"
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
                    UnifiedLLMRequest(messages=[LLMMessage(role="user", content=prompt)]),
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
            self._logger.error("Error occurred while evaluating stage result", zap.any("error", exc))
            raise

        self._logger.info(
            "Stage result evaluation parsed",
            zap.any("step_id", step.id),
            zap.any("passed", passed),
            zap.any("recovery_action", None if stage_recovery is None else stage_recovery.value),
            zap.any("feedback", feedback),
        )
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
