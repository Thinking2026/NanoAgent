from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from infra.observability.tracing.tracer import Tracer
from schemas.task import EvaluationReport, EvaluationTarget, Plan, PlanStep, Task
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
            f"  Step {s.order}: goal={s.goal}, description={s.description}"
            for s in plan.step_list
        )
        prompt = (
            f"Review the following execution plan for the given task.\n"
            f"Task: {task.description}\n"
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
            response = llmgateway.generate(
                UnifiedLLMRequest(messages=[LLMMessage(role="user", content=prompt)]),
                provider,
            )
        except Exception as exc:
            self._logger.error("Error occurred while evaluating plan", zap.any("error", exc))
            raise

        passed, feedback, need_clarification, clarification_question = _parse_plan_review(
            response.assistant_message.content
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
            f"- feedback: string (improvement suggestions if not passed, empty string if passed)\n\n"
            f"Respond with only valid JSON."
        )
        provider = self._config.get("llm.quality_provider", ["deepseek"])[0] if self._config else "deepseek"
        try:
            response = llmgateway.generate(
                UnifiedLLMRequest(messages=[LLMMessage(role="user", content=prompt)]),
                provider,
            )
        except Exception as exc:
            self._logger.error("Error occurred while evaluating task result", zap.any("error", exc))
            raise

        passed, feedback = _parse_evaluation(response.assistant_message.content)
        return EvaluationReport(
            target_type=EvaluationTarget.TASK_RESULT,
            target_id=str(task.id),
            passed=passed,
            feedback=feedback,
            evaluated_at=datetime.now(timezone.utc),
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
            f"Result: {result}\n\n"
            f"Return a JSON object with:\n"
            f"- passed: boolean\n"
            f"- feedback: string (improvement suggestions if not passed, empty string if passed)\n\n"
            f"Respond with only valid JSON."
        )
        provider = self._config.get("llm.quality_provider", ["deepseek"])[0] if self._config else "deepseek"
        try:
            response = llmgateway.generate(
                UnifiedLLMRequest(messages=[LLMMessage(role="user", content=prompt)]),
                provider,
            )
        except Exception as exc:
            self._logger.error("Error occurred while evaluating stage result", zap.any("error", exc))
            raise

        passed, feedback = _parse_evaluation(response.assistant_message.content)
        return EvaluationReport(
            target_type=EvaluationTarget.STAGE_RESULT,
            target_id=str(step.id),
            passed=passed,
            feedback=feedback,
            evaluated_at=datetime.now(timezone.utc),
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
