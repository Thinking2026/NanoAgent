from __future__ import annotations

import json
from typing import TYPE_CHECKING

from utils.time.time import now as _time_now
from uuid import uuid4

from infra.observability.tracing.tracer import Tracer
from schemas.event_bus import EventBus
from schemas.ids import PlanId, PlanStepId, TaskId
from schemas.task import (
    Plan,
    PlanStep,
    Task,
)
from schemas.types import LLMMessage, UnifiedLLMRequest
from agent.models.evaluate.quality_evaluator import QualityEvaluator
from agent.events.events import UserClarificationRequested
from utils.log.log import Logger, zap

if TYPE_CHECKING:
    from agent.application.driver import PipelineDriver
    from config.config import ConfigReader
    from llm.llm_gateway import LLMGateway

_MAX_PLAN_RETRIES = 3

_MAKE_PLAN_SYSTEM_PROMPT = """\
You are a planning assistant. Given a task description, produce a structured execution plan.
Return a JSON object with a single key "steps", whose value is an array of step objects.
Each step object must have:
  - "goal": string — one-sentence objective
  - "description": string — detailed explanation of what to do
  - "key_results": array of strings — measurable outcomes that confirm the step is done

Respond with only valid JSON. No markdown fences."""

_RENEW_PLAN_SYSTEM_PROMPT = """\
You are a planning assistant. Revise the given execution plan based on the provided feedback.
Return a JSON object with a single key "steps" in the same format as before.
Respond with only valid JSON. No markdown fences."""

_RENEW_STEP_SYSTEM_PROMPT = """\
You are a planning assistant. Revise the given plan step based on the provided feedback.
Return a JSON object representing a single step with keys: goal, description, key_results.
Respond with only valid JSON. No markdown fences."""

_RENEW_FROM_STEP_SYSTEM_PROMPT = """\
You are a planning assistant. Revise the given steps and all subsequent steps based on feedback.
Return a JSON object with a single key "steps" (array of step objects with goal, description, key_results).
The revised steps must cover the same objectives as the original steps from the given index onward.
Respond with only valid JSON. No markdown fences."""


def _task_context(task: Task) -> str:
    parts = [f"Task: {task.description}"]
    if task.intent:
        parts.append(f"Intent: {task.intent}")
    if task.output_constraints:
        parts.append(f"Output constraints: {task.output_constraints}")
    if task.required_tools:
        parts.append(f"Available tools: {', '.join(task.required_tools)}")
    if task.notes:
        parts.append(f"Notes: {task.notes}")
    if task.related_knowledge_entries:
        knowledge = "\n".join(
            f"  - {e.entry.title}: {e.entry.content}"
            for e in task.related_knowledge_entries
        )
        parts.append(f"Relevant knowledge:\n{knowledge}")
    if task.related_user_preference_entries:
        prefs = "\n".join(
            f"  - {e.entry.content}" for e in task.related_user_preference_entries
        )
        parts.append(f"User preferences:\n{prefs}")
    return "\n".join(parts)


def _parse_steps(content: str) -> list[dict]:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        inner = lines[1:-1] if lines[-1].startswith("```") else lines[1:]
        content = "\n".join(inner)
    data = json.loads(content)
    return data.get("steps", data) if isinstance(data, dict) else data


def _build_plan(task_id: TaskId, raw_steps: list[dict]) -> Plan:
    steps = [
        PlanStep(
            id=PlanStepId(str(uuid4())),
            goal=s.get("goal", ""),
            description=s.get("description", ""),
            order=idx + 1,
            key_results=s.get("key_results", []),
        )
        for idx, s in enumerate(raw_steps)
    ]
    return Plan(
        id=PlanId(str(uuid4())),
        task_id=task_id,
        step_list=steps,
        created_at=_time_now(),
    )


class Planner:
    """Responsible for creating and revising execution plans."""
    def __init__(self, 
            config:ConfigReader, 
            logger:Logger, 
            tracer: Tracer, 
            event_bus: EventBus,
            evaluator: QualityEvaluator):
        self._config = config
        self._logger = logger
        self._tracer = tracer
        self._event_bus = event_bus
        self._evaluator = evaluator

    def set_driver(self, driver: PipelineDriver) -> None:
        self._driver = driver

    def make_plan(
        self,
        task: Task,
        llm_api: LLMGateway,
    ) -> Plan:
        """Generate a plan for *task*, evaluate it, and retry on failure.

        If the evaluator signals that user clarification is needed, a
        UserClarificationRequested event is published and the clarification is
        mocked (simulating a blocking wait on the message queue).
        """
        context = _task_context(task)
        extra_context = ""

        for attempt in range(1, _MAX_PLAN_RETRIES + 1):
            prompt = self._build_make_plan_prompt(context, extra_context)
            plan = self._call_llm_for_plan(task.id, prompt, llm_api)

            report = self._evaluator.evaluate_plan(task, plan, llm_api)

            if report.need_user_clarification:
                # Publish clarification event then mock the user's reply.
                event = UserClarificationRequested(
                    task_id=task.id,
                    order=str(attempt),
                    question=report.clarification_question,
                )
                self._logger.info(
                    "UserClarificationRequested published (mocked)",
                    zap.any("task_id", task.id),
                    zap.any("question", report.clarification_question),
                )
                self._event_bus.publish(event)
                cmd = self._driver.loop_user_messages(timeout=300.0)
                clarification = cmd.content if cmd is not None else ""
                extra_context = f"\nUser clarification: {clarification}"
                continue

            if report.passed:
                self._logger.info(
                    "Plan evaluation passed",
                    zap.any("task_id", task.id),
                    zap.any("plan_id", plan.id),
                    zap.any("attempt", attempt),
                )
                return plan

            self._logger.info(
                "Plan evaluation failed, retrying",
                zap.any("task_id", task.id),
                zap.any("attempt", attempt),
                zap.any("feedback", report.feedback),
            )
            extra_context = f"\nPrevious plan was rejected. Feedback: {report.feedback}"

        self._logger.error(
            "Plan evaluation failed after max retries, returning last plan",
            zap.any("task_id", task.id),
        )
        return plan  # type: ignore[return-value]  # assigned in last loop iteration

    def renew_plan(
        self,
        task: Task,
        feedback: str,
        llm_api: LLMGateway,
    ) -> Plan:
        """Regenerate the full plan for *task* incorporating *feedback*."""
        context = _task_context(task)
        prompt = (
            f"{context}\n\n"
            f"The previous plan was unsatisfactory. Feedback:\n{feedback}\n\n"
            f"Produce a revised execution plan."
        )
        plan = self._call_llm_for_plan(task.id, prompt, llm_api, system=_RENEW_PLAN_SYSTEM_PROMPT)
        self._logger.info(
            "Plan renewed",
            zap.any("task_id", task.id),
            zap.any("plan_id", plan.id),
        )
        return plan

    def renew_plan_step(
        self,
        step: PlanStep,
        feedback: str,
        llm_api: LLMGateway,
    ) -> PlanStep:
        """Regenerate a single *step* incorporating *feedback*."""
        prompt = (
            f"Current step:\n"
            f"  goal: {step.goal}\n"
            f"  description: {step.description}\n"
            f"  key_results: {step.key_results}\n\n"
            f"Feedback: {feedback}\n\n"
            f"Produce a revised step."
        )
        provider = self._config.get("llm.plan_provider", ["deepseek"])[0] if self._config else "deepseek"
        response = llm_api.generate(
            UnifiedLLMRequest(
                messages=[LLMMessage(role="user", content=prompt)],
                system_prompt=_RENEW_STEP_SYSTEM_PROMPT,
            ),
            provider,
        )
        try:
            raw = json.loads(response.assistant_message.content.strip())
        except Exception:
            raw = {}

        revised = PlanStep(
            id=step.id,
            goal=raw.get("goal", step.goal),
            description=raw.get("description", step.description),
            order=step.order,
            key_results=raw.get("key_results", step.key_results),
        )
        self._logger.info(
            "Plan step renewed",
            zap.any("step_id", step.id),
        )
        return revised

    def renew_plan_from_step(
        self,
        plan: Plan,
        from_index: int,
        feedback: str,
        llm_api: LLMGateway,
    ) -> Plan:
        """修订 from_index 及之后的所有 step，from_index 之前的 step 保持不变。

        保留原 step ID（按位置对应），保持 plan.id 不变（是修订而非新计划）。
        LLM 解析失败时返回原 plan（安全 fallback）。
        """
        preserved_steps = list(plan.step_list[:from_index])
        steps_to_revise = plan.step_list[from_index:]

        if not steps_to_revise:
            return plan

        steps_text = "\n".join(
            f"  Step {s.order}: goal={s.goal}, description={s.description}"
            for s in steps_to_revise
        )
        prompt = (
            f"The following steps need to be revised based on feedback.\n"
            f"Steps to revise:\n{steps_text}\n\n"
            f"Feedback: {feedback}\n\n"
            f"Produce revised versions of these steps."
        )
        provider = self._config.get("llm.plan_provider", ["deepseek"])[0] if self._config else "deepseek"
        response = llm_api.generate(
            UnifiedLLMRequest(
                messages=[LLMMessage(role="user", content=prompt)],
                system_prompt=_RENEW_FROM_STEP_SYSTEM_PROMPT,
            ),
            provider,
        )
        try:
            raw_steps = _parse_steps(response.assistant_message.content)
        except Exception as exc:
            self._logger.error(
                "Failed to parse revised steps, returning original plan",
                zap.any("plan_id", plan.id),
                zap.any("from_index", from_index),
                zap.any("error", exc),
            )
            return plan

        base_order = steps_to_revise[0].order
        revised_steps = [
            PlanStep(
                id=steps_to_revise[i].id if i < len(steps_to_revise) else PlanStepId(str(uuid4())),
                goal=s.get("goal", ""),
                description=s.get("description", ""),
                order=base_order + i,
                key_results=s.get("key_results", []),
            )
            for i, s in enumerate(raw_steps)
        ]

        new_plan = Plan(
            id=plan.id,
            task_id=plan.task_id,
            step_list=preserved_steps + revised_steps,
            created_at=plan.created_at,
        )
        self._logger.info(
            "Plan renewed from step",
            zap.any("plan_id", plan.id),
            zap.any("from_index", from_index),
            zap.any("revised_step_count", len(revised_steps)),
        )
        return new_plan

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_make_plan_prompt(self, context: str, extra_context: str) -> str:
        return f"{context}{extra_context}\n\nProduce an execution plan for this task."

    def _call_llm_for_plan(
        self,
        task_id: TaskId,
        prompt: str,
        llm_api: LLMGateway,
        system: str = _MAKE_PLAN_SYSTEM_PROMPT,
    ) -> Plan:
        provider = self._config.get("llm.plan_provider", ["deepseek"])[0] if self._config else "deepseek"
        response = llm_api.generate(
            UnifiedLLMRequest(
                messages=[LLMMessage(role="user", content=prompt)],
                system_prompt=system,
            ),
            provider,
        )
        try:
            raw_steps = _parse_steps(response.assistant_message.content)
        except Exception as exc:
            Logger.get_instance().error(
                "Failed to parse plan from LLM response",
                zap.any("task_id", task_id),
                zap.any("error", exc),
            )
            raw_steps = []
        return _build_plan(task_id, raw_steps)
