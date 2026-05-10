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

_PLAN_STEP_SCHEMA = """\
Each step object must have exactly these keys:
  - "goal": string — one-sentence objective for this step
  - "description": string — concrete execution instructions, including what to inspect, compute, decide, or produce
  - "key_results": array of strings — observable acceptance criteria proving this step is complete
  - "inputs": array of strings — task fields, prior step outputs, files, entities, or knowledge this step must use
  - "required_tools": array of strings — only tools that should be used in this step, chosen from available/matched tools
  - "constraints": array of strings — hard constraints and important soft preferences that apply to this step
  - "risks": array of strings — known risks this step must mitigate or verify
  - "dependencies": array of integers — 1-based step orders that must be complete before this step, [] if none
  - "execution_notes": string — concise tactical guidance for the executor, "" if none"""

_MAKE_PLAN_SYSTEM_PROMPT = f"""\
You are an expert planning engine for an industrial-grade agent runtime.
Given an analyzed Task object, produce a structured execution plan that is feasible, tool-aware,
risk-aware, and directly executable by a stage executor.

## Planning Principles
- Plan for the user's true goal, not just the surface wording.
- Use every relevant Task field: goal, intent, task type, entities, constraints, tool matches,
  reasoning depth, output constraints, notes, knowledge, preferences, risks, confidence, and estimated steps.
- Preserve all explicit user constraints. Treat strict constraints as mandatory.
- Assign tools only where they are needed. Do not invent unavailable tools.
- Make steps small enough to evaluate independently, but avoid unnecessary micro-steps.
- For low-confidence, ambiguous, or risky tasks, include an early clarification/verification step unless
  the provided clarification context resolves the issue.
- For data freshness or external-world tasks, include verification steps and cite the relevant risk.
- End with a synthesis/delivery step that satisfies output constraints.

## Output Schema
Return a single JSON object with one key "steps", whose value is an array of step objects.
{_PLAN_STEP_SCHEMA}

Respond with only valid JSON. No markdown fences."""

_RENEW_PLAN_SYSTEM_PROMPT = f"""\
You are an expert planning engine. Revise the given execution plan based on the provided feedback.
Keep the revised plan aligned with the analyzed Task object and fix the root cause described in feedback.
Return a single JSON object with one key "steps".
{_PLAN_STEP_SCHEMA}
Respond with only valid JSON. No markdown fences."""

_RENEW_STEP_SYSTEM_PROMPT = f"""\
You are an expert planning engine. Revise the given plan step based on the provided feedback.
Keep the step executable and preserve the original objective unless the feedback requires a correction.
Return a JSON object representing a single step.
{_PLAN_STEP_SCHEMA}
Respond with only valid JSON. No markdown fences."""

_RENEW_FROM_STEP_SYSTEM_PROMPT = f"""\
You are an expert planning engine. Revise the given steps and all subsequent steps based on feedback.
The revised steps must cover the same remaining task objectives while fixing downstream dependencies.
Return a single JSON object with one key "steps".
{_PLAN_STEP_SCHEMA}
Respond with only valid JSON. No markdown fences."""


def _task_context(task: Task) -> str:
    """Render analyzed Task fields with planner-facing semantics."""
    payload = {
        "core": {
            "description": task.description,
            "task_goal": task.task_goal,
            "intent": task.intent,
            "task_type": task.task_type,
            "status": task.status.value if hasattr(task.status, "value") else str(task.status),
        },
        "planning_controls": {
            "complexity_level": task.complexity.level,
            "complexity_features": task.complexity.features,
            "reasoning_depth": task.reasoning_depth.value,
            "estimated_steps": task.estimated_steps,
            "confidence": task.confidence,
            "output_constraints": task.output_constraints,
            "notes": task.notes,
        },
        "entities": [
            {
                "type": e.type,
                "value": e.value,
                "raw": e.raw,
                "normalized": e.normalized,
                "planner_semantics": "Use as concrete inputs and tool parameters when relevant.",
            }
            for e in task.entities
        ],
        "action_constraints": [
            {
                "description": c.description,
                "strict": c.strict,
                "source": c.source,
                "planner_semantics": "Strict constraints are mandatory; soft constraints are preferences to preserve when feasible.",
            }
            for c in task.action_constraints
        ],
        "tool_matches": [
            {
                "tool_name": m.tool_name,
                "match_score": m.match_score,
                "required_params": m.required_params,
                "reasoning": m.reasoning,
                "planner_semantics": "May be assigned to steps when it advances the objective; do not use unmatched tools.",
            }
            for m in task.tool_matches
        ],
        "available_tools": task.required_tools,
        "risks": [
            {
                "category": r.category,
                "description": r.description,
                "severity": r.severity,
                "planner_semantics": "Create verification or mitigation work in the relevant step.",
            }
            for r in task.risks
        ],
        "related_knowledge": [
            {
                "title": e.entry.title,
                "type": e.entry.entry_type.value if hasattr(e.entry.entry_type, "value") else str(e.entry.entry_type),
                "content": e.entry.content,
                "confidence": e.confidence,
            }
            for e in task.related_knowledge_entries
        ],
        "user_preferences": [
            {
                "content": e.entry.content,
                "confidence": e.confidence,
            }
            for e in task.related_user_preference_entries
        ],
    }
    return (
        "Analyzed Task object. Interpret fields as follows: description is the original user request; "
        "task_goal and intent define the desired outcome; entities are concrete inputs; action_constraints "
        "govern behavior; tool_matches/available_tools constrain tool use; risks require mitigation; "
        "output_constraints define the final deliverable.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _parse_steps(content: str) -> list[dict]:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        inner = lines[1:-1] if lines[-1].startswith("```") else lines[1:]
        content = "\n".join(inner)
    data = json.loads(content)
    return data.get("steps", data) if isinstance(data, dict) else data


def _string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _int_list(value) -> list[int]:
    items = value if isinstance(value, list) else []
    result: list[int] = []
    for item in items:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def _plan_step_from_raw(raw: dict, order: int, step_id: PlanStepId | None = None) -> PlanStep:
    return PlanStep(
        id=step_id or PlanStepId(str(uuid4())),
        goal=str(raw.get("goal", "")),
        description=str(raw.get("description", "")),
        order=order,
        key_results=_string_list(raw.get("key_results", [])),
        inputs=_string_list(raw.get("inputs", [])),
        required_tools=_string_list(raw.get("required_tools", [])),
        constraints=_string_list(raw.get("constraints", [])),
        risks=_string_list(raw.get("risks", [])),
        dependencies=_int_list(raw.get("dependencies", [])),
        execution_notes=str(raw.get("execution_notes", "")),
    )


def _build_plan(task_id: TaskId, raw_steps: list[dict]) -> Plan:
    steps = [
        _plan_step_from_raw(s, idx + 1)
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
            f"  key_results: {step.key_results}\n"
            f"  inputs: {step.inputs}\n"
            f"  required_tools: {step.required_tools}\n"
            f"  constraints: {step.constraints}\n"
            f"  risks: {step.risks}\n"
            f"  dependencies: {step.dependencies}\n"
            f"  execution_notes: {step.execution_notes}\n\n"
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

        if not raw:
            raw = {
                "goal": step.goal,
                "description": step.description,
                "key_results": step.key_results,
                "inputs": step.inputs,
                "required_tools": step.required_tools,
                "constraints": step.constraints,
                "risks": step.risks,
                "dependencies": step.dependencies,
                "execution_notes": step.execution_notes,
            }
        revised = _plan_step_from_raw(raw, step.order, step.id)
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
            f"  Step {s.order}: goal={s.goal}, description={s.description}, "
            f"key_results={s.key_results}, inputs={s.inputs}, tools={s.required_tools}, "
            f"constraints={s.constraints}, risks={s.risks}, dependencies={s.dependencies}, "
            f"execution_notes={s.execution_notes}"
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
            _plan_step_from_raw(
                s,
                base_order + i,
                steps_to_revise[i].id if i < len(steps_to_revise) else None,
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
        return (
            f"{context}{extra_context}\n\n"
            "Produce the best execution plan for this task. Calibrate step count using estimated_steps, "
            "complexity_level, and reasoning_depth, but prefer correctness over matching the estimate exactly."
        )

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
