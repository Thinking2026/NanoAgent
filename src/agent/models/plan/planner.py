from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

from schemas.errors import AGENT_MAX_ITERATIONS_EXCEEDED, build_pipeline_error
from utils.time.time import now as _time_now
from uuid import uuid4

from infra.observability.tracing.tracer import Tracer
from infra.rendering_engine import Jinja2PromptRenderer, PromptRenderer
from schemas.event_bus import EventBus
from schemas.ids import PlanId, PlanStepId, TaskId
from schemas.task import (
    Plan,
    PlanStep,
    StepInput,
    StepDependency,
    Task,
    ToolMatch,
)
from schemas.types import LLMMessage, UnifiedLLMRequest
from agent.models.evaluate.quality_evaluator import QualityEvaluator
from agent.events.events import UserClarificationRequested
from utils.log.log import Logger, zap

if TYPE_CHECKING:
    from agent.application.driver import PipelineDriver
    from config.config import ConfigReader
    from llm.llm_gateway import LLMGateway



def _parse_plan_response(content: str) -> tuple[list[dict], list[dict]]:
    """Parse LLM plan response. Returns (raw_steps, tool_scores).

    Content is expected to be clean JSON (already repaired by LLMGateway).
    """
    data = json.loads(content)
    if isinstance(data, dict):
        steps = data.get("steps", [])
        tool_scores = data.get("tool_scores", [])
    else:
        steps = data
        tool_scores = []
    return steps, tool_scores


def _parse_steps(content: str) -> list[dict]:
    steps, _ = _parse_plan_response(content)
    return steps


def _apply_planner_scores(task: Task, tool_scores: list[dict]) -> Task:
    """Return a new Task with planner_score applied to each ToolMatch."""
    if not tool_scores:
        return task
    score_map = {
        entry["tool_name"]: float(entry.get("planner_score", 0.0))
        for entry in tool_scores
        if isinstance(entry, dict) and "tool_name" in entry
    }
    if not score_map:
        return task
    updated_matches = [
        ToolMatch(
            tool_name=m.tool_name,
            match_score=m.match_score,
            reasoning=m.reasoning,
            planner_score=score_map.get(m.tool_name, 0.0),
        )
        for m in task.tool_matches
    ]
    return dataclasses.replace(task, tool_matches=updated_matches)


def _string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _parse_step_inputs(value) -> list[StepInput]:
    """Accepts both old list[str] and new list[dict] formats."""
    if value is None:
        return []
    result: list[StepInput] = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, str):
            result.append(StepInput(source="entity", value=item))
        elif isinstance(item, dict):
            step_ref_raw = item.get("step_ref")
            result.append(StepInput(
                source=str(item.get("source", "entity")),
                value=str(item.get("value", "")),
                step_ref=int(step_ref_raw) if step_ref_raw is not None else None,
                constraint_note=str(item.get("constraint_note", "")),
            ))
    return result


def _parse_step_dependencies(value) -> list[StepDependency]:
    """Accepts both old list[int] and new list[dict] formats."""
    if value is None:
        return []
    result: list[StepDependency] = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, (int, float)):
            try:
                result.append(StepDependency(step_order=int(item), depends_on=["output_constraints"]))
            except (TypeError, ValueError):
                continue
        elif isinstance(item, dict):
            try:
                result.append(StepDependency(
                    step_order=int(item.get("step_order", 0)),
                    depends_on=_string_list(item.get("depends_on", [])),
                ))
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
        inputs=_parse_step_inputs(raw.get("inputs", [])),
        required_tools=_string_list(raw.get("required_tools", [])),
        action_constraints=_string_list(raw.get("action_constraints", raw.get("constraints", []))),
        risks=_string_list(raw.get("risks", [])),
        dependencies=_parse_step_dependencies(raw.get("dependencies", [])),
        execution_notes=str(raw.get("execution_notes", "")),
        output_constraints=str(raw.get("output_constraints", raw.get("expected_output", ""))),
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
            evaluator: QualityEvaluator,
            renderer: PromptRenderer | None = None):
        self._config = config
        self._logger = logger
        self._tracer = tracer
        self._event_bus = event_bus
        self._evaluator = evaluator
        self._renderer: PromptRenderer = renderer or Jinja2PromptRenderer()
        self._max_plan_retries = int(self._config.get("planner.max_plan_retries", 3))
        self._loop_msg_timeout = self._config.positive_float(
            "agent.latency.loop_user_message_timeout_seconds", 300.0
        )
        self._driver: PipelineDriver | None = None

    def set_driver(self, driver: PipelineDriver) -> None:
        self._driver = driver

    def make_plan(
        self,
        task: Task,
        llm_api: LLMGateway,
    ) -> tuple[Plan, Task]:
        """Generate a plan for *task*, evaluate it, and retry on failure.

        Returns (plan, task) where task has planner_score applied to each ToolMatch.
        """
        extra_context = ""
        self._logger.info(
            "Make Plan started",
            zap.any("task_id", task.id),
            zap.any("task_type", task.task_type),
            zap.any("max_retries", self._max_plan_retries),
        )

        for attempt in range(1, self._max_plan_retries + 1):
            with self._tracer.start_span(
                "planner.make_plan_attempt",
                "planning",
                {"task_id": task.id, "current_attempt_times": attempt, "has_extra_context": bool(extra_context)},
            ) as span:
                prompt = self._renderer.render("planner/make_plan_user.j2", {
                    "task": task,
                    "extra_context": extra_context,
                })
                system_prompt = self._renderer.render("planner/system_make_plan.j2", {})
                plan, tool_scores = self._call_llm_for_plan(task.id, prompt, llm_api, system=system_prompt)
                task = _apply_planner_scores(task, tool_scores)
                span.add_attributes({"plan_id": plan.id, "step_count": len(plan.step_list)})

                report = self._evaluator.evaluate_plan(task, plan, llm_api)
                span.add_attributes(
                    {
                        "evaluation_passed": report.passed,
                        "need_user_clarification": report.need_user_clarification,
                    }
                )

            if report.need_user_clarification:
                clarification = self._handle_clarification(task, report.clarification_question)
                extra_context = f"\nUser clarification: {clarification}"
                continue

            if report.passed:
                self._logger.info(
                    "Plan evaluation passed",
                    zap.any("task_id", task.id),
                    zap.any("plan_id", plan.id),
                    zap.any("current_attempt_times", attempt),
                )
                return plan, task

            self._logger.info(
                "Plan evaluation failed, retrying",
                zap.any("task_id", task.id),
                zap.any("current_attempt_times", attempt),
                zap.any("feedback", report.feedback),
            )
            extra_context = f"\nPrevious plan was rejected. Feedback: {report.feedback}"

        self._logger.error(
            "Plan evaluation failed after max retries, returning last plan",
            zap.any("task_id", task.id)
        )
        raise build_pipeline_error(AGENT_MAX_ITERATIONS_EXCEEDED, "Exceed max attempts for makeing a plan") 

    def renew_plan(
        self,
        task: Task,
        feedback: str,
        llm_api: LLMGateway,
    ) -> tuple[Plan, Task]:
        """Regenerate the full plan for *task* incorporating *feedback*, with evaluation and retry.

        Returns (plan, task) where task has planner_score applied to each ToolMatch.
        """
        accumulated_feedback = feedback
        self._logger.info(
            "Renew Plan started",
            zap.any("task_id", task.id),
            zap.any("max_retries", self._max_plan_retries),
        )
        for attempt in range(1, self._max_plan_retries + 1):
            with self._tracer.start_span(
                "planner.renew_plan_attempt",
                "planning",
                {"task_id": task.id, "current_attempt_times": attempt},
            ) as span:
                prompt = self._renderer.render("planner/renew_plan_user.j2", {
                    "task": task,
                    "feedback": accumulated_feedback,
                })
                system_prompt = self._renderer.render("planner/system_renew_plan.j2", {})
                plan, tool_scores = self._call_llm_for_plan(task.id, prompt, llm_api, system=system_prompt)
                task = _apply_planner_scores(task, tool_scores)
                span.add_attributes({"plan_id": plan.id, "step_count": len(plan.step_list)})
                report = self._evaluator.evaluate_plan(task, plan, llm_api)
                span.add_attributes({
                    "evaluation_passed": report.passed,
                    "need_user_clarification": report.need_user_clarification,
                })

            if report.need_user_clarification:
                clarification = self._handle_clarification(task, report.clarification_question)
                accumulated_feedback = f"{feedback}\nUser clarification: {clarification}"
                continue

            if report.passed:
                self._logger.info(
                    "Renewed plan evaluation passed",
                    zap.any("task_id", task.id),
                    zap.any("plan_id", plan.id),
                    zap.any("current_attempt_times", attempt),
                )
                return plan, task

            self._logger.info(
                "Renewed plan evaluation failed, retrying",
                zap.any("task_id", task.id),
                zap.any("current_attempt_times", attempt),
                zap.any("feedback", report.feedback),
            )
            accumulated_feedback = f"{feedback}\n\nEvaluation feedback: {report.feedback}"

        self._logger.error(
            "Renewed plan evaluation failed after max retries",
            zap.any("task_id", task.id),
        )
        raise build_pipeline_error(AGENT_MAX_ITERATIONS_EXCEEDED, "Exceed max attempts for renewing a plan")

    def renew_plan_step(
        self,
        task: Task,
        step: PlanStep,
        feedback: str,
        llm_api: LLMGateway,
        plan: Plan | None = None,
    ) -> PlanStep:
        """Regenerate a single *step* incorporating *feedback*, with evaluation and retry."""
        accumulated_feedback = feedback
        provider = self._config.get("llm.plan_provider", ["deepseek"])[0] if self._config else "deepseek"
        self._logger.info(
            "Plan step renewal started",
            zap.any("step_id", step.id),
            zap.any("step_order", step.order),
            zap.any("provider", provider),
            zap.any("max_retries", self._max_plan_retries),
        )
        for attempt in range(1, self._max_plan_retries + 1):
            step_dict = {
                "goal": step.goal,
                "description": step.description,
                "output_constraints": step.output_constraints,
                "key_results": step.key_results,
                "inputs": [dataclasses.asdict(i) for i in step.inputs],
                "required_tools": step.required_tools,
                "action_constraints": step.action_constraints,
                "risks": step.risks,
                "dependencies": [dataclasses.asdict(d) for d in step.dependencies],
                "execution_notes": step.execution_notes,
            }
            prompt = self._renderer.render("planner/renew_step_user.j2", {
                "task": task,
                "step": step,
                "step_dict": step_dict,
                "feedback": accumulated_feedback,
            })
            system_prompt = self._renderer.render("planner/system_renew_step.j2", {})
            with self._tracer.start_span(
                "planner.renew_plan_step_attempt",
                "planning",
                {"step_id": step.id, "step_order": step.order, "provider": provider, "current_attempt_times": attempt},
            ) as span:
                response = llm_api.generate(
                    UnifiedLLMRequest(
                        messages=[LLMMessage(role="user", content=prompt)],
                        system_prompt=system_prompt,
                        json_mode=True,
                        json_required_keys=["goal", "description"],
                    ),
                    provider,
                )
                try:
                    raw = json.loads(response.assistant_message.content)
                except Exception as exc:
                    self._logger.error(
                        "Failed to parse renewed plan step, using original step",
                        zap.any("step_id", step.id),
                        zap.any("error", exc),
                    )
                    raw = {}

                if not raw:
                    raw = {
                        "goal": step.goal,
                        "description": step.description,
                        "output_constraints": step.output_constraints,
                        "key_results": step.key_results,
                        "inputs": [dataclasses.asdict(i) for i in step.inputs],
                        "required_tools": step.required_tools,
                        "action_constraints": step.action_constraints,
                        "risks": step.risks,
                        "dependencies": [dataclasses.asdict(d) for d in step.dependencies],
                        "execution_notes": step.execution_notes,
                    }
                revised = _plan_step_from_raw(raw, step.order, step.id)
                if plan is not None:
                    updated_steps = [revised if s.id == step.id else s for s in plan.step_list]
                else:
                    updated_steps = [revised]
                temp_plan = Plan(
                    id=PlanId(str(uuid4())),
                    task_id=task.id,
                    step_list=updated_steps,
                    created_at=_time_now(),
                )
                report = self._evaluator.evaluate_plan(task, temp_plan, llm_api)
                span.add_attributes({
                    "evaluation_passed": report.passed,
                    "need_user_clarification": report.need_user_clarification,
                })

            if report.need_user_clarification:
                clarification = self._handle_clarification(task, report.clarification_question)
                accumulated_feedback = f"{feedback}\nUser clarification: {clarification}"
                continue

            if report.passed:
                self._logger.info(
                    "Plan step renewed and evaluation passed",
                    zap.any("step_id", step.id),
                    zap.any("current_attempt_times", attempt),
                )
                return revised

            self._logger.info(
                "Plan step evaluation failed, retrying",
                zap.any("step_id", step.id),
                zap.any("current_attempt_times", attempt),
                zap.any("feedback", report.feedback),
            )
            accumulated_feedback = f"{feedback}\n\nEvaluation feedback: {report.feedback}"

        self._logger.error(
            "Plan step evaluation failed after max retries",
            zap.any("step_id", step.id),
        )
        raise build_pipeline_error(AGENT_MAX_ITERATIONS_EXCEEDED, "Exceed max attempts for renewing a plan step")

    def renew_plan_from_step(
        self,
        task: Task,
        plan: Plan,
        from_index: int,
        feedback: str,
        llm_api: LLMGateway,
    ) -> Plan:
        """修订 from_index 及之后的所有 step，from_index 之前的 step 保持不变。

        保留原 step ID（按位置对应），保持 plan.id 不变（是修订而非新计划）。
        LLM 解析失败时返回原 plan（安全 fallback）。评测不过时进行有限次重试。
        """
        preserved_steps = list(plan.step_list[:from_index])
        steps_to_revise = plan.step_list[from_index:]

        if not steps_to_revise:
            return plan

        accumulated_feedback = feedback
        provider = self._config.get("llm.plan_provider", ["deepseek"])[0] if self._config else "deepseek"
        self._logger.info(
            "Plan renewal from step started",
            zap.any("plan_id", plan.id),
            zap.any("from_index", from_index),
            zap.any("steps_to_revise", len(steps_to_revise)),
            zap.any("provider", provider),
            zap.any("max_retries", self._max_plan_retries),
        )

        for attempt in range(1, self._max_plan_retries + 1):
            steps_to_revise_dicts = [
                {
                    "order": s.order,
                    "goal": s.goal,
                    "description": s.description,
                    "output_constraints": s.output_constraints,
                    "key_results": s.key_results,
                    "inputs": [dataclasses.asdict(i) for i in s.inputs],
                    "required_tools": s.required_tools,
                    "action_constraints": s.action_constraints,
                    "risks": s.risks,
                    "dependencies": [dataclasses.asdict(d) for d in s.dependencies],
                    "execution_notes": s.execution_notes,
                }
                for s in steps_to_revise
            ]
            prompt = self._renderer.render("planner/renew_from_step_user.j2", {
                "task": task,
                "preserved_steps": preserved_steps,
                "steps_to_revise": steps_to_revise,
                "steps_to_revise_dicts": steps_to_revise_dicts,
                "feedback": accumulated_feedback,
            })
            system_prompt = self._renderer.render("planner/system_renew_from_step.j2", {})
            with self._tracer.start_span(
                "planner.renew_plan_from_step_attempt",
                "planning",
                {
                    "plan_id": plan.id,
                    "from_index": from_index,
                    "steps_to_revise": len(steps_to_revise),
                    "provider": provider,
                    "current_attempt_times": attempt,
                },
            ) as span:
                response = llm_api.generate(
                    UnifiedLLMRequest(
                        messages=[LLMMessage(role="user", content=prompt)],
                        system_prompt=system_prompt,
                        json_mode=True,
                        json_required_keys=["steps"],
                    ),
                    provider,
                )
                raw_steps = _parse_steps(response.assistant_message.content)

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
                report = self._evaluator.evaluate_plan(task, new_plan, llm_api)
                span.add_attributes({
                    "evaluation_passed": report.passed,
                    "need_user_clarification": report.need_user_clarification,
                    "revised_step_count": len(revised_steps),
                })

            if report.need_user_clarification:
                clarification = self._handle_clarification(task, report.clarification_question)
                accumulated_feedback = f"{feedback}\nUser clarification: {clarification}"
                continue

            if report.passed:
                self._logger.info(
                    "Plan renewed from step, evaluation passed",
                    zap.any("plan_id", plan.id),
                    zap.any("from_index", from_index),
                    zap.any("revised_step_count", len(revised_steps)),
                    zap.any("current_attempt_times", attempt),
                )
                return new_plan

            self._logger.info(
                "Plan-from-step evaluation failed, retrying",
                zap.any("plan_id", plan.id),
                zap.any("current_attempt_times", attempt),
                zap.any("feedback", report.feedback),
            )
            accumulated_feedback = f"{feedback}\n\nEvaluation feedback: {report.feedback}"

        self._logger.error(
            "Plan-from-step evaluation failed after max retries",
            zap.any("plan_id", plan.id),
        )
        raise build_pipeline_error(AGENT_MAX_ITERATIONS_EXCEEDED, "Exceed max attempts for renewing plan from step")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _handle_clarification(self, task: Task, question: str) -> str:
        event = UserClarificationRequested(task_id=task.id, question=question)
        self._logger.info(
            "Request user's clarification start",
            zap.any("task_id", task.id),
            zap.any("question", question),
        )
        self._event_bus.publish(event)
        cmd = self._driver.loop_user_messages(timeout=self._loop_msg_timeout)
        clarification = cmd.content if cmd is not None else ""
        self._logger.info(
            "Receive user's clarification",
            zap.any("task_id", task.id),
            zap.any("user_clarification", clarification),
        )
        return clarification

    def _call_llm_for_plan(
        self,
        task_id: TaskId,
        prompt: str,
        llm_api: LLMGateway,
        system: str = "",
    ) -> tuple[Plan, list[dict]]:
        provider = self._config.get("llm.plan_provider", ["deepseek"])[0] if self._config else "deepseek"
        self._logger.info(
            "Call LLM for making plan",
            zap.any("task_id", task_id),
            zap.any("provider", provider),
            zap.any("prompt_length", len(prompt)),
        )
        response = llm_api.generate(
            UnifiedLLMRequest(
                messages=[LLMMessage(role="user", content=prompt)],
                system_prompt=system,
                json_mode=True,
                json_required_keys=["steps"],
            ),
            provider,
        )
        try:
            raw_steps, tool_scores = _parse_plan_response(response.assistant_message.content)
        except Exception as exc:
            self._logger.error(
                "Failed to parse plan from LLM response",
                zap.any("task_id", task_id),
                zap.any("error", exc),
            )
            raise
        plan = _build_plan(task_id, raw_steps)
        self._logger.info(
            "Get a plan from LLM response",
            zap.any("task_id", task_id),
            zap.any("plan_id", plan.id),
            zap.any("step_count", len(plan.step_list)),
            zap.any("tool_scores_count", len(tool_scores)),
        )
        return plan, tool_scores
