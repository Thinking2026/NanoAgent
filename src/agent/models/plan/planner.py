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
  - "goal": string — one-sentence objective for this step, starting with an action verb
  - "description": string — concrete execution instructions: what to inspect, compute, call, decide, or produce
  - "expected_output": string — artifact this step produces for downstream steps (data type, format, key fields);
    use "" for the final user-facing delivery step
  - "key_results": array of strings — observable acceptance criteria proving this step is complete
  - "inputs": array of strings — task entities, prior step expected_outputs, files, or knowledge this step uses
  - "required_tools": array of strings — tool names from tool_matches to use in this step; [] if none
  - "constraints": array of strings — hard constraints and important soft preferences for this step
  - "risks": array of strings — risk categories this step must mitigate or verify
  - "dependencies": array of integers — 1-based step orders that must complete before this step; [] if none
  - "execution_notes": string — tactical guidance: tool param bindings, entity values, fallback instructions; "" if none"""

_MAKE_PLAN_SYSTEM_PROMPT = f"""\
## Role
You are an industrial-grade planning engine for an autonomous agent runtime.
Given a fully analyzed Task object, produce a structured, executable plan that is
tool-aware, risk-aware, constraint-respecting, and directly consumable by a stage executor.

## Planning Principles

### Core Identity Fields (description, task_goal, intent)
- Plan for `intent` — the true goal behind the request — not just the surface wording of `description`.
- Every step must advance `task_goal`. If a step does not contribute to `task_goal`, remove it.
- `task_type` tells you the domain; use it to select appropriate verification patterns
  (e.g. data_analysis → include a data-validation step; code_generation → include a test/review step).

### Entities
- Each entity in `entities` is a concrete, normalized input (stock code, date, filename, URL, number).
- Bind entity values directly into step `inputs` and as tool parameter values in `execution_notes`.
- Do not paraphrase entities — use the `value` field verbatim as the tool argument.

### Constraints
- `action_constraints` with `strict: true` are MANDATORY. Encode them in the step's `constraints` list
  and in `execution_notes` so the executor cannot miss them.
- `action_constraints` with `strict: false` are preferences. Preserve them when feasible; note trade-offs
  in `execution_notes` if they conflict with other requirements.

### Tool Assignments
- Use `tool_matches` as the authoritative source of available tools. Do not invent tools not listed there.
- Assign a tool to a step only when `match_score >= 0.7` or the step's objective explicitly requires it.
- For each tool assigned, list its `required_params` in the step's `execution_notes` so the executor
  knows exactly which parameters to populate.
- If no tool covers a required action, note the gap in `risks` and plan a manual/reasoning fallback.

### Risks
- For each item in `risks`, create a mitigation action in the most relevant step:
  - severity "high" → dedicate a separate verification step or add a mandatory check to `key_results`.
  - severity "medium" → add a check to `key_results` and a note in `execution_notes`.
  - severity "low" → add a note in `execution_notes`.
- Cite the risk category in the step's `risks` list so the executor and evaluator can trace it.

### Knowledge and Preferences
- `related_knowledge_entries` contain domain facts (SOPs, terminology, business rules). Incorporate
  relevant knowledge into step `description` and `execution_notes` — do not ignore it.
- `related_user_preference_entries` shape output style (format, language, verbosity). Apply them in
  the final synthesis/delivery step's `constraints` and `execution_notes`.

### Output Constraints
- `output_constraints` defines the required format, length, and language of the final deliverable.
- The last step must explicitly satisfy `output_constraints` in its `goal`, `key_results`, and `constraints`.

## Step Count Calibration
Use `estimated_steps` as the baseline. Apply these modifiers:
- `complexity_level` 1-2 + `reasoning_depth` "single-step reasoning": prefer estimated_steps or fewer.
- `complexity_level` 3 + "multi-step reasoning": estimated_steps ± 1 is acceptable.
- `complexity_level` 4 + "multi-step reasoning": up to estimated_steps + 2 if risks or knowledge require it.
- `confidence` < 0.7: add one early clarification/verification step unless extra_context already resolves it.
- Never add steps purely to match the estimate. Correctness and completeness take priority.

## Output Schema
Return a single JSON object with one key "steps", whose value is an array of step objects.
{_PLAN_STEP_SCHEMA}

## Field Filling Rules

goal: One sentence. Start with an action verb. Example: "Fetch daily closing prices for AAPL from 2024-01-01 to 2024-12-31."

description: 2-5 sentences of concrete instructions. Name the tool, the entity values, and the expected data shape.
  Do not write vague instructions like "analyze the data" — write "compute 30-day rolling average of the closing price column."

expected_output: Describe the artifact this step hands to the next step. Examples:
  "DataFrame with columns [date, close, volume] for AAPL, 252 rows"
  "Python dict mapping ticker to annualized_return float"
  "Markdown table with columns [metric, value] summarizing portfolio performance"
  Use "" only for the final user-facing delivery step.

key_results: 2-4 measurable criteria. Each must be independently verifiable by the evaluator.
  Bad: "data is fetched" — Good: "DataFrame has 252 rows, no NaN in close column, date range matches request"

inputs: List every concrete input by name or value. Include entity values, prior step expected_outputs,
  file paths, and knowledge entry titles. Example: ["AAPL", "2024-01-01", "2024-12-31", "step 1 output: AAPL price DataFrame"]

required_tools: Only tools from tool_matches. Use exact tool_name values. [] if this step is pure reasoning.

constraints: Include all strict action_constraints that apply to this step. Also include output_constraints
  if this is the final step. Example: ["Must use real-time data source", "Output in Traditional Chinese"]

risks: Reference risk categories from the task risks list. Example: ["data_staleness", "missing_tool"]

dependencies: 1-based order numbers. Example: [1, 2] means steps 1 and 2 must complete first.

execution_notes: Tactical detail the executor needs but that does not fit elsewhere. Include:
  - Exact tool parameter bindings: "call get_stock_price(ticker='AAPL', start='2024-01-01', end='2024-12-31')"
  - Fallback if a tool fails: "if get_stock_price fails, use search tool with query 'AAPL historical prices 2024'"
  - Knowledge to apply: "per SOP-001, exclude trading halts from the date range"
  - Preference notes: "user prefers concise bullet-point summaries, avoid tables unless data > 10 rows"

Respond with only valid JSON. No markdown fences."""

_RENEW_PLAN_SYSTEM_PROMPT = f"""\
You are an industrial-grade planning engine. Revise the given execution plan based on the provided feedback.
Keep the revised plan aligned with the analyzed Task object and fix the root cause described in feedback.
Re-read intent, entities, action_constraints, tool_matches, and risks from the task context before planning.
Return a single JSON object with one key "steps".
{_PLAN_STEP_SCHEMA}
Respond with only valid JSON. No markdown fences."""

_RENEW_STEP_SYSTEM_PROMPT = f"""\
You are an industrial-grade planning engine. Revise the given plan step based on the provided feedback.
Keep the step executable and preserve the original objective unless the feedback requires a correction.
Update expected_output if the revised step produces a different artifact than the original.
Return a JSON object representing a single step.
{_PLAN_STEP_SCHEMA}
Respond with only valid JSON. No markdown fences."""

_RENEW_FROM_STEP_SYSTEM_PROMPT = f"""\
You are an industrial-grade planning engine. Revise the given steps and all subsequent steps based on feedback.
The revised steps must cover the same remaining task objectives while fixing downstream dependencies.
Fix any inputs that reference prior steps' expected_outputs to match the revised outputs.
You may add or remove steps as needed to correctly complete the task from this point.
Return only the revised steps (not the preserved steps) as a single JSON object with one key "steps".
{_PLAN_STEP_SCHEMA}
Respond with only valid JSON. No markdown fences."""


def _task_context(task: Task) -> str:
    """Render analyzed Task fields as a structured, instruction-rich prompt section."""
    lines: list[str] = []

    lines.append("## Analyzed Task")
    lines.append(
        "The following sections describe the fully analyzed task. "
        "Each section includes an instruction line telling you how to use that data when planning."
    )

    # Core Identity
    lines.append("\n### Core Identity")
    lines.append(
        "Use these fields to anchor every step to the user's true goal. "
        "Plan for `intent`, not just `description`."
    )
    lines.append(f"- description: {task.description}")
    lines.append(f"- task_goal: {task.task_goal}")
    lines.append(f"- intent: {task.intent}")
    lines.append(f"- task_type: {task.task_type}")

    # Planning Controls
    lines.append("\n### Planning Controls")
    lines.append(
        "Use complexity_level and reasoning_depth to calibrate step count. "
        "Use estimated_steps as a baseline. Apply output_constraints in the final step."
    )
    lines.append(f"- complexity_level: {task.complexity.level}")
    if task.complexity.features:
        lines.append(f"- complexity_features: {task.complexity.features}")
    lines.append(f"- reasoning_depth: {task.reasoning_depth.value}")
    lines.append(f"- estimated_steps: {task.estimated_steps}")
    lines.append(f"- confidence: {task.confidence}")
    if task.output_constraints:
        lines.append(f"- output_constraints: {task.output_constraints}")
    if task.notes:
        lines.append(f"- notes: {task.notes}")

    # Entities
    if task.entities:
        lines.append("\n### Entities")
        lines.append(
            "These are concrete, normalized inputs extracted from the user request. "
            "Bind each entity's `value` directly into step `inputs` and as tool parameter values in `execution_notes`."
        )
        for e in task.entities:
            normalized_note = f" (normalized from '{e.raw}')" if e.normalized and e.raw != e.value else ""
            lines.append(f"- [{e.type}] {e.value}{normalized_note}")

    # Constraints
    if task.action_constraints:
        strict = [c for c in task.action_constraints if c.strict]
        soft = [c for c in task.action_constraints if not c.strict]
        lines.append("\n### Constraints")
        lines.append(
            "STRICT constraints are mandatory — encode them in every relevant step's `constraints` list. "
            "Soft constraints are preferences — preserve them when feasible."
        )
        if strict:
            lines.append("Strict (mandatory):")
            for c in strict:
                lines.append(f"  - [STRICT/{c.source}] {c.description}")
        if soft:
            lines.append("Soft (preferences):")
            for c in soft:
                lines.append(f"  - [soft/{c.source}] {c.description}")

    # Tool Assignments
    if task.tool_matches:
        lines.append("\n### Tool Assignments")
        lines.append(
            "Assign tools to steps based on match_score. "
            "Score >= 0.9: primary tool for that need. "
            "Score 0.7-0.89: use with parameter care. "
            "Score 0.5-0.69: auxiliary only. "
            "Do not use tools not listed here."
        )
        for m in sorted(task.tool_matches, key=lambda x: x.match_score, reverse=True):
            params_str = ", ".join(m.required_params) if m.required_params else "none"
            lines.append(
                f"- {m.tool_name} (score={m.match_score:.2f}) "
                f"params=[{params_str}] — {m.reasoning}"
            )

    # Risks
    if task.risks:
        lines.append("\n### Risks")
        lines.append(
            "For each risk, create a mitigation action in the most relevant step. "
            "High severity → dedicated verification step or mandatory key_result. "
            "Medium → add to key_results. Low → note in execution_notes."
        )
        for r in task.risks:
            lines.append(f"- [{r.severity.upper()}/{r.category}] {r.description}")

    # Knowledge Context
    if task.related_knowledge_entries:
        lines.append("\n### Knowledge Context")
        lines.append(
            "Incorporate relevant knowledge into step descriptions and execution_notes. "
            "Higher confidence entries are more authoritative."
        )
        for ke in task.related_knowledge_entries:
            entry_type = ke.entry.entry_type.value if hasattr(ke.entry.entry_type, "value") else str(ke.entry.entry_type)
            lines.append(f"- [{entry_type} / conf={ke.confidence:.2f}] {ke.entry.title}: {ke.entry.content}")

    # User Preferences
    if task.related_user_preference_entries:
        lines.append("\n### User Preferences")
        lines.append(
            "Apply these preferences in the final delivery step's constraints and execution_notes "
            "to shape output style, format, and language."
        )
        for pe in task.related_user_preference_entries:
            lines.append(f"- [conf={pe.confidence:.2f}] {pe.entry.content}")

    return "\n".join(lines)


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
        expected_output=str(raw.get("expected_output", "")),
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
        self._logger.info(
            "Plan generation started",
            zap.any("task_id", task.id),
            zap.any("task_type", task.task_type),
            zap.any("estimated_steps", task.estimated_steps),
            zap.any("max_retries", _MAX_PLAN_RETRIES),
        )

        for attempt in range(1, _MAX_PLAN_RETRIES + 1):
            with self._tracer.start_span(
                "planner.make_plan_attempt",
                "planning",
                {"task_id": task.id, "attempt": attempt, "has_extra_context": bool(extra_context)},
            ) as span:
                prompt = self._build_make_plan_prompt(context, extra_context)
                plan = self._call_llm_for_plan(task.id, prompt, llm_api)
                span.add_attributes({"plan_id": plan.id, "step_count": len(plan.step_list)})

                report = self._evaluator.evaluate_plan(task, plan, llm_api)
                span.add_attributes(
                    {
                        "evaluation_passed": report.passed,
                        "need_user_clarification": report.need_user_clarification,
                    }
                )

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
            zap.any("last_plan_id", plan.id if "plan" in locals() else None),
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
            f"## Feedback on Previous Plan\n"
            f"{feedback}\n\n"
            f"## Instruction\n"
            f"Produce a completely revised execution plan that addresses the feedback above. "
            f"Re-read intent, entities, action_constraints, tool_matches, and risks from the task context "
            f"before planning. Do not repeat the mistakes identified in the feedback."
        )
        with self._tracer.start_span(
            "planner.renew_plan",
            "planning",
            {"task_id": task.id, "feedback": feedback},
        ) as span:
            plan = self._call_llm_for_plan(task.id, prompt, llm_api, system=_RENEW_PLAN_SYSTEM_PROMPT)
            span.add_attributes({"plan_id": plan.id, "step_count": len(plan.step_list)})
        self._logger.info(
            "Plan renewed",
            zap.any("task_id", task.id),
            zap.any("plan_id", plan.id),
            zap.any("step_count", len(plan.step_list)),
        )
        return plan

    def renew_plan_step(
        self,
        task: Task,
        step: PlanStep,
        feedback: str,
        llm_api: LLMGateway,
    ) -> PlanStep:
        """Regenerate a single *step* incorporating *feedback*."""
        step_json = json.dumps({
            "goal": step.goal,
            "description": step.description,
            "expected_output": step.expected_output,
            "key_results": step.key_results,
            "inputs": step.inputs,
            "required_tools": step.required_tools,
            "constraints": step.constraints,
            "risks": step.risks,
            "dependencies": step.dependencies,
            "execution_notes": step.execution_notes,
        }, ensure_ascii=False, indent=2)
        prompt = (
            f"## Task Context\n"
            f"Goal: {task.task_goal}\n"
            f"Intent: {task.intent}\n\n"
            f"## Current Step (order {step.order})\n"
            f"{step_json}\n\n"
            f"## Feedback\n"
            f"{feedback}\n\n"
            f"## Instruction\n"
            f"Revise the step above to address the feedback. "
            f"Preserve the original objective unless the feedback explicitly requires changing it. "
            f"Keep all fields — do not omit any key. "
            f"Update `expected_output` if the revised step produces a different artifact."
        )
        provider = self._config.get("llm.plan_provider", ["deepseek"])[0] if self._config else "deepseek"
        self._logger.info(
            "Plan step renewal started",
            zap.any("step_id", step.id),
            zap.any("step_order", step.order),
            zap.any("provider", provider),
            zap.any("feedback", feedback),
        )
        with self._tracer.start_span(
            "planner.renew_plan_step",
            "planning",
            {"step_id": step.id, "step_order": step.order, "provider": provider},
        ):
            response = llm_api.generate(
                UnifiedLLMRequest(
                    messages=[LLMMessage(role="user", content=prompt)],
                    system_prompt=_RENEW_STEP_SYSTEM_PROMPT,
                ),
                provider,
            )
            try:
                raw = json.loads(response.assistant_message.content.strip())
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
                "expected_output": step.expected_output,
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
        task: Task,
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

        preserved_summary = (
            "\n".join(
                f"  Step {s.order}: {s.goal} → expected_output: {s.expected_output or '(none)'}"
                for s in preserved_steps
            )
            if preserved_steps
            else "  (none — revising from the first step)"
        )

        steps_to_revise_json = json.dumps(
            [
                {
                    "order": s.order,
                    "goal": s.goal,
                    "description": s.description,
                    "expected_output": s.expected_output,
                    "key_results": s.key_results,
                    "inputs": s.inputs,
                    "required_tools": s.required_tools,
                    "constraints": s.constraints,
                    "risks": s.risks,
                    "dependencies": s.dependencies,
                    "execution_notes": s.execution_notes,
                }
                for s in steps_to_revise
            ],
            ensure_ascii=False,
            indent=2,
        )

        prompt = (
            f"## Task Context\n"
            f"Goal: {task.task_goal}\n"
            f"Intent: {task.intent}\n\n"
            f"## Preserved Steps (do not revise these)\n"
            f"{preserved_summary}\n\n"
            f"## Steps to Revise (starting at order {steps_to_revise[0].order})\n"
            f"{steps_to_revise_json}\n\n"
            f"## Feedback\n"
            f"{feedback}\n\n"
            f"## Instruction\n"
            f"Revise the steps listed above to address the feedback. "
            f"The revised steps must cover the same remaining task objectives. "
            f"Fix downstream dependencies: if a step's `inputs` referenced an earlier step's `expected_output`, "
            f"update those references to match the revised outputs. "
            f"You may add or remove steps as needed to correctly complete the task from this point. "
            f"Return only the revised steps (not the preserved steps). "
            f"Keep all fields — do not omit any key."
        )
        provider = self._config.get("llm.plan_provider", ["deepseek"])[0] if self._config else "deepseek"
        self._logger.info(
            "Plan renewal from step started",
            zap.any("plan_id", plan.id),
            zap.any("from_index", from_index),
            zap.any("steps_to_revise", len(steps_to_revise)),
            zap.any("provider", provider),
        )
        with self._tracer.start_span(
            "planner.renew_plan_from_step",
            "planning",
            {
                "plan_id": plan.id,
                "from_index": from_index,
                "steps_to_revise": len(steps_to_revise),
                "provider": provider,
            },
        ):
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
        parts = [context]

        if extra_context:
            parts.append(f"\n## Additional Context\n{extra_context.strip()}")

        parts.append(
            "\n## Planning Instructions\n"
            "Using the Analyzed Task above, produce the execution plan now. Follow these steps:\n"
            "1. Read `intent` and `task_goal` — every plan step must advance these.\n"
            "2. Read `entities` — bind each entity value into the relevant step's `inputs` "
            "and `execution_notes` as tool parameter values.\n"
            "3. Read `action_constraints` — encode all STRICT constraints into every step they affect. "
            "Preserve soft constraints where feasible.\n"
            "4. Read `tool_matches` — assign tools by score. Do not use tools not listed.\n"
            "5. Read `risks` — create mitigation actions: high severity gets a dedicated check, "
            "medium goes into key_results, low goes into execution_notes.\n"
            "6. Read `related_knowledge_entries` — incorporate relevant knowledge into step descriptions.\n"
            "7. Read `related_user_preference_entries` — apply preferences in the final delivery step.\n"
            "8. Calibrate step count: start from `estimated_steps`, adjust for `complexity_level` "
            "and `reasoning_depth` per the Step Count Calibration rules.\n"
            "9. The final step must satisfy `output_constraints` in its goal, key_results, and constraints.\n"
            "10. Set `expected_output` on every non-final step to describe what it hands to the next step."
        )

        parts.append(
            "\nProduce the best execution plan for this task. "
            "Prefer correctness and completeness over matching the step count estimate exactly."
        )

        return "\n".join(parts)

    def _call_llm_for_plan(
        self,
        task_id: TaskId,
        prompt: str,
        llm_api: LLMGateway,
        system: str = _MAKE_PLAN_SYSTEM_PROMPT,
    ) -> Plan:
        provider = self._config.get("llm.plan_provider", ["deepseek"])[0] if self._config else "deepseek"
        self._logger.info(
            "Calling LLM for plan",
            zap.any("task_id", task_id),
            zap.any("provider", provider),
            zap.any("prompt_length", len(prompt)),
        )
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
            self._logger.error(
                "Failed to parse plan from LLM response",
                zap.any("task_id", task_id),
                zap.any("error", exc),
            )
            raw_steps = []
        plan = _build_plan(task_id, raw_steps)
        self._logger.info(
            "Plan parsed from LLM response",
            zap.any("task_id", task_id),
            zap.any("plan_id", plan.id),
            zap.any("step_count", len(plan.step_list)),
        )
        return plan
