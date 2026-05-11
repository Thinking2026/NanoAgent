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


_EVALUATOR_SYSTEM_PROMPT = """\
You are an industrial-grade quality evaluator for an autonomous agent runtime.
Your job is to reason carefully about whether a plan or execution result satisfies
all requirements — both explicit (stated in the task) and implicit (reasonably inferred).

Evaluation principles:
- Evidence-based: every pass/fail verdict must cite specific steps, fields, or result content.
- Causal: explain WHY something satisfies or violates a requirement, not just THAT it does.
- Conservative: when evidence is ambiguous, lean toward flagging the issue rather than passing.
- Proportional: recovery_action cost must match the severity of the failure.

Respond with only valid JSON. No markdown fences."""


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
        steps_lines: list[str] = []
        for s in plan.step_list:
            steps_lines.append(f"\n  [Step {s.order}]")
            steps_lines.append(f"    goal:              {s.goal}")
            steps_lines.append(f"    description:       {s.description}")
            steps_lines.append(f"    output_constraints:{s.output_constraints or '(none)'}")
            steps_lines.append(f"    key_results:       {s.key_results}")
            steps_lines.append(f"    required_tools:    {s.required_tools}")
            steps_lines.append(f"    action_constraints:{s.action_constraints}")
            steps_lines.append(f"    risks:             {s.risks}")
            steps_lines.append(f"    execution_notes:   {s.execution_notes or '(none)'}")
            if s.inputs:
                steps_lines.append("    inputs:")
                for inp in s.inputs:
                    ref = f" (from step {inp.step_ref})" if inp.step_ref is not None else ""
                    note = f" [constraint: {inp.constraint_note}]" if inp.constraint_note else ""
                    steps_lines.append(f"      - [{inp.source}] {inp.value}{ref}{note}")
            if s.dependencies:
                steps_lines.append("    dependencies:")
                for dep in s.dependencies:
                    steps_lines.append(f"      - step {dep.step_order} must complete first, providing: {dep.depends_on}")
        steps_text = "\n".join(steps_lines)

        constraints_text = (
            "\n".join(
                f"  - [{'STRICT' if c.strict else 'soft'}/{c.source}] {c.description}"
                for c in task.action_constraints
            )
            if task.action_constraints else "  (none)"
        )
        entities_text = (
            "\n".join(
                f"  - [{e.type}] {e.value}" + (f" (raw: '{e.raw}')" if e.normalized else "")
                for e in task.entities
            )
            if task.entities else "  (none)"
        )
        risks_text = (
            "\n".join(
                f"  - [{r.severity.upper()}/{r.category}] {r.description}"
                for r in task.risks
            )
            if task.risks else "  (none)"
        )

        prompt = (
            f"## Task Under Evaluation\n"
            f"description:       {task.description}\n"
            f"task_goal:         {task.task_goal}\n"
            f"intent:            {task.intent}\n"
            f"task_type:         {task.task_type}\n"
            f"required_tools:    {task.required_tools}\n"
            f"output_constraints:{task.output_constraints or '(none)'}\n"
            f"\n## Action Constraints\n{constraints_text}\n"
            f"\n## Entities\n{entities_text}\n"
            f"\n## Known Risks\n{risks_text}\n"
            f"\n## Execution Plan ({len(plan.step_list)} steps)\n{steps_text}\n"
            f"\n## Evaluation Instructions\n"
            f"Perform the following three analyses in order. Use them as evidence for your verdict.\n"
            f"\n### Phase 1 — Causal Requirement Coverage\n"
            f"For each requirement below, identify which step(s) satisfy it and explain causally using the template:\n"
            f"  'Because [Step X does A] and [Step Y does B], [requirement Z] is satisfied.'\n"
            f"Requirements to check:\n"
            f"  a) task_goal is achieved by the final step's output\n"
            f"  b) output_constraints are met by the final step's key_results and action_constraints\n"
            f"  c) each STRICT action_constraint is encoded in at least one step's action_constraints\n"
            f"  d) each entity value appears in at least one step's inputs or execution_notes\n"
            f"  e) each HIGH/MEDIUM risk has a mitigation in a step's key_results or risks list\n"
            f"\n### Phase 2 — Forward Trace\n"
            f"Walk the plan from Step 1 to the final step:\n"
            f"  - For each step, identify what it produces given its inputs and description.\n"
            f"  - Verify each step's inputs are available from prior steps or the initial task context.\n"
            f"  - Confirm the final step's output satisfies the task_goal.\n"
            f"  Flag any broken link: a step whose inputs reference a prior step's output that does not match.\n"
            f"\n### Phase 3 — Backward Trace\n"
            f"Start from the final step and work backwards to Step 1:\n"
            f"  - For each step, verify its required inputs are produced by the preceding step(s).\n"
            f"  - Verify Step 1's inputs are satisfiable from the task's initial context (entities, description).\n"
            f"  Flag any step whose required input is not produced by any prior step.\n"
            f"\n## Output\n"
            f"Return a JSON object with exactly these keys:\n"
            f"- \"passed\": boolean — true only if all three phases above show no blocking issues\n"
            f"- \"feedback\": string — if not passed, cite the specific step(s) and requirement(s) that failed "
            f"using causal reasoning; empty string if passed\n"
            f"- \"need_user_clarification\": boolean — true only if the plan cannot proceed without additional "
            f"user input that is not resolvable by replanning\n"
            f"- \"clarification_question\": string — the specific question to ask the user; empty string if not needed\n"
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
                    UnifiedLLMRequest(
                        messages=[LLMMessage(role="user", content=prompt)],
                        system_prompt=_EVALUATOR_SYSTEM_PROMPT,
                        max_tokens=2048,
                        temperature=0.0,
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
        constraints_text = (
            "\n".join(
                f"  - [{'STRICT' if c.strict else 'soft'}/{c.source}] {c.description}"
                for c in task.action_constraints
            )
            if task.action_constraints else "  (none)"
        )
        entities_text = (
            "\n".join(
                f"  - [{e.type}] {e.value}" + (f" (raw: '{e.raw}')" if e.normalized else "")
                for e in task.entities
            )
            if task.entities else "  (none)"
        )
        risks_text = (
            "\n".join(
                f"  - [{r.severity.upper()}/{r.category}] {r.description}"
                for r in task.risks
            )
            if task.risks else "  (none)"
        )

        prompt = (
            f"## Task Under Evaluation\n"
            f"description:       {task.description}\n"
            f"task_goal:         {task.task_goal}\n"
            f"intent:            {task.intent}\n"
            f"task_type:         {task.task_type}\n"
            f"required_tools:    {task.required_tools}\n"
            f"output_constraints:{task.output_constraints or '(none)'}\n"
            f"\n## Action Constraints\n{constraints_text}\n"
            f"\n## Entities (values that must appear in or be addressed by the result)\n{entities_text}\n"
            f"\n## Known Risks (issues the result must not exhibit)\n{risks_text}\n"
            f"\n## Execution Result\n{result}\n"
            f"\n## Evaluation Instructions\n"
            f"Check the result against ALL of the following requirement categories:\n"
            f"\n### Explicit Requirements\n"
            f"  a) task_goal: does the result directly achieve the stated goal?\n"
            f"  b) output_constraints: does the result satisfy every format, length, and content constraint?\n"
            f"  c) STRICT action_constraints: does the result comply with every hard constraint?\n"
            f"  d) entities: are all entity values correctly reflected in the result "
            f"(correct ticker, date range, filename, etc.)?\n"
            f"\n### Implicit Requirements\n"
            f"  e) intent: does the result address the TRUE goal behind the request, not just the surface wording?\n"
            f"  f) completeness: is the result complete, or does it only partially address the task "
            f"(e.g. missing a required section, truncated data, unanswered sub-question)?\n"
            f"  g) risk mitigation: does the result avoid the known risk conditions listed above?\n"
            f"\n### Failure Classification\n"
            f"If the result fails, classify the root cause:\n"
            f"  - Transient execution error (tool timeout, network blip, empty API response): "
            f"the plan is sound, only execution failed → RETRY_SAME_PLAN\n"
            f"  - Fundamental plan flaw (wrong approach, missing required step, wrong tool used, "
            f"constraint systematically violated): the plan must be regenerated → REPLAN_ALL\n"
            f"\n## Output\n"
            f"Return a JSON object with exactly these keys:\n"
            f"- \"passed\": boolean — true only if all explicit and implicit requirements above are satisfied\n"
            f"- \"feedback\": string — if not passed, cite the specific requirement(s) violated and the evidence "
            f"from the result; empty string if passed\n"
            f"- \"recovery_action\": string — only when passed is false:\n"
            f"  RETRY_SAME_PLAN: transient execution failure; plan is sound.\n"
            f"  REPLAN_ALL: plan has a fundamental flaw.\n"
            f"  Omit or set to null when passed is true.\n"
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
                    UnifiedLLMRequest(
                        messages=[LLMMessage(role="user", content=prompt)],
                        system_prompt=_EVALUATOR_SYSTEM_PROMPT,
                        max_tokens=1024,
                        temperature=0.0,
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
        inputs_text = (
            "\n".join(
                f"  - [{inp.source}] {inp.value}"
                + (f" (from step {inp.step_ref})" if inp.step_ref is not None else "")
                + (f" [constraint: {inp.constraint_note}]" if inp.constraint_note else "")
                for inp in step.inputs
            )
            if step.inputs else "  (none)"
        )
        deps_text = (
            "\n".join(
                f"  - step {dep.step_order} must complete first, providing: {dep.depends_on}"
                for dep in step.dependencies
            )
            if step.dependencies else "  (none)"
        )
        key_results_text = (
            "\n".join(f"  - {kr}" for kr in step.key_results)
            if step.key_results else "  (none)"
        )
        action_constraints_text = (
            "\n".join(f"  - {c}" for c in step.action_constraints)
            if step.action_constraints else "  (none)"
        )
        risks_text = (
            "\n".join(f"  - {r}" for r in step.risks)
            if step.risks else "  (none)"
        )

        prompt = (
            f"## Step Under Evaluation\n"
            f"order:             {step.order}\n"
            f"goal:              {step.goal}\n"
            f"description:       {step.description}\n"
            f"output_constraints:{step.output_constraints or '(none)'}\n"
            f"required_tools:    {step.required_tools}\n"
            f"execution_notes:   {step.execution_notes or '(none)'}\n"
            f"\n## Step Inputs\n{inputs_text}\n"
            f"\n## Step Dependencies\n{deps_text}\n"
            f"\n## Action Constraints\n{action_constraints_text}\n"
            f"\n## Key Results (acceptance criteria)\n{key_results_text}\n"
            f"\n## Risks / Checks Required\n{risks_text}\n"
            f"\n## Execution Result\n{result}\n"
            f"\n## Evaluation Instructions\n"
            f"Check the result against ALL of the following requirement categories:\n"
            f"\n### Explicit Requirements\n"
            f"  a) key_results: does the result satisfy EVERY acceptance criterion listed above? "
            f"Check each one individually.\n"
            f"  b) output_constraints: does the result match the required artifact format, data type, "
            f"and key fields described in output_constraints?\n"
            f"  c) action_constraints: does the result comply with every constraint listed?\n"
            f"  d) required_tools: were the correct tools used (infer from result content if not explicit)?\n"
            f"\n### Implicit Requirements\n"
            f"  e) goal alignment: does the result actually achieve the step goal, not just partially?\n"
            f"  f) input consumption: does the result demonstrate that the step inputs were correctly used "
            f"(correct entity values, correct prior step output referenced)?\n"
            f"  g) downstream readiness: does the result produce the artifact described in output_constraints "
            f"in a form that downstream steps can consume? (Check data type, format, completeness.)\n"
            f"  h) risk mitigation: does the result address the risk checks listed above?\n"
            f"  i) execution_notes compliance: if execution_notes specified tool parameters, fallback behavior, "
            f"or knowledge to apply — does the result reflect that guidance?\n"
            f"\n### Failure Classification\n"
            f"If the result fails, classify the root cause using the lowest-cost option that fits:\n"
            f"  RETRY_SAME_STEP:  transient error (tool timeout, empty response, network blip); "
            f"the step plan is correct, just re-execute.\n"
            f"  REPLAN_THIS_STEP: this step's approach is wrong (wrong tool, wrong parameters, wrong logic) "
            f"but prior and subsequent steps are unaffected.\n"
            f"  REPLAN_FROM_HERE: this step's failure invalidates subsequent steps' preconditions "
            f"(e.g. output_constraints not met, so downstream inputs are broken).\n"
            f"  REPLAN_ALL:       the overall plan has a fundamental flaw exposed by this step's failure.\n"
            f"\n## Output\n"
            f"Return a JSON object with exactly these keys:\n"
            f"- \"passed\": boolean — true only if all explicit and implicit requirements above are satisfied\n"
            f"- \"feedback\": string — if not passed, cite the specific requirement(s) violated and the evidence "
            f"from the result; empty string if passed\n"
            f"- \"recovery_action\": string — only when passed is false; choose the lowest-cost option:\n"
            f"  RETRY_SAME_STEP | REPLAN_THIS_STEP | REPLAN_FROM_HERE | REPLAN_ALL\n"
            f"  Omit or set to null when passed is true.\n"
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
                    UnifiedLLMRequest(
                        messages=[LLMMessage(role="user", content=prompt)],
                        system_prompt=_EVALUATOR_SYSTEM_PROMPT,
                        max_tokens=768,
                        temperature=0.0,
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
