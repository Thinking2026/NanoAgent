from __future__ import annotations

"""Loop state objects for the Stage level loop.

The Stage level loop used to carry six free-floating locals across ~200 lines.
They are collected here so `StageExecutor.execute()` reads as the loop skeleton
and the counter/escalation semantics live in one place.
"""

from dataclasses import dataclass, field
from enum import Enum, auto

from schemas.task import EvaluationReport, Plan, StageRecoveryAction


# ── Stage start reason labels (shown to user) ─────────────────────────────────

class StageStartReason(str, Enum):
    NEW          = "A. New stage execution"
    EVAL_RETRY   = "B. Eval failed — step updated, retrying"
    MODEL_SWITCH = "C. Model switched, retrying"
    REPLAN       = "D. Execution failed — plan updated, retrying"
    REPLAN_FROM  = "E. Eval failed — replanned from current step"
    REPLAN_ALL   = "F. Eval failed — full replan, restarting from step 1"


# ── Internal outcome codes from the reasoning loop ────────────────────────────

class StageOutcome(Enum):
    SUCCESS      = auto()  # stage.complete() was called
    SWITCH_MODEL = auto()  # LLMError that warrants a provider switch
    FATAL        = auto()  # cancelled / unrecoverable error


@dataclass
class StageResult:
    """Returned by the reasoning loop; carries outcome and the raw LLM error."""
    outcome: StageOutcome
    llm_error: object | None = None


@dataclass
class StageRecoveryResult:
    """Returned by _apply_stage_recovery; carries updated loop variables."""
    plan: Plan
    step_index: int
    start_reason: StageStartReason
    reset_replan_counter: bool  # True only for REPLAN_ALL (restarts from step 0)


def escalate_recovery_action(action: StageRecoveryAction) -> StageRecoveryAction:
    """Escalate to the next costlier recovery action when the same failure repeats."""
    escalation = {
        StageRecoveryAction.RETRY_SAME_STEP: StageRecoveryAction.REPLAN_THIS_STEP,
        StageRecoveryAction.REPLAN_THIS_STEP: StageRecoveryAction.REPLAN_FROM_HERE,
        StageRecoveryAction.REPLAN_FROM_HERE: StageRecoveryAction.REPLAN_ALL,
        StageRecoveryAction.REPLAN_ALL: StageRecoveryAction.REPLAN_ALL,
    }
    return escalation.get(action, action)


@dataclass
class StageLoopState:
    """Mutable state carried across iterations of the Stage level loop."""

    step_index: int = 0
    start_reason: StageStartReason = StageStartReason.NEW
    replan_attempts: int = 0          # per-step replan attempts, reset on REPLAN_ALL / pass
    total_replan_count: int = 0       # REPLAN_ALL count across the whole plan
    same_failure_count: int = 0       # consecutive RETRY_SAME_STEP decisions
    correction_feedback: list[str] = field(default_factory=list)

    def resolve_action(self, report: EvaluationReport) -> StageRecoveryAction:
        """Pick the recovery action, escalating when RETRY_SAME_STEP keeps repeating.

        Escalates after 2 consecutive RETRY_SAME_STEP decisions regardless of
        whether the feedback wording changed.
        """
        action = report.recovery_action or StageRecoveryAction.REPLAN_THIS_STEP
        if action == StageRecoveryAction.RETRY_SAME_STEP:
            self.same_failure_count += 1
        else:
            self.same_failure_count = 0
        if self.same_failure_count >= 2:
            return escalate_recovery_action(action)
        return action

    def on_recovery(self, action: StageRecoveryAction, feedback: str) -> None:
        """Update counters after a recovery action has been applied."""
        if action == StageRecoveryAction.REPLAN_ALL:
            self.replan_attempts = 0
            self.total_replan_count += 1
            self.correction_feedback = []
        else:
            self.replan_attempts += 1
            self.correction_feedback.append(feedback)

    def on_stage_passed(self) -> None:
        """Clear per-step retry state after an eval-passed stage."""
        self.correction_feedback = []
        self.replan_attempts = 0
        self.same_failure_count = 0

    def advance(self) -> None:
        """Move to the next step as a fresh execution."""
        self.step_index += 1
        self.start_reason = StageStartReason.NEW

    def apply_recovery(self, recovery: StageRecoveryResult) -> Plan:
        """Adopt the plan/step/reason produced by a recovery action."""
        self.step_index = recovery.step_index
        self.start_reason = recovery.start_reason
        return recovery.plan


@dataclass
class ReactState:
    """Mutable state shared by the reasoning-loop decision handlers."""
    tool_consecutive_count: int = 0
