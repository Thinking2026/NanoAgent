from __future__ import annotations

from dataclasses import dataclass, field, fields as _dc_fields
from datetime import datetime
from typing import Any

from utils.time.time import now as _time_now

from schemas.types import UserCommandType


@dataclass
class DomainEvent:
    task_id: str = ""
    user_id: str = ""
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=_time_now)

    @classmethod
    def with_meta(cls, **kwargs) -> "DomainEvent":
        known = {f.name for f in _dc_fields(cls)}
        extra = {k: v for k, v in kwargs.items() if k not in known}
        base_kwargs = {k: v for k, v in kwargs.items() if k in known}
        inst = cls(**base_kwargs)
        inst.metadata.update(extra)
        return inst


# ── 分析 ──────────────────────────────────────────────────────────────────────

@dataclass
class TaskAnalysisStarted(DomainEvent):
    content: str = "task analysis started...\n"

@dataclass
class TaskAnalysisSucceed(DomainEvent):
    content: str = "task analysis succeeded...\n"

@dataclass
class TaskAnalysisFailed(DomainEvent):
    content: str = "task analysis failed...\n"


# ── 计划 ──────────────────────────────────────────────────────────────────────

@dataclass
class PlanGenerateStarted(DomainEvent):
    plan_id: str = ""
    content: str = "plan generation started...\n"

@dataclass
class PlanGenerateSucceed(DomainEvent):
    plan_id: str = ""
    content: str = "plan generation succeeded...\n"

@dataclass
class PlanGenerateFailed(DomainEvent):
    plan_id: str = ""
    content: str = "plan generation failed...\n"

# ── 评测 ──────────────────────────────────────────────────────────────────────

@dataclass
class PlanEvaluateStarted(DomainEvent):
    plan_id: str = ""
    content: str = "plan evaluation started...\n"

@dataclass
class PlanEvaluateSucceed(DomainEvent):
    plan_id: str = ""
    content: str = "plan evaluation succeeded...\n"

@dataclass
class PlanEvaluateFailed(DomainEvent):
    plan_id: str = ""
    content: str = "plan evaluation failed...\n"


# ── 用户交互 ──────────────────────────────────────────────────────────────────

@dataclass
class UserClarificationRequested(DomainEvent):
    question: str = ""
    content: str = "clarification requested...\n"

@dataclass
class UserClarificationReceived(DomainEvent):
    question: str = ""
    content: str = "clarification received...\n"

@dataclass
class UserGuidanceReceived(DomainEvent):
    question: str = ""
    content: str = "user guidance received...\n"

@dataclass
class TaskPaused(DomainEvent):
    reason: str = ""
    content: str = "task paused...\n"

@dataclass
class TaskResumed(DomainEvent):
    reason: str = ""
    content: str = "task resumed...\n"

@dataclass
class TaskCancelled(DomainEvent):
    reason: str = ""
    content: str = "task cancelled...\n"


# ── Task 生命周期 ──────────────────────────────────────────────────────────────

@dataclass
class TaskExecutionStarted(DomainEvent):
    content: str = "task execution started...\n"

@dataclass
class TaskResultEvaluateStarted(DomainEvent):
    content: str = "task result evaluation started...\n"

@dataclass
class TaskResultEvaluatePassed(DomainEvent):
    content: str = "task result evaluation passed...\n"

@dataclass
class TaskResultEvaluateRejected(DomainEvent):
    reason: str = ""
    content: str = "task result evaluation rejected...\n"

@dataclass
class TaskExecutionSucceed(DomainEvent):
    content: str = "task execution succeeded...\n"

@dataclass
class TaskExecutionFailed(DomainEvent):
    content: str = "task execution failed...\n"


# ── Stage 生命周期 ─────────────────────────────────────────────────────────────

@dataclass
class StageExecutionStarted(DomainEvent):
    order: str = ""
    content: str = "stage execution started...\n"

@dataclass
class StageExecutionSucceed(DomainEvent):
    order: str = ""
    result: str = ""
    content: str = "stage execution succeeded...\n"

@dataclass
class StageExecutionFailed(DomainEvent):
    order: str = ""
    error: str = ""
    content: str = "stage execution failed...\n"

# ── LLM ───────────────────────────────────────────────────────────────────────

@dataclass
class NextDecisionMade(DomainEvent):
    order: str = ""
    model: str = ""
    content: str = "next decision made...\n"


# ── 工具调用 ──────────────────────────────────────────────────────────────────

@dataclass
class ToolCallStarted(DomainEvent):
    order: str = ""
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    content: str = "tool call started...\n"

@dataclass
class ToolCallResultProduced(DomainEvent):
    order: str = ""
    tool_name: str = ""
    content: str = "tool call result produced...\n"

@dataclass
class ToolCallFailed(DomainEvent):
    order: str = ""
    tool_name: str = ""
    error: str = ""
    content: str = "tool call failed...\n"

# ── REPLAN相关 ──────────────────────────────────────────────────────────────────

@dataclass
class RePlanStarted(DomainEvent):
    content: str = "replan started...\n"

@dataclass
class RePlanSucceed(DomainEvent):
    content: str = "replan succeeded...\n"

@dataclass
class RePlanFailed(DomainEvent):
    content: str = "replan failed...\n"

@dataclass
class UserCommand(DomainEvent):
    type: UserCommandType = UserCommandType.CANCEL
    user_id: str | None = None

ALL_EVENTS = [
    "DomainEvent",
    "TaskAnalysisStarted",
    "TaskAnalysisSucceed",
    "TaskAnalysisFailed",
    "PlanGenerateStarted",
    "PlanGenerateSucceed",
    "PlanGenerateFailed",
    "PlanEvaluateStarted",
    "PlanEvaluateSucceed",
    "PlanEvaluateFailed",
    "UserClarificationRequested",
    "UserClarificationReceived",
    "UserGuidanceReceived",
    "TaskPaused",
    "TaskResumed",
    "TaskCancelled",
    "TaskExecutionStarted",
    "TaskResultEvaluateStarted",
    "TaskResultEvaluatePassed",
    "TaskResultEvaluateRejected",
    "TaskExecutionSucceed",
    "TaskExecutionFailed",
    "StageExecutionStarted",
    "StageExecutionSucceed",
    "StageExecutionFailed",
    "NextDecisionMade",
    "ToolCallStarted",
    "ToolCallResultProduced",
    "ToolCallFailed",
    "RePlanStarted",
    "RePlanSucceed",
    "RePlanFailed",
    "UserCommand",
]
