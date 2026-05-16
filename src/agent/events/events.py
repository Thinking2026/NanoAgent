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
    content: str = "plan generation started...\n"

@dataclass
class PlanGenerateSucceed(DomainEvent):
    content: str = "plan generation succeeded...\n"

@dataclass
class PlanGenerateFailed(DomainEvent):
    content: str = "plan generation failed...\n"

# ── 评测 ──────────────────────────────────────────────────────────────────────

@dataclass
class PlanEvaluateStarted(DomainEvent):
    content: str = "plan evaluation started...\n"

@dataclass
class PlanEvaluateSucceed(DomainEvent):
    content: str = "plan evaluation succeeded...\n"

@dataclass
class PlanEvaluateFailed(DomainEvent):
    content: str = "plan evaluation failed...\n"

@dataclass
class TaskResultEvaluateStarted(DomainEvent):
    content: str = "task result evaluation started...\n"

@dataclass
class TaskResultEvaluateSucceed(DomainEvent):
    content: str = "task result evaluation succeeded...\n"

@dataclass
class TaskResultEvaluateFailed(DomainEvent):
    content: str = "task result evaluation failed...\n"

@dataclass
class StageResultEvaluateStarted(DomainEvent):
    content: str = "stage result evaluation started...\n"

@dataclass
class StageResultEvaluateSucceed(DomainEvent):
    content: str = "stage result evaluation succeeded...\n"

@dataclass
class StageResultEvaluateFailed(DomainEvent):
    content: str = "stage result evaluation failed...\n"


# ── 用户交互 ──────────────────────────────────────────────────────────────────

@dataclass
class UserClarificationRequested(DomainEvent):
    content: str = "clarification requested...\n"

@dataclass
class UserClarificationReceived(DomainEvent):
    content: str = "clarification received...\n"

@dataclass
class UserGuidanceReceived(DomainEvent):
    content: str = "user guidance received...\n"

@dataclass
class TaskPaused(DomainEvent):
    content: str = "task paused...\n"

@dataclass
class TaskResumed(DomainEvent):
    content: str = "task resumed...\n"

@dataclass
class TaskCancelled(DomainEvent):
    content: str = "task cancelled...\n"


# ── Task 生命周期 ──────────────────────────────────────────────────────────────

@dataclass
class TaskExecutionStarted(DomainEvent):
    content: str = "task execution started...\n"

@dataclass
class TaskResultEvaluateStarted(DomainEvent):
    content: str = "task result evaluation started...\n"

@dataclass
class TaskExecutionSucceed(DomainEvent):
    content: str = "task execution succeeded...\n"

@dataclass
class TaskExecutionFailed(DomainEvent):
    content: str = "task execution failed...\n"


# ── Stage 生命周期 ─────────────────────────────────────────────────────────────

@dataclass
class StageExecutionStarted(DomainEvent):
    content: str = "stage execution started...\n"

@dataclass
class StageExecutionSucceed(DomainEvent):
    content: str = "stage execution succeeded...\n"

@dataclass
class StageExecutionFailed(DomainEvent):
    content: str = "stage execution failed...\n"

# ── LLM ───────────────────────────────────────────────────────────────────────

@dataclass
class NextDecisionMade(DomainEvent):
    content: str = "next decision made...\n"


# ── 工具调用 ──────────────────────────────────────────────────────────────────

@dataclass
class ToolCallStarted(DomainEvent):
    content: str = "tool call started...\n"

@dataclass
class ToolCallResultProduced(DomainEvent):
    content: str = "tool call result produced...\n"

@dataclass
class ToolCallFailed(DomainEvent):
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

ALL_EVENTS = [
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
    "TaskResultEvaluateSucceed",
    "TaskResultEvaluateFailed",
    "StageResultEvaluateStarted",
    "StageResultEvaluateSucceed",
    "StageResultEvaluateFailed",
]
