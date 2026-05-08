from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from utils.time.time import now as _time_now

from schemas.types import UserCommandType


@dataclass(frozen=True)
class DomainEvent:
    content:str = ""
    occurred_at: datetime = field(default_factory=_time_now)


# ── 分析 ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskAnalysisCompleted(DomainEvent):
    task_id: str = ""


# ── 计划 ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExecutionPlanFinalized(DomainEvent):
    task_id: str = ""
    plan_id: str = ""


# ── 用户交互 ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class UserSuggestionRequested(DomainEvent):
    task_id: str = ""


@dataclass(frozen=True)
class UserClarificationRequested(DomainEvent):
    task_id: str = ""
    order: str = ""
    question: str = ""


# ── Task 生命周期 ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskExecutionStarted(DomainEvent):
    task_id: str = ""


@dataclass(frozen=True)
class TaskResultProduced(DomainEvent):
    task_id: str = ""

@dataclass(frozen=True)
class TaskExecutionFailed(DomainEvent):
    task_id: str = ""

@dataclass(frozen=True)
class TaskPaused(DomainEvent):
    task_id: str = ""
    reason: str = ""


@dataclass(frozen=True)
class TaskCancelled(DomainEvent):
    task_id: str = ""
    reason: str = ""

@dataclass(frozen=True)
class ExecutionProgressProvided(DomainEvent):
    task_id: str = ""


# ── Stage 生命周期 ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StageExecutionStarted(DomainEvent):
    task_id: str = ""
    order: str = ""


@dataclass(frozen=True)
class StageResultProduced(DomainEvent):
    task_id: str = ""
    order: str = ""


# ── LLM ───────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LLMResponseGenerated(DomainEvent):
    task_id: str = ""
    order: str = ""
    model: str = ""


# ── 工具调用 ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolCallStarted(DomainEvent):
    task_id: str = ""
    order: str = ""
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallResultProduced(DomainEvent):
    task_id: str = ""
    order: str = ""
    tool_name: str = ""

@dataclass(frozen=True)
class ToolCallFailed(DomainEvent):
    task_id: str = ""
    order: str = ""
    tool_name: str = ""
    error: str = ""
    
@dataclass(frozen=True)
class UserCommand(DomainEvent):
    type: UserCommandType = UserCommandType.CANCEL
    task_id: str = ""
    user_id: str | None = None

ALL_EVENTS = [
    "DomainEvent",
    "TaskAnalysisCompleted",
    "ExecutionPlanFinalized",
    "UserSuggestionRequested",
    "UserClarificationRequested",
    "TaskExecutionStarted",
    "TaskResultProduced",
    "TaskExecutionFailed",
    "TaskPaused",
    "TaskCancelled",
    "StageExecutionStarted",
    "StageResultProduced",
    "LLMResponseGenerated",
    "ToolCallStarted",
    "ToolCallResultProduced",
    "ToolCallFailed",
]
