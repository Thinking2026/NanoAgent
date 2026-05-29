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
        if "content" not in base_kwargs:
            inst.content = inst.render_content()
        return inst

    def render_content(self) -> str:
        return self.content


def _clip(value: Any, limit: int = 120) -> str:
    text = "" if value is None else str(value).strip()
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _count_label(count: Any, unit: str) -> str:
    try:
        return f"{int(count)} {unit}"
    except (TypeError, ValueError):
        return f"{count} {unit}"


def _passed_text(value: Any) -> str:
    return "通过" if bool(value) else "未通过"


# ── 分析 ──────────────────────────────────────────────────────────────────────

@dataclass
class TaskAnalysisStarted(DomainEvent):
    content: str = "正在理解任务需求"

    def render_content(self) -> str:
        task = _clip(self.metadata.get("task_description"), 100)
        return f"正在理解任务需求：{task}" if task else self.content

@dataclass
class TaskAnalysisSucceed(DomainEvent):
    content: str = "任务分析完成"

    def render_content(self) -> str:
        goal = _clip(self.metadata.get("task_goal"), 100)
        task_type = self.metadata.get("task_type")
        if goal and task_type:
            return f"任务分析完成：目标是“{goal}”，类型为 {task_type}"
        if goal:
            return f"任务分析完成：目标是“{goal}”"
        return self.content

@dataclass
class TaskAnalysisFailed(DomainEvent):
    content: str = "任务分析失败"


# ── 计划 ──────────────────────────────────────────────────────────────────────

@dataclass
class PlanGenerateStarted(DomainEvent):
    content: str = "正在制定执行计划"

@dataclass
class PlanGenerateSucceed(DomainEvent):
    content: str = "执行计划已确定"

    def render_content(self) -> str:
        steps = self.metadata.get("steps")
        if steps:
            return f"执行计划已确定：共 {_count_label(steps, '步')}"
        return self.content

@dataclass
class PlanGenerateFailed(DomainEvent):
    content: str = "执行计划生成失败"

# ── 评测 ──────────────────────────────────────────────────────────────────────

@dataclass
class PlanEvaluateStarted(DomainEvent):
    content: str = "正在检查执行计划是否可靠"

@dataclass
class PlanEvaluateSucceed(DomainEvent):
    content: str = "执行计划检查完成"

    def render_content(self) -> str:
        if "passed" in self.metadata:
            return f"执行计划检查完成：{_passed_text(self.metadata.get('passed'))}"
        return self.content

@dataclass
class PlanEvaluateFailed(DomainEvent):
    content: str = "执行计划检查失败"

@dataclass
class TaskResultEvaluateStarted(DomainEvent):
    content: str = "正在复核最终结果"

@dataclass
class TaskResultEvaluateSucceed(DomainEvent):
    content: str = "最终结果复核完成"

    def render_content(self) -> str:
        if "passed" in self.metadata:
            return f"最终结果复核完成：{_passed_text(self.metadata.get('passed'))}"
        return self.content

@dataclass
class TaskResultEvaluateFailed(DomainEvent):
    content: str = "最终结果复核失败"

@dataclass
class StageResultEvaluateStarted(DomainEvent):
    content: str = "正在检查当前步骤结果"

    def render_content(self) -> str:
        order = self.metadata.get("order")
        return f"正在检查第 {order} 步结果" if order else self.content

@dataclass
class StageResultEvaluateSucceed(DomainEvent):
    content: str = "步骤结果检查完成"

    def render_content(self) -> str:
        order = self.metadata.get("order")
        passed = self.metadata.get("passed")
        if order and "passed" in self.metadata:
            return f"第 {order} 步结果检查完成：{_passed_text(passed)}"
        return self.content

@dataclass
class StageResultEvaluateFailed(DomainEvent):
    content: str = "步骤结果检查失败"


# ── 用户交互 ──────────────────────────────────────────────────────────────────

@dataclass
class UserClarificationRequested(DomainEvent):
    content: str = "需要用户补充信息"

@dataclass
class UserClarificationReceived(DomainEvent):
    content: str = "已收到用户补充信息"

@dataclass
class UserGuidanceReceived(DomainEvent):
    content: str = "已收到用户的新指引"

@dataclass
class TaskPaused(DomainEvent):
    content: str = "任务已暂停"

@dataclass
class TaskResumed(DomainEvent):
    content: str = "任务已恢复"

@dataclass
class TaskCancelled(DomainEvent):
    content: str = "任务已取消"


# ── Task 生命周期 ──────────────────────────────────────────────────────────────

@dataclass
class TaskExecutionStarted(DomainEvent):
    content: str = "开始执行任务"

    def render_content(self) -> str:
        total = self.metadata.get("total_steps")
        return f"开始执行任务：将按 {_count_label(total, '步')}推进" if total else self.content

@dataclass
class TaskExecutionSucceed(DomainEvent):
    content: str = "任务已完成"

@dataclass
class TaskExecutionFailed(DomainEvent):
    content: str = "任务执行失败"


# ── Stage 生命周期 ─────────────────────────────────────────────────────────────

@dataclass
class StageExecutionStarted(DomainEvent):
    content: str = "开始执行步骤"

    def render_content(self) -> str:
        order = self.metadata.get("step_order")
        total = self.metadata.get("total_steps")
        goal = _clip(self.metadata.get("stage_goal"), 100)
        prefix = f"开始执行第 {order}/{total} 步" if order and total else self.content
        return f"{prefix}：{goal}" if goal else prefix

@dataclass
class StageExecutionSucceed(DomainEvent):
    content: str = "步骤执行完成"

    def render_content(self) -> str:
        order = self.metadata.get("step_order")
        total = self.metadata.get("total_steps")
        if order and total:
            return f"第 {order}/{total} 步执行完成"
        if order:
            return f"第 {order} 步执行完成"
        return self.content

@dataclass
class StageExecutionFailed(DomainEvent):
    content: str = "步骤执行失败"

# ── LLM ───────────────────────────────────────────────────────────────────────

@dataclass
class NextDecisionMade(DomainEvent):
    content: str = "已完成一次推理决策"

    def render_content(self) -> str:
        decision = self.metadata.get("decision")
        order = self.metadata.get("step_order")
        if order and decision:
            return f"第 {order} 步正在推进：下一步动作是 {decision}"
        if decision:
            return f"已完成一次推理决策：{decision}"
        return self.content


# ── 工具调用 ──────────────────────────────────────────────────────────────────

@dataclass
class ToolCallStarted(DomainEvent):
    content: str = "正在调用工具"

    def render_content(self) -> str:
        name = self.metadata.get("tool_name")
        return f"正在调用工具：{name}" if name else self.content

@dataclass
class ToolCallResultProduced(DomainEvent):
    content: str = "工具调用完成"

    def render_content(self) -> str:
        name = self.metadata.get("tool_name")
        success = self.metadata.get("success")
        if name and success is not None:
            return f"工具调用完成：{name}（{'成功' if success else '失败'}）"
        return f"工具调用完成：{name}" if name else self.content

@dataclass
class ToolCallFailed(DomainEvent):
    content: str = "工具调用失败"

# ── REPLAN相关 ──────────────────────────────────────────────────────────────────

@dataclass
class RePlanStarted(DomainEvent):
    content: str = "正在调整执行计划"

@dataclass
class RePlanSucceed(DomainEvent):
    content: str = "执行计划已调整"

@dataclass
class RePlanFailed(DomainEvent):
    content: str = "执行计划调整失败"

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
