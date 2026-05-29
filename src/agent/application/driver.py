from __future__ import annotations

from typing import TYPE_CHECKING

from agent.events.events import (
    DomainEvent, TaskCancelled, TaskExecutionFailed, TaskExecutionSucceed, TaskPaused,
    UserClarificationRequested, UserCommand, ALL_EVENTS,
)
from schemas.errors import PARAMETER_FORGET_SET, build_logic_error
from schemas.ids import TaskId, CheckpointId, UserId
from schemas.types import UserCommandType, UserMessage, UserMsgType
from schemas.task import TaskResult
from schemas.event_bus import EventBus

if TYPE_CHECKING:
    from agent.application.pipeline import Pipeline
    from agent.application.pipeline_thread import PipelineThread

_METADATA_LABELS = {
    "task_description": "原始任务",
    "task_type": "任务类型",
    "task_goal": "任务目标",
    "intent": "用户意图",
    "complexity": "复杂度",
    "estimated_steps": "预计步骤",
    "required_tools": "可用工具",
    "risks": "风险提示",
    "plan": "计划概览",
    "steps": "步骤数",
    "total_steps": "总步骤",
    "step_order": "当前步骤",
    "stage_goal": "步骤目标",
    "stage_description": "步骤说明",
    "required_stage_tools": "本步工具",
    "start_reason": "启动原因",
    "provider": "模型",
    "decision": "决策",
    "iteration": "推理轮次",
    "tool_name": "工具",
    "arguments": "参数",
    "success": "是否成功",
    "result": "结果摘要",
    "reason": "原因",
    "error": "错误",
    "feedback": "评审反馈",
    "passed": "是否通过",
    "question": "需要补充",
    "message": "消息",
    "progress": "进度",
    "retry": "重试次数",
    "max_retries": "最大重试",
    "recovery_action": "恢复策略",
}

_HIDDEN_METADATA_KEYS = {
    "content",
}


def _format_value(value) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (list, tuple, set)):
        if not value:
            return "无"
        return ", ".join(_format_value(v) for v in value)
    if isinstance(value, dict):
        if not value:
            return "无"
        items = [f"{k}={_format_value(v)}" for k, v in value.items()]
        text = ", ".join(items)
    else:
        text = str(value)
    text = " ".join(text.strip().split())
    return text if len(text) <= 240 else f"{text[:237]}..."


def _format_content(event: DomainEvent) -> str:
    lines = [event.content.strip()]
    if len(event.task_id) > 0:
        lines.append(f"- 任务ID：{event.task_id}")
    for k, v in event.metadata.items():
        if k in _HIDDEN_METADATA_KEYS or k.startswith("_"):
            continue
        label = _METADATA_LABELS.get(k, k)
        lines.append(f"- {label}：{_format_value(v)}")
    return "\n".join(lines)

class PipelineDriver:
    # Metadata key used by callers to declare message intent explicitly.
    # If absent, the driver infers intent from context (see _classify).
    MSG_TYPE_KEY = "msg_type"

    def __init__(
        self,
        event_bus: EventBus,
        thread: PipelineThread,
    ) -> None:
        self._thread = thread
        self._pipeline: Pipeline = None
        for event_type in ALL_EVENTS:
            event_bus.subscribe(event_type, self.publish_event)

    def use_pipeline(self, pipeline: Pipeline) -> None:
        pipeline.set_driver(self)
        self._pipeline = pipeline

    # ------------------------------------------------------------------
    # Task lifecycle entry points
    # ------------------------------------------------------------------

    def submit_task(self, user_id: UserId, task_description: str) -> TaskResult:
        """Run a task synchronously and return the result."""
        if self._pipeline is None:
            raise build_logic_error(code=PARAMETER_FORGET_SET, message="pipeline is none")
        return self._pipeline.run(user_id=user_id, task_description=task_description)

    def submit_checkpoint(self, user_id: UserId, checkpoint_path: str) -> TaskResult:
        """Resume execution from a checkpoint file."""
        if self._pipeline is None:
            raise build_logic_error(code=PARAMETER_FORGET_SET, message="pipeline is none")
        return self._pipeline.run(
            user_id=user_id,
            task_description=checkpoint_path,
            msg_type=UserMsgType.LOAD_CHECKPOINT,
        )

    def loop_user_messages(self, timeout: float) -> UserCommand | None:
        if self._thread is None:
            raise build_logic_error(code=PARAMETER_FORGET_SET, message="self._thread is none")
        user_message = self._thread.loop_user_message(timeout)
        if user_message is not None:
            return self.convert_user_message(user_message)
        return None

    def convert_user_message(self, message: UserMessage) -> UserCommand | None:
        if message is not None:
            if message.msg_type == UserMsgType.CANCEL:
                return UserCommand(type=UserCommandType.CANCEL, task_id=message.task_id, user_id=message.user_id)
            elif message.msg_type == UserMsgType.RESUME:
                return UserCommand(type=UserCommandType.RESUME, task_id=message.task_id, user_id=message.user_id)
            elif message.msg_type == UserMsgType.CLARIFICATION:
                return UserCommand(type=UserCommandType.CLARIFICATION, task_id=message.task_id, user_id=message.user_id, content=message.content)
            elif message.msg_type == UserMsgType.GUIDANCE:
                return UserCommand(type=UserCommandType.GUIDANCE, task_id=message.task_id, user_id=message.user_id, content=message.content)
        return None

    def convert_pipeline_event(self, event: DomainEvent) -> UserMessage | None:
        if event is not None:
            is_last = isinstance(event, TaskExecutionSucceed) or isinstance(event, TaskExecutionFailed)
            return UserMessage(
                msg_type=UserMsgType.AGENT_PROGESS,
                task_id=event.task_id,
                user_id=event.user_id,
                content=_format_content(event),
                metadata={"is_last_message": True} if is_last else {}
            )
        return None
    
    def publish_event(self, event: DomainEvent) -> None:
        msg = self.convert_pipeline_event(event)
        if msg is not None:
            self._thread.publish_msg_to_user(msg)
