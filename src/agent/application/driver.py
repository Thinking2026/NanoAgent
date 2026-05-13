from __future__ import annotations

from typing import TYPE_CHECKING

from agent.events.events import DomainEvent, TaskCancelled, TaskPaused, UserClarificationRequested, UserCommand, ALL_EVENTS
from schemas.errors import PARAMETER_FORGET_SET, build_logic_error
from schemas.ids import TaskId, CheckpointId, UserId
from schemas.types import UserCommandType, UserMessage, UserMsgType
from schemas.task import TaskResult
from schemas.event_bus import EventBus

if TYPE_CHECKING:
    from agent.application.pipeline import Pipeline
    from agent.application.pipeline_thread import PipelineThread

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
        if self._pipeline is None:
            raise build_logic_error(code=PARAMETER_FORGET_SET, message="pipeline is none")
        """Run a task synchronously and return the result."""
        return self._pipeline.run(user_id=user_id, task_description=task_description)

    def loop_user_messages(self, timeout: float) -> UserCommand | None:
        if self._thread is None:
            raise build_logic_error(code=PARAMETER_FORGET_SET, message="thread is none")
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
        if isinstance(event, TaskCancelled):
            return UserMessage(msg_type=UserMsgType.CANCEL, task_id=event.task_id, user_id=None, content=event.reason)
        elif isinstance(event, TaskPaused):
            return UserMessage(msg_type=UserMsgType.PAUSE_FROM_AGENT, task_id=event.task_id, user_id=None, content=event.reason)
        elif isinstance(event, UserClarificationRequested):
            return UserMessage(msg_type=UserMsgType.CLARIFICATION, task_id=event.task_id, user_id=None, content=event.question)

        return UserMessage(msg_type=UserMsgType.PROGRESS_FROM_AGENT, task_id=event.task_id, user_id=None, content=event.content)
    
    def publish_event(self, event: DomainEvent) -> None:
        msg = self.convert_pipeline_event(event)
        if msg is not None:
            self._thread.publish_msg_to_user(msg)