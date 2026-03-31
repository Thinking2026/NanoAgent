from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from context.formatter import MessageFormatter
from context.shared_context import SharedContext
from llm import BaseLLMClient
from rag.rag_service import RAGService
from schemas import AgentError, ChatMessage, SessionStatus
from tools import ToolRegistry


@dataclass(slots=True)
class AgentExecutionResult:
    user_messages: list[ChatMessage] = field(default_factory=list)
    error: AgentError | None = None
    should_reset: bool = False


class Agent(ABC):
    def __init__(
        self,
        shared_context: SharedContext,
        message_formatter: MessageFormatter,
        llm_client: BaseLLMClient,
        tool_registry: ToolRegistry,
        rag_service: RAGService,
        max_tool_iterations: int,
    ) -> None:
        self._shared_context = shared_context
        self._message_formatter = message_formatter
        self._llm_client = llm_client
        self._tool_registry = tool_registry
        self._rag_service = rag_service
        self._max_tool_iterations = max_tool_iterations
        self._cur_react_attempt_iterations = 0

    def begin_session(self) -> None:
        self._cur_react_attempt_iterations = 0
        self._shared_context.set_session_status(SessionStatus.IN_PROGRESS)

    def reset(self) -> None:
        self._cur_react_attempt_iterations = 0
        self._shared_context.archive_current_task()
        self._shared_context.set_session_status(SessionStatus.NEW_TASK)

    def release_resources(self) -> None:
        self.reset()

    def get_react_attempt_iterations(self) -> int:
        return self._cur_react_attempt_iterations

    @abstractmethod
    def run(
        self,
        session_status: SessionStatus,
        user_message: ChatMessage | None,
    ) -> AgentExecutionResult:
        raise NotImplementedError
