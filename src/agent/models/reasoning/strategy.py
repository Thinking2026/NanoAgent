from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from agent.models.reasoning.decision import NextDecision
from schemas import LLMMessage, UnifiedLLMRequest, LLMResponse

if TYPE_CHECKING:
    from schemas import ToolCall, ToolResult

class Strategy(ABC):
    @abstractmethod
    def build_llm_request(self, request: UnifiedLLMRequest) -> UnifiedLLMRequest:
        """Format conversation into an LLMRequest for this reasoning mode."""
        raise NotImplementedError

    @abstractmethod
    def parse_llm_response(self, response: LLMResponse) -> NextDecision:
        """Parse an LLMResponse into a structured decision."""
        raise NotImplementedError

    @abstractmethod
    def format_tool_observation(
        self,
        tool_call: ToolCall,
        result: ToolResult,
    ) -> LLMMessage:
        """Format a tool result as a conversation message."""
        raise NotImplementedError
