from __future__ import annotations

from typing import TYPE_CHECKING

from agent.models.reasoning.decision import NextDecision
from schemas.types import ContextWindow, UnifiedLLMRequest

if TYPE_CHECKING:
    from agent.models.reasoning.strategy import Strategy
    from llm.llm_gateway import LLMGateway
    from schemas import LLMMessage, ToolCall, ToolResult
    from tools.tool_registry import ToolRegistry


class ReasoningManager:
    """Entity responsible for executing a single LLM reasoning step.

    Delegates request building and response parsing to the injected Strategy.
    Does not handle tool execution, context writes, or provider switching —
    those are the responsibility of StageExecutor.
    """

    def __init__(self, config: ConfigReader, logger: Logger, tracer: Tracer, llm_gateway: LLMGateway, strategy: Strategy) -> None:
        self._config = config
        self._logger = logger
        self._tracer = tracer
        self._llm_gateway = llm_gateway
        self._strategy = strategy

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def reason_once(
        self,
        context_window: ContextWindow,
        provider_name: str,
    ) -> NextDecision:
        raw_request = UnifiedLLMRequest(
            messages=context_window.messages,
            tool_schemas=context_window.tool_schemas,
        )
        request = self._strategy.build_llm_request(raw_request)
        response = self._llm_gateway.generate(request, provider_name)
        return self._strategy.parse_llm_response(response)

    def set_llm_gateway(self, llm_gateway: LLMGateway) -> None:
        """Replace the current gateway (called by StageExecutor on provider fallback)."""
        self._llm_gateway = llm_gateway

    def get_llm_gateway(self) -> LLMGateway:
        return self._llm_gateway

    def format_tool_observation(
        self,
        tool_call: ToolCall,
        result: ToolResult,
    ) -> LLMMessage:
        """Format a ToolResult for context injection using the active strategy."""
        return self._strategy.format_tool_observation(tool_call, result)
