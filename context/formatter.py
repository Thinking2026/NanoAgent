from __future__ import annotations

import json
from typing import Any

from schemas import ChatMessage, LLMRequest, LLMResponse


class MessageFormatter:
    def __init__(self, max_messages: int | None = None) -> None:
        self._max_messages = max_messages

    def normalize_user_message(self, raw_text: str) -> ChatMessage:
        content = raw_text.strip()
        return ChatMessage(role="user", content=content)

    def build_system_prompt(self, base_prompt: str, context: list[dict[str, Any]]) -> str:
        if not context:
            return base_prompt

        context_block = "\n".join(
            f"- {item['title']}: {item['content']}" for item in context
        )
        return (
            f"{base_prompt}\n\n"
            "External context:\n"
            f"{context_block}\n\n"
            "Use the context when it is relevant, but say when you are unsure."
        )

    def build_request(
        self,
        system_prompt: str,
        conversation: list[ChatMessage],
        tools: list[dict[str, Any]],
        context: list[dict[str, Any]],
        max_messages: int | None = None,
    ) -> LLMRequest:
        effective_max_messages = self._max_messages if max_messages is None else max_messages
        trimmed_conversation = self._trim_conversation(
            conversation=conversation,
            max_messages=effective_max_messages,
        )
        return LLMRequest(
            system_prompt=system_prompt,
            messages=trimmed_conversation,
            tools=tools,
            context=context,
        )

    def format_tool_observation(
        self,
        tool_name: str,
        output: str,
        llm_raw_tool_call_id: str | None = None,
    ) -> ChatMessage:
        return ChatMessage(
            role="conversation",
            content=output,
            metadata={
                "tool_name": tool_name,
                "llm_raw_tool_call_id": llm_raw_tool_call_id,
                "conversation_source": "tool",
            },
        )

    def format_rag_observation(
        self,
        query: str,
        context: list[dict[str, Any]],
    ) -> ChatMessage:
        return ChatMessage(
            role="conversation",
            content=json.dumps(
                {
                    "query": query,
                    "matches": context,
                },
                ensure_ascii=False,
            ),
            metadata={
                "conversation_source": "rag",
                "query": query,
            },
        )

    def parse_response(self, response: LLMResponse) -> LLMResponse:
        return response

    @staticmethod
    def _trim_conversation(
        conversation: list[ChatMessage],
        max_messages: int | None,
    ) -> list[ChatMessage]:
        if max_messages is None or max_messages <= 0:
            return list(conversation)
        if len(conversation) <= max_messages:
            return list(conversation)
        return list(conversation[-max_messages:])
