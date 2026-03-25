from __future__ import annotations

from typing import Any

from schemas import ChatMessage, LLMRequest, LLMResponse


class MessageFormatter:
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
    ) -> LLMRequest:
        return LLMRequest(
            system_prompt=self.build_system_prompt(system_prompt, context),
            messages=conversation,
            tools=tools,
            context=context,
        )

    def format_tool_observation(
        self,
        tool_name: str,
        output: str,
        call_id: str | None = None,
    ) -> ChatMessage:
        return ChatMessage(
            role="tool",
            content=output,
            metadata={"tool_name": tool_name, "tool_call_id": call_id},
        )

    def parse_response(self, response: LLMResponse) -> LLMResponse:
        return response
