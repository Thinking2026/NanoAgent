from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from llm.llm_api import BaseLLMClient
from schemas import ChatMessage, LLMRequest, LLMResponse, ToolCall, build_error


class ClaudeLLMClient(BaseLLMClient):
    provider_name = "claude"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 60.0,
        max_tokens: int = 1024,
        anthropic_version: str = "2023-06-01",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._anthropic_version = anthropic_version
        self._extra_headers = extra_headers or {}

    @classmethod
    def from_settings(
        cls,
        api_key: str | None,
        model: str,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 60.0,
        max_tokens: int = 1024,
        anthropic_version: str = "2023-06-01",
    ) -> "ClaudeLLMClient":
        resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not resolved_api_key:
            raise build_error("LLM_CONFIG_ERROR", "Missing API key for Claude client.")
        return cls(
            api_key=resolved_api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_tokens=max_tokens,
            anthropic_version=anthropic_version,
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        payload: dict[str, object] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": self._serialize_messages(request),
        }
        if request.system_prompt:
            payload["system"] = request.system_prompt

        tools = self._serialize_tools(request.tools)
        if tools:
            payload["tools"] = tools

        response_data = self._post_json("/v1/messages", payload)
        return self._parse_message_response(response_data)

    def _post_json(self, path: str, payload: dict[str, object]) -> dict:
        http_request = urllib.request.Request(
            f"{self._base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": self._anthropic_version,
                **self._extra_headers,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self._timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise build_error("LLM_HTTP_ERROR", f"Claude API HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise build_error("LLM_NETWORK_ERROR", f"Claude API request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise build_error("LLM_TIMEOUT", f"Claude API request timed out: {exc}") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise build_error("LLM_RESPONSE_PARSE_ERROR", f"Claude API returned invalid JSON: {exc}") from exc

    @staticmethod
    def _serialize_messages(request: LLMRequest) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = []
        for message in request.messages:
            serialized = ClaudeLLMClient._serialize_message(message)
            if serialized is not None:
                messages.append(serialized)
        return messages

    @staticmethod
    def _serialize_message(message: ChatMessage) -> dict[str, object] | None:
        if message.role == "user":
            return {"role": "user", "content": message.content}
        if message.role == "assistant":
            tool_calls = message.metadata.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                content: list[dict[str, object]] = []
                if message.content:
                    content.append({"type": "text", "text": message.content})
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    tool_name = tool_call.get("name")
                    tool_call_id = (
                        tool_call.get("llm_raw_tool_call_id")
                    )
                    tool_arguments = tool_call.get("arguments")
                    if not isinstance(tool_name, str) or not isinstance(tool_call_id, str):
                        continue
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call_id,
                            "name": tool_name,
                            "input": tool_arguments if isinstance(tool_arguments, dict) else {},
                        }
                    )
                return {"role": "assistant", "content": content}
            return {"role": "assistant", "content": message.content}
        if message.metadata.get("conversation_source") == "tool":
            tool_call_id = (
                message.metadata.get("llm_raw_tool_call_id")
                or message.metadata.get("tool_call_id")
            )
            if not tool_call_id:
                return {"role": "user", "content": message.content}
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": message.content,
                    }
                ],
            }
        return {"role": "assistant", "content": message.content}

    @staticmethod
    def _serialize_tools(tools: list[dict]) -> list[dict[str, object]]:
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"],
            }
            for tool in tools
        ]

    @staticmethod
    def _parse_message_response(response_data: dict) -> LLMResponse:
        content_blocks = response_data.get("content")
        if not isinstance(content_blocks, list):
            raise build_error(
                "LLM_RESPONSE_ERROR",
                f"Claude API returned invalid content blocks: {response_data}",
            )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text", "")
                if text:
                    text_parts.append(str(text))
                continue
            if block_type == "tool_use":
                try:
                    tool_calls.append(
                        ToolCall(
                            name=str(block["name"]),
                            arguments=dict(block.get("input") or {}),
                            llm_raw_tool_call_id=str(block["id"]),
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise build_error(
                        "LLM_RESPONSE_PARSE_ERROR",
                        f"Claude API returned an invalid tool use payload: {exc}",
                    ) from exc

        return LLMResponse(
            assistant_message=ChatMessage(
                role="assistant",
                content="\n".join(text_parts).strip(),
                metadata={
                    "tool_calls_count": len(tool_calls),
                    "tool_calls": [
                        {
                            "name": tool_call.name,
                            "llm_raw_tool_call_id": tool_call.llm_raw_tool_call_id,
                            "arguments": tool_call.arguments,
                        }
                        for tool_call in tool_calls
                    ],
                },
            ),
            tool_calls=tool_calls,
            raw_response=response_data,
            finish_reason=str(response_data.get("stop_reason", "stop")),
        )
