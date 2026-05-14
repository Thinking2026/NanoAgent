from __future__ import annotations

import json
import os

from llm.llm_gateway import BaseLLMClient, classify_http_error, classify_config_error, classify_timeout_error, classify_json_error
from schemas import (
    ConfigError,
    HttpError,
    LLMNormalizedError,
    LLMNormalizedErrorCode,
    LLMMessage,
    UnifiedLLMRequest,
    LLMResponse,
    LLMUsage,
    ToolCall,
    build_pipeline_error,
)
from schemas.errors import CONFIG_ERROR


class ClaudeLLMClient(BaseLLMClient):
    provider_name = "claude"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 60.0,
        max_tokens: int = 1048576,
        anthropic_version: str = "2023-06-01",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._anthropic_version = anthropic_version
        self._init_http(
            base_url=base_url,
            default_headers={
                "x-api-key": api_key,
                "anthropic-version": anthropic_version,
                **(extra_headers or {}),
            },
            timeout=timeout,
        )

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
            raise build_pipeline_error(CONFIG_ERROR, "Missing API key for Claude client.")
        return cls(
            api_key=resolved_api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_tokens=max_tokens,
            anthropic_version=anthropic_version,
        )

    def generate(self, request: UnifiedLLMRequest) -> LLMResponse:
        model = request.model_override or self._model
        last_message = request.messages[-1].content if request.messages else ""
        with self._start_span(
            "llm.generate",
            attributes={
                "provider": self.provider_name,
                "model": model,
                "message_count": len(request.messages),
                "last_user_message": last_message,
            },
        ) as span:
            try:
                payload: dict[str, object] = {
                    "model": model,
                    "max_tokens": request.max_tokens or self._max_tokens,
                }
                if request.temperature is not None:
                    payload["temperature"] = request.temperature
                if request.system_prompt:
                    payload["system"] = request.system_prompt

                tools = self._serialize_tools(request.tool_schemas)
                if tools:
                    payload["tools"] = tools

                messages = self._serialize_messages(request)
                use_prefill = request.json_mode and not tools
                if use_prefill:
                    messages.append({"role": "assistant", "content": "{"})
                payload["messages"] = messages

                response_data = self._post_json("/v1/messages", payload)
                response = self._parse_message_response(response_data, prepend_brace=use_prefill)
            except HttpError as exc:
                if exc.status == 529:
                    raise LLMNormalizedError(
                        LLMNormalizedErrorCode.PROVIDER_OVERLOADED,
                        f"Claude overloaded: {exc.body}",
                        raw_status=529,
                        provider=self.provider_name,
                    ) from exc
                raise classify_http_error(exc, provider=self.provider_name) from exc
            except ConfigError as exc:
                raise classify_config_error(exc, provider=self.provider_name) from exc
            except TimeoutError as exc:
                raise classify_timeout_error(exc, provider=self.provider_name) from exc
            except json.JSONDecodeError as exc:
                raise classify_json_error(exc, provider=self.provider_name) from exc
            span.add_attributes(
                {
                    "finish_reason": response.finish_reason,
                    "tool_calls_count": len(response.tool_calls),
                    "tool_calls": [
                        {"name": tc.name, "llm_raw_tool_call_id": tc.llm_raw_tool_call_id}
                        for tc in response.tool_calls
                    ],
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                    "completion_tokens": response.usage.completion_tokens if response.usage else None,
                    "response_text": response.assistant_message.content,
                }
            )
            return response

    def _post_json(self, path: str, payload: dict[str, object]) -> dict:
        return self._http.post_json(path, payload)

    @staticmethod
    def _serialize_messages(request: UnifiedLLMRequest) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = []
        for message in request.messages:
            serialized = ClaudeLLMClient._serialize_message(message)
            if serialized is not None:
                messages.append(serialized)
        return messages

    @staticmethod
    def _serialize_message(message: LLMMessage) -> dict[str, object] | None:
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
                    tool_call_id = tool_call.get("llm_raw_tool_call_id")
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
        if message.role == "tool":
            tool_call_id = message.metadata.get("llm_raw_tool_call_id")
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
        return None

    @staticmethod
    def _serialize_tools(tools: list[dict] | None) -> list[dict[str, object]]:
        if not tools:
            return []
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"],
            }
            for tool in tools
        ]

    @staticmethod
    def _parse_message_response(response_data: dict, prepend_brace: bool = False) -> LLMResponse:
        content_blocks = response_data.get("content")
        if not isinstance(content_blocks, list):
            raise LLMNormalizedError(
                LLMNormalizedErrorCode.RESPONSE_ERROR,
                f"Claude API returned invalid content blocks: {response_data}",
            )

        raw_finish_reason = str(response_data.get("stop_reason", "stop"))
        if raw_finish_reason == "max_tokens":
            # Truncated — still parse what we have; caller sees finish_reason="length"
            raise LLMNormalizedError(LLMNormalizedErrorCode.OUTPUT_TRUNCATED, "Response is truncated")
        if raw_finish_reason == "content_filter":
            raise LLMNormalizedError(LLMNormalizedErrorCode.CONTENT_FILTERED, f"Claude content filter triggered: {response_data}")

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
                    raise LLMNormalizedError(
                        LLMNormalizedErrorCode.TOOL_CALL_PARSE_ERROR,
                        f"Claude API returned an invalid tool use payload: {exc}",
                    ) from exc

        raw_finish_reason = str(response_data.get("stop_reason", "stop"))
        finish_reason_map = {
            "end_turn": "stop",
            "tool_use": "tool_use",
            "max_tokens": "length",
        }
        usage_data = response_data.get("usage") or {}
        prompt_tokens = int(usage_data.get("input_tokens") or 0)
        completion_tokens = int(usage_data.get("output_tokens") or 0)
        return LLMResponse(
            assistant_message=LLMMessage(
                role="assistant",
                content=("{" + "\n".join(text_parts).strip() if prepend_brace else "\n".join(text_parts).strip()),
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
            finish_reason=finish_reason_map.get(raw_finish_reason, raw_finish_reason),
            usage=LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ) if usage_data else None,
            raw_response=response_data,
        )
