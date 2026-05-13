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
    CONFIG_ERROR,
    ToolCall,
    build_pipeline_error,
)


class OpenAILLMClient(BaseLLMClient):
    provider_name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._model = model
        self._init_http(
            base_url=base_url,
            default_headers={
                "Authorization": f"Bearer {api_key}",
                **(extra_headers or {}),
            },
            timeout=timeout,
        )

    @classmethod
    def from_settings(
        cls,
        api_key: str | None,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ) -> "OpenAILLMClient":
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise build_pipeline_error(CONFIG_ERROR, "Missing API key for OpenAI client.")
        return cls(
            api_key=resolved_api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
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
                payload = {
                    "model": model,
                    "messages": self._serialize_messages(request),
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                }
                print(payload)
                tools = self._serialize_tools(request.tool_schemas)
                if tools:
                    payload["tools"] = tools
                    payload["tool_choice"] = "auto"
                response_data = self._post_json("/chat/completions", payload)
                response = self._parse_chat_completion(response_data)
            except HttpError as exc:
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

    def _post_json(self, path: str, payload: dict) -> dict:
        return self._http.post_json(path, payload)

    @staticmethod
    def _serialize_messages(request: UnifiedLLMRequest) -> list[dict]:
        serialized_messages: list[dict] = []
        if request.system_prompt:
            serialized_messages.append({"role": "system", "content": request.system_prompt})
        for message in request.messages:
            serialized = {"role": message.role, "content": message.content}
            if message.role == "assistant":
                tool_calls = message.metadata.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    serialized["tool_calls"] = OpenAILLMClient._serialize_assistant_tool_calls(
                        tool_calls
                    )
            if message.role == "tool":
                tool_call_id = message.metadata.get("llm_raw_tool_call_id")
                if tool_call_id:
                    serialized["tool_call_id"] = tool_call_id
            serialized_messages.append(serialized)
        return serialized_messages

    @staticmethod
    def _serialize_tools(tools: list[dict] | None) -> list[dict]:
        if not tools:
            return []
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in tools
        ]

    @staticmethod
    def _serialize_assistant_tool_calls(tool_calls: list[dict]) -> list[dict]:
        serialized_tool_calls: list[dict] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_call_id = tool_call.get("llm_raw_tool_call_id")
            tool_name = tool_call.get("name")
            arguments = tool_call.get("arguments") or {}
            if not isinstance(tool_call_id, str) or not isinstance(tool_name, str):
                continue
            serialized_tool_calls.append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )
        return serialized_tool_calls

    @classmethod
    def _parse_chat_completion(cls, response_data: dict) -> LLMResponse:
        choices = response_data.get("choices") or []
        if not choices:
            raise LLMNormalizedError(LLMNormalizedErrorCode.EMPTY_CHOICES, f"OpenAI API returned no choices: {response_data}")
        first_choice = choices[0]
        message = first_choice.get("message") or {}
        usage_data = response_data.get("usage") or {}
        finish_reason = first_choice.get("finish_reason", "stop")
        if finish_reason == "tool_calls":
            finish_reason = "tool_use"
        if finish_reason == "content_filter":
            raise LLMNormalizedError(LLMNormalizedErrorCode.CONTENT_FILTERED, f"OpenAI content filter triggered: {response_data}")
        if finish_reason == "length":
            finish_reason = "length"  # preserved; caller checks FINISH_REASON_LENGTH via response
            print("hahah")
            raise LLMNormalizedError(LLMNormalizedErrorCode.OUTPUT_TRUNCATED, "Response is truncated")
        try:
            tool_calls = [
                ToolCall(
                    name=tool_call["function"]["name"],
                    arguments=json.loads(tool_call["function"]["arguments"] or "{}"),
                    llm_raw_tool_call_id=tool_call["id"],
                )
                for tool_call in (message.get("tool_calls") or [])
            ]
        except (KeyError, TypeError) as exc:
            raise LLMNormalizedError(
                LLMNormalizedErrorCode.TOOL_CALL_PARSE_ERROR,
                f"OpenAI API returned an invalid tool call payload: {exc}",
            ) from exc
        except json.JSONDecodeError as exc:
            raise LLMNormalizedError(
                LLMNormalizedErrorCode.TOOL_CALL_PARSE_ERROR,
                f"OpenAI tool call arguments are not valid JSON: {exc}",
            ) from exc
        return LLMResponse(
            assistant_message=LLMMessage(
                role=message.get("role", "assistant"),
                content=message.get("content") or "",
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
            finish_reason=finish_reason,
            usage=LLMUsage(
                prompt_tokens=int(usage_data.get("prompt_tokens") or 0),
                completion_tokens=int(usage_data.get("completion_tokens") or 0),
                total_tokens=int(usage_data.get("total_tokens") or 0),
            ) if usage_data else None,
            raw_response=response_data,
        )
