from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from llm.llm_api import BaseLLMClient
from schemas import ChatMessage, LLMRequest, LLMResponse, ToolCall, build_error


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
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._extra_headers = extra_headers or {}

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
            raise build_error("LLM_CONFIG_ERROR", "Missing API key for OpenAI client.")
        return cls(
            api_key=resolved_api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        payload = {
            "model": self._model,
            "messages": self._serialize_messages(request),
        }
        tools = self._serialize_tools(request.tools)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        response_data = self._post_json("/chat/completions", payload)
        return self._parse_chat_completion(response_data)

    def _post_json(self, path: str, payload: dict) -> dict:
        request_url = f"{self._base_url}{path}"
        http_request = urllib.request.Request(
            request_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                **self._extra_headers,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self._timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise build_error("LLM_HTTP_ERROR", f"OpenAI API HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise build_error("LLM_NETWORK_ERROR", f"OpenAI API request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise build_error("LLM_TIMEOUT", f"OpenAI API request timed out: {exc}") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise build_error("LLM_RESPONSE_PARSE_ERROR", f"OpenAI API returned invalid JSON: {exc}") from exc

    @staticmethod
    def _serialize_messages(request: LLMRequest) -> list[dict]:
        serialized_messages: list[dict] = [{"role": "system", "content": request.system_prompt}]
        for message in request.messages:
            role = OpenAILLMClient._map_message_role(message)
            serialized = {"role": role, "content": message.content}
            if role == "assistant":
                tool_calls = message.metadata.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    serialized["tool_calls"] = OpenAILLMClient._serialize_assistant_tool_calls(
                        tool_calls
                    )
            if role == "tool":
                tool_call_id = (
                    message.metadata.get("llm_raw_tool_call_id")
                    or message.metadata.get("tool_call_id")
                )
                if tool_call_id:
                    serialized["tool_call_id"] = tool_call_id
            serialized_messages.append(serialized)
        return serialized_messages

    @staticmethod
    def _map_message_role(message: ChatMessage) -> str:
        if message.role == "conversation" and message.metadata.get("conversation_source") == "tool":
            return "tool"
        if message.role == "conversation":
            return "assistant"
        return message.role

    @staticmethod
    def _serialize_tools(tools: list[dict]) -> list[dict]:
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
            llm_raw_tool_call_id = (
                tool_call.get("llm_raw_tool_call_id")
                or tool_call.get("id")
            )
            function_payload = tool_call.get("function")
            if not isinstance(llm_raw_tool_call_id, str) or not isinstance(function_payload, dict):
                continue
            serialized_tool_calls.append(
                {
                    "id": llm_raw_tool_call_id,
                    "type": "function",
                    "function": function_payload,
                }
            )
        return serialized_tool_calls

    @classmethod
    def _parse_chat_completion(cls, response_data: dict) -> LLMResponse:
        choices = response_data.get("choices") or []
        if not choices:
            raise build_error("LLM_RESPONSE_ERROR", f"OpenAI API returned no choices: {response_data}")
        first_choice = choices[0]
        message = first_choice.get("message") or {}
        try:
            tool_calls = [
                ToolCall(
                    name=tool_call["function"]["name"],
                    arguments=json.loads(tool_call["function"]["arguments"] or "{}"),
                    llm_raw_tool_call_id=tool_call["id"],
                )
                for tool_call in (message.get("tool_calls") or [])
            ]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise build_error(
                "LLM_RESPONSE_PARSE_ERROR",
                f"OpenAI API returned an invalid tool call payload: {exc}",
            ) from exc
        return LLMResponse(
            assistant_message=ChatMessage(
                role=message.get("role", "assistant"),
                content=message.get("content") or "",
                metadata={
                    "tool_calls_count": len(tool_calls),
                    "tool_calls": [
                        {
                            "id": tool_call.llm_raw_tool_call_id,
                            "llm_raw_tool_call_id": tool_call.llm_raw_tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                            },
                        }
                        for tool_call in tool_calls
                    ],
                },
            ),
            tool_calls=tool_calls,
            raw_response=response_data,
            finish_reason=first_choice.get("finish_reason", "stop"),
        )
