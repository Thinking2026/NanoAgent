from __future__ import annotations

import json
import os
import uuid
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Iterable

from schemas import ChatMessage, LLMRequest, LLMResponse, ToolCall


class BaseLLMClient(ABC):
    provider_name: str = "base"

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    provider_name = "mock"

    def generate(self, request: LLMRequest) -> LLMResponse:
        latest_message = request.messages[-1]

        if latest_message.role == "tool":
            tool_name = latest_message.metadata.get("tool_name", "tool")
            content = (
                f"我已经拿到 `{tool_name}` 的结果：{latest_message.content}。"
                "基于这个 observation，这就是本轮的最终回答。"
            )
            return LLMResponse(
                assistant_message=ChatMessage(role="assistant", content=content),
                raw_response={"mode": "tool_follow_up"},
                finish_reason="stop",
            )

        user_content = latest_message.content.lower()
        if self._should_call_time_tool(user_content):
            return self._tool_response("current_time", {})
        if self._should_call_echo_tool(user_content):
            text = latest_message.content.split(":", 1)[-1].strip()
            return self._tool_response("echo", {"text": text or latest_message.content})

        context_hint = ""
        if request.context:
            top_doc = request.context[0]
            context_hint = f" 我还参考了资料《{top_doc['title']}》。"

        content = (
            "这是一个来自 MockLLMClient 的直接回答。"
            f"你刚才的问题是：{latest_message.content}.{context_hint}"
        )
        return LLMResponse(
            assistant_message=ChatMessage(role="assistant", content=content),
            raw_response={"mode": "direct_answer"},
            finish_reason="stop",
        )

    @staticmethod
    def _should_call_time_tool(user_content: str) -> bool:
        keywords = ["time", "clock", "几点", "时间"]
        return any(keyword in user_content for keyword in keywords)

    @staticmethod
    def _should_call_echo_tool(user_content: str) -> bool:
        keywords = ["echo", "repeat", "复述", "回显"]
        return any(keyword in user_content for keyword in keywords)

    @staticmethod
    def _tool_response(name: str, arguments: dict) -> LLMResponse:
        tool_call = ToolCall(name=name, arguments=arguments, call_id=str(uuid.uuid4()))
        return LLMResponse(
            assistant_message=ChatMessage(
                role="assistant",
                content=f"Calling tool: {name}",
                metadata={"tool_call": name},
            ),
            tool_calls=[tool_call],
            raw_response={"mode": "tool_call"},
            finish_reason="tool_calls",
        )


class OpenAICompatibleLLMClient(BaseLLMClient):
    provider_name = "openai_compatible"

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
    def from_env(cls) -> "OpenAICompatibleLLMClient":
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY or LLM_API_KEY for OpenAI-compatible client.")

        model = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "gpt-4.1-mini"
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
        return cls(api_key=api_key, model=model, base_url=base_url)

    @classmethod
    def from_settings(
        cls,
        api_key: str | None,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ) -> "OpenAICompatibleLLMClient":
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        if not resolved_api_key:
            raise ValueError("Missing API key for OpenAI-compatible client.")
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
            raise RuntimeError(f"OpenAI-compatible API HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI-compatible API request failed: {exc.reason}") from exc

        return json.loads(body)

    def _serialize_messages(self, request: LLMRequest) -> list[dict]:
        serialized_messages: list[dict] = [
            {"role": "system", "content": request.system_prompt}
        ]
        for message in request.messages:
            serialized = {"role": message.role, "content": message.content}
            if message.role == "tool":
                tool_call_id = message.metadata.get("tool_call_id")
                if tool_call_id:
                    serialized["tool_call_id"] = tool_call_id
            serialized_messages.append(serialized)
        return serialized_messages

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
    def _parse_chat_completion(response_data: dict) -> LLMResponse:
        choices = response_data.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenAI-compatible API returned no choices: {response_data}")

        first_choice = choices[0]
        message = first_choice.get("message") or {}
        tool_calls = [
            ToolCall(
                name=tool_call["function"]["name"],
                arguments=json.loads(tool_call["function"]["arguments"] or "{}"),
                call_id=tool_call["id"],
            )
            for tool_call in (message.get("tool_calls") or [])
        ]

        assistant_message = ChatMessage(
            role=message.get("role", "assistant"),
            content=message.get("content") or "",
            metadata={"tool_calls_count": len(tool_calls)},
        )
        return LLMResponse(
            assistant_message=assistant_message,
            tool_calls=tool_calls,
            raw_response=response_data,
            finish_reason=first_choice.get("finish_reason", "stop"),
        )


class LLMProviderRegistry:
    def __init__(self, providers: Iterable[BaseLLMClient] | None = None) -> None:
        self._providers: dict[str, BaseLLMClient] = {}
        for provider in providers or []:
            self.register(provider)

    def register(self, provider: BaseLLMClient) -> None:
        self._providers[provider.provider_name] = provider

    def get(self, provider_name: str) -> BaseLLMClient:
        try:
            return self._providers[provider_name]
        except KeyError as exc:
            available = ", ".join(sorted(self._providers)) or "<none>"
            raise ValueError(
                f"Unknown LLM provider: {provider_name}. Available providers: {available}"
            ) from exc

    def list_providers(self) -> list[str]:
        return sorted(self._providers)


class DynamicLLMClient(BaseLLMClient):
    def __init__(self, registry: LLMProviderRegistry, default_provider: str) -> None:
        self._registry = registry
        self._provider_name = default_provider

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def use_provider(self, provider_name: str) -> None:
        self._registry.get(provider_name)
        self._provider_name = provider_name

    def generate(self, request: LLMRequest) -> LLMResponse:
        provider = self._registry.get(self._provider_name)
        return provider.generate(request)
