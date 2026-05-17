from __future__ import annotations

import json
from abc import ABC, abstractmethod

from schemas.types import UnifiedLLMRequest, ALL_ROLES


TokenEstimation = dict[str, int]
"""Keys: 'system', 'user', 'assistant', 'tool', 'total'"""

class BaseTokenEstimator(ABC):
    def estimate(self, request: UnifiedLLMRequest, roles: list[str] | str | None = None) -> TokenEstimation:
        if roles is None:
            target_roles = list(ALL_ROLES)
        elif isinstance(roles, str):
            target_roles = [roles]
        else:
            target_roles = list(roles)

        result: TokenEstimation = {role: _estimate_by_role(request, self._count, role) for role in target_roles}
        result["total"] = sum(result[r] for r in target_roles)
        return result

    @abstractmethod
    def _count(self, text: str) -> int:
        raise NotImplementedError


class ClaudeTokenEstimator(BaseTokenEstimator):
    _CHARS_PER_TOKEN = 3.5

    def _count(self, text: str) -> int:
        return int(len(text) / self._CHARS_PER_TOKEN)


class OpenAICompatibleTokenEstimator(BaseTokenEstimator):
    def __init__(self) -> None:
        import tiktoken
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _count(self, text: str) -> int:
        return len(self._enc.encode(text))


def _estimate_by_role(request: UnifiedLLMRequest, count: ..., role: str) -> int:
    if role == "system":
        system_tokens = count(request.system_prompt or "") + (count(json.dumps(request.tool_schemas)) if request.tool_schemas else 0)
        return system_tokens
    elif role == "user":
        user_tokens = 0
        for msg in request.messages:
            if msg.role == "user":
                user_tokens += count(msg.content)
        return user_tokens
    elif role == "assistant":
        assistant_tokens = 0
        for msg in request.messages:
            if msg.role == "assistant":
                tokens = count(msg.content)
                tool_calls = msg.metadata.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        tokens += count(tc.get("name") or "")
                        tokens += count(tc.get("llm_raw_tool_call_id") or "")
                        args = tc.get("arguments")
                        if args is not None:
                            tokens += count(json.dumps(args) if not isinstance(args, str) else args)
                assistant_tokens += tokens
        return assistant_tokens
    elif role == "tool":
        tool_tokens = 0
        for msg in request.messages:
            if msg.role == "tool":
                tokens = count(msg.content)
                tokens += count(msg.metadata.get("tool_name") or "")
                tokens += count(msg.metadata.get("llm_raw_tool_call_id") or "")
                tool_tokens += tokens
        return tool_tokens
    else:
        raise ValueError(f"Unknown role: {role!r}")


class TokenEstimatorFactory:
    _REGISTRY: dict[str, type[BaseTokenEstimator]] = {
        "claude":   ClaudeTokenEstimator,
        "openai":   OpenAICompatibleTokenEstimator,
        "deepseek": OpenAICompatibleTokenEstimator,
        "qwen":     OpenAICompatibleTokenEstimator,
        "kimi":     OpenAICompatibleTokenEstimator,
        "glm":      OpenAICompatibleTokenEstimator,
        "minmax":   OpenAICompatibleTokenEstimator,
    }
    _cache: dict[str, BaseTokenEstimator] = {}

    @classmethod
    def get_estimator(cls, provider_name: str) -> BaseTokenEstimator:
        if provider_name not in cls._cache:
            estimator_cls = cls._REGISTRY.get(provider_name)
            if estimator_cls is None:
                raise ValueError(f"Unknown LLM provider: {provider_name!r}")
            cls._cache[provider_name] = estimator_cls()
        return cls._cache[provider_name]
