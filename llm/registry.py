from __future__ import annotations

import time
from typing import Iterable

from llm.llm_api import BaseLLMClient
from schemas import LLMRequest, LLMResponse, build_error


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
            raise build_error(
                "LLM_PROVIDER_NOT_FOUND",
                f"Unknown LLM provider: {provider_name}. Available providers: {available}",
            ) from exc

    def list_providers(self) -> list[str]:
        return sorted(self._providers)


class DynamicLLMClient(BaseLLMClient):
    provider_name = "dynamic"

    def __init__(self, registry: LLMProviderRegistry, default_provider: str) -> None:
        self._registry = registry
        self._provider_name = default_provider

    @property
    def current_provider_name(self) -> str:
        return self._provider_name

    def use_provider(self, provider_name: str) -> None:
        self._registry.get(provider_name)
        self._provider_name = provider_name

    def generate(self, request):
        provider = self._registry.get(self._provider_name)
        return provider.generate(request)


class FallbackLLMClient(BaseLLMClient):
    provider_name = "fallback"

    def __init__(
        self,
        registry: LLMProviderRegistry,
        provider_priority: list[str],
        retry_delays: tuple[float, ...] = (1.0, 2.0, 4.0),
    ) -> None:
        if not provider_priority:
            raise build_error("LLM_CONFIG_ERROR", "provider_priority cannot be empty")
        if not retry_delays:
            raise build_error("LLM_CONFIG_ERROR", "retry_delays cannot be empty")
        self._registry = registry
        self._provider_priority = provider_priority
        self._retry_delays = retry_delays

    def generate(self, request: LLMRequest) -> LLMResponse:
        failure_messages: list[str] = []
        max_attempts = len(self._retry_delays)

        for provider_name in self._provider_priority:
            provider = self._registry.get(provider_name)
            for attempt_idx in range(max_attempts):
                try:
                    return provider.generate(request)
                except Exception as exc:
                    failure_messages.append(
                        f"{provider_name}[attempt {attempt_idx + 1}/{max_attempts}]: {exc}"
                    )
                    if attempt_idx < max_attempts - 1:
                        time.sleep(self._retry_delays[attempt_idx])

        raise build_error(
            "LLM_ALL_PROVIDERS_FAILED",
            "All configured LLM providers failed. " + " | ".join(failure_messages),
        )
