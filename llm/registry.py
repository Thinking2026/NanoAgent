from __future__ import annotations

from typing import Iterable

from llm.llm_api import BaseLLMClient
from schemas import build_error


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
