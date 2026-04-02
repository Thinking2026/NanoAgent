from __future__ import annotations

from abc import ABC, abstractmethod
import time

from schemas import AgentError, ChatMessage, LLMRequest, LLMResponse, build_error
from tracing import SpanHandle, Tracer


class BaseLLMClient(ABC):
    provider_name: str = "base"

    def set_tracer(self, tracer: Tracer | None) -> "BaseLLMClient":
        self._tracer = tracer
        return self

    def _start_span(
        self,
        name: str,
        attributes: dict | None = None,
    ) -> SpanHandle:
        tracer = getattr(self, "_tracer", None)
        if tracer is None:
            return SpanHandle(None)
        return tracer.start_span(name=name, kind="llm", attributes=attributes)

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError


class DynamicLLMClient(BaseLLMClient):
    provider_name = "dynamic"

    def __init__(self, registry: "LLMProviderRegistry", default_provider: str) -> None:
        self._registry = registry
        self._provider_name = default_provider

    @property
    def current_provider_name(self) -> str:
        return self._provider_name

    def use_provider(self, provider_name: str) -> None:
        self._registry.get(provider_name)
        self._provider_name = provider_name

    def generate(self, request: LLMRequest) -> LLMResponse:
        provider = self._registry.get(self._provider_name)
        return provider.generate(request)


class FallbackLLMClient(BaseLLMClient):
    provider_name = "fallback"

    def __init__(
        self,
        registry: "LLMProviderRegistry",
        provider_priority: list[str],
        max_attempts: int = 4,
        retry_delays: tuple[float, ...] = (1.0, 2.0, 4.0),
        enable_provider_fallback: bool = False,
    ) -> None:
        if not provider_priority:
            raise build_error("LLM_CONFIG_ERROR", "provider_priority cannot be empty")
        if max_attempts <= 0:
            raise build_error("LLM_CONFIG_ERROR", "max_attempts must be greater than 0")
        self._registry = registry
        self._provider_priority = provider_priority
        self._max_attempts = max_attempts
        self._retry_delays = self._normalize_retry_delays(retry_delays, max_attempts)
        self._enable_provider_fallback = enable_provider_fallback

    def generate(self, request: LLMRequest) -> LLMResponse:
        failure_messages: list[str] = []
        provider_names = (
            self._provider_priority
            if self._enable_provider_fallback
            else [self._provider_priority[0]]
        )

        for provider_name in provider_names:
            provider = self._registry.get(provider_name)
            for attempt_idx in range(self._max_attempts):
                try:
                    return provider.generate(request)
                except AgentError as exc:
                    repaired = self._try_parse_error_self_repair(provider, request, exc)
                    if repaired is not None:
                        return repaired

                    failure_messages.append(
                        f"{provider_name}[attempt {attempt_idx + 1}/{self._max_attempts}]: {exc}"
                    )
                    if attempt_idx < self._max_attempts - 1:
                        time.sleep(self._retry_delays[attempt_idx])
                except Exception as exc:
                    failure_messages.append(
                        f"{provider_name}[attempt {attempt_idx + 1}/{self._max_attempts}]: {exc}"
                    )
                    if attempt_idx < self._max_attempts - 1:
                        time.sleep(self._retry_delays[attempt_idx])

        raise build_error(
            "LLM_ALL_PROVIDERS_FAILED",
            "All attempted LLM providers failed. " + " | ".join(failure_messages),
        )

    @staticmethod
    def _normalize_retry_delays(
        retry_delays: tuple[float, ...],
        max_attempts: int,
    ) -> tuple[float, ...]:
        target_length = max(0, max_attempts - 1)
        if target_length == 0:
            return ()
        delays = [delay for delay in retry_delays if delay > 0]
        if not delays:
            delays = [1.0]
        while len(delays) < target_length:
            delays.append(delays[-1] * 2)
        return tuple(delays[:target_length])

    def _try_parse_error_self_repair(
        self,
        provider: BaseLLMClient,
        request: LLMRequest,
        error: AgentError,
    ) -> LLMResponse | None:
        if error.code not in {"LLM_RESPONSE_PARSE_ERROR", "LLM_RESPONSE_ERROR"}:
            return None

        repair_prompt = (
            "Your previous output could not be parsed by the client. "
            "Please regenerate a valid response following the expected tool-call/text format. "
            "Below is the parser error and raw output details captured by client.\n\n"
            f"{error.message}"
        )
        repaired_request = LLMRequest(
            system_prompt=request.system_prompt,
            messages=[*request.messages, ChatMessage(role="user", content=repair_prompt)],
            tools=request.tools,
            context=request.context,
        )
        try:
            return provider.generate(repaired_request)
        except Exception:
            return None
