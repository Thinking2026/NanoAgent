from __future__ import annotations

import os

from llm.providers.openai_api import OpenAILLMClient
from schemas import CONFIG_ERROR, UnifiedLLMRequest, LLMResponse, build_pipeline_error


class DeepSeekLLMClient(OpenAILLMClient):
    provider_name = "deepseek"

    def __init__(self, thinking_model: str = "deepseek-reasoner", **kwargs) -> None:
        super().__init__(**kwargs)
        self._thinking_model = thinking_model

    @classmethod
    def from_settings(
        cls,
        api_key: str | None,
        model: str,
        base_url: str = "https://api.deepseek.com",
        timeout: float = 60.0,
        max_tokens: int = 128000,
        enable_thinking: bool = False,
        thinking_model: str = "deepseek-reasoner",
    ) -> "DeepSeekLLMClient":
        resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not resolved_api_key:
            raise build_pipeline_error(CONFIG_ERROR, "Missing API key for DeepSeek client.")
        return cls(
            api_key=resolved_api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            thinking_model=thinking_model,
        )

    def generate(self, request: UnifiedLLMRequest) -> LLMResponse:
        if request.enable_thinking or self._enable_thinking:
            from dataclasses import replace
            request = replace(request, model_override=request.model_override or self._thinking_model)
        return super().generate(request)
