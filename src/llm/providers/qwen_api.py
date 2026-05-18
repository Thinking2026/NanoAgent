from __future__ import annotations

import os

from llm.providers.openai_api import OpenAILLMClient
from schemas import CONFIG_ERROR, build_pipeline_error


class QwenLLMClient(OpenAILLMClient):
    provider_name = "qwen"

    @classmethod
    def from_settings(
        cls,
        api_key: str | None,
        model: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout: float = 60.0,
        max_tokens: int = 4096,
        enable_thinking: bool = False,
        default_temperature: float = 1.0,
    ) -> "QwenLLMClient":
        resolved_api_key = api_key or os.getenv("QWEN_API_KEY")
        if not resolved_api_key:
            raise build_pipeline_error(CONFIG_ERROR, "Missing API key for Qwen client.")
        return cls(
            api_key=resolved_api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            default_temperature=default_temperature,
        )
