from __future__ import annotations

import os

from llm.providers.openai_api import OpenAILLMClient
from schemas import CONFIG_ERROR, build_pipeline_error
from utils.log.log import Logger, zap


class GLMLLMClient(OpenAILLMClient):
    provider_name = "glm"

    @classmethod
    def from_settings(
        cls,
        api_key: str | None,
        model: str,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        timeout: float = 60.0,
        max_tokens: int = 4096,
        enable_thinking: bool = False,
    ) -> "GLMLLMClient":
        resolved_api_key = api_key or os.getenv("GLM_API_KEY")
        if not resolved_api_key:
            raise build_pipeline_error(CONFIG_ERROR, "Missing API key for GLM client.")
        logger = Logger.get_instance()
        logger.info("GLM provider initialized", zap.any("model", model), zap.any("base_url", base_url))
        return cls(
            api_key=resolved_api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
