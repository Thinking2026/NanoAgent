from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm.llm_gateway import classify_http_error
from llm.registry import LLMProviderRegistry
from agent.models.model_routing.provider_router import ModelSelector, ModelRoutingDecision
from schemas.errors import (
    PipelineError,
    HttpError,
    LLMNormalizedError,
    LLMNormalizedErrorCode,
    ErrorCategory,
    LLM_NETWORK_ERROR,
    LLM_TIMEOUT,
    LLM_RESPONSE_PARSE_ERROR,
    LLM_RESPONSE_ERROR,
    CONFIG_ERROR,
)
from schemas.types import UnifiedLLMRequest


# ---------------------------------------------------------------------------
# classify_http_error
# ---------------------------------------------------------------------------

def test_classify_429_is_rate_limited():
    exc = HttpError(status=429, body="too many requests", retry_after=10.0)
    err = classify_http_error(exc)
    assert err.code == LLMNormalizedErrorCode.RATE_LIMITED
    assert err.category == ErrorCategory.RATE_LIMIT
    assert err.retry_after == pytest.approx(10.0)


def test_classify_401_is_auth_failed():
    exc = HttpError(status=401, body="unauthorized")
    err = classify_http_error(exc)
    assert err.code == LLMNormalizedErrorCode.AUTH_FAILED
    assert err.category == ErrorCategory.AUTH


def test_classify_403_is_auth_failed():
    exc = HttpError(status=403, body="forbidden")
    err = classify_http_error(exc)
    assert err.code == LLMNormalizedErrorCode.AUTH_FAILED


def test_classify_400_context_too_long():
    exc = HttpError(status=400, body="context_length_exceeded: too many tokens")
    err = classify_http_error(exc)
    assert err.code == LLMNormalizedErrorCode.CONTEXT_TOO_LONG
    assert err.category == ErrorCategory.CONTEXT


def test_classify_400_context_too_long_variant():
    exc = HttpError(status=400, body="maximum context length exceeded")
    err = classify_http_error(exc)
    assert err.code == LLMNormalizedErrorCode.CONTEXT_TOO_LONG


def test_classify_500_is_http_5xx():
    exc = HttpError(status=500, body="internal server error")
    err = classify_http_error(exc)
    assert err.code == LLMNormalizedErrorCode.HTTP_5XX
    assert err.category == ErrorCategory.TRANSIENT


def test_classify_400_non_context_is_5xx():
    exc = HttpError(status=400, body="bad request unrelated")
    err = classify_http_error(exc)
    assert err.code == LLMNormalizedErrorCode.HTTP_5XX


# ---------------------------------------------------------------------------
# classify_agent_error
# ---------------------------------------------------------------------------

def test_classify_network_error():
    exc = PipelineError(code=LLM_NETWORK_ERROR, message="connection refused")
    err = classify_agent_error(exc)
    assert err.code == LLMNormalizedErrorCode.NETWORK_ERROR
    assert err.category == ErrorCategory.TRANSIENT


def test_classify_timeout():
    exc = PipelineError(code=LLM_TIMEOUT, message="timed out")
    err = classify_agent_error(exc)
    assert err.code == LLMNormalizedErrorCode.TIMEOUT


def test_classify_response_parse_error():
    exc = PipelineError(code=LLM_RESPONSE_PARSE_ERROR, message="parse failed")
    err = classify_agent_error(exc)
    assert err.code == LLMNormalizedErrorCode.RESPONSE_PARSE_ERROR


def test_classify_response_error():
    exc = PipelineError(code=LLM_RESPONSE_ERROR, message="bad response")
    err = classify_agent_error(exc)
    assert err.code == LLMNormalizedErrorCode.RESPONSE_ERROR


def test_classify_config_error():
    exc = PipelineError(code=CONFIG_ERROR, message="missing key")
    err = classify_agent_error(exc)
    assert err.code == LLMNormalizedErrorCode.CONFIG_ERROR


def test_classify_unknown_agent_error_defaults_to_response_error():
    exc = PipelineError(code="SOME_UNKNOWN_CODE", message="unknown")
    err = classify_agent_error(exc)
    assert err.code == LLMNormalizedErrorCode.RESPONSE_ERROR

# ---------------------------------------------------------------------------
# LLMProviderRegistry
# ---------------------------------------------------------------------------

def make_mock_provider(name: str):
    provider = MagicMock()
    provider.provider_name = name
    return provider


def test_registry_register_and_get():
    registry = LLMProviderRegistry()
    p = make_mock_provider("claude")
    registry.register(p)
    assert registry.get("claude") is p


def test_registry_get_unknown_raises():
    registry = LLMProviderRegistry()
    with pytest.raises(Exception, match="Unknown LLM provider"):
        registry.get("nonexistent")


def test_registry_list_providers():
    registry = LLMProviderRegistry()
    registry.register(make_mock_provider("claude"))
    registry.register(make_mock_provider("openai"))
    providers = registry.list_providers()
    assert sorted(providers) == ["claude", "openai"]


def test_registry_init_with_providers():
    p1 = make_mock_provider("claude")
    p2 = make_mock_provider("openai")
    registry = LLMProviderRegistry([p1, p2])
    assert registry.get("claude") is p1
    assert registry.get("openai") is p2


def test_registry_overwrite_provider():
    registry = LLMProviderRegistry()
    p1 = make_mock_provider("claude")
    p2 = make_mock_provider("claude")
    registry.register(p1)
    registry.register(p2)
    assert registry.get("claude") is p2


# ---------------------------------------------------------------------------
# ModelSelector
# ---------------------------------------------------------------------------

def make_selector(names: list[str], enable_fallback: bool = False) -> ModelSelector:
    return ModelSelector(priority_chain=names, enable_fallback=enable_fallback)


def test_model_selector_primary_is_first():
    selector = make_selector(["claude", "openai"])
    decision = selector.route()
    assert decision.primary == "claude"


def test_model_selector_no_fallback_by_default():
    selector = make_selector(["claude", "openai"], enable_fallback=False)
    decision = selector.route()
    assert decision.fallbacks == []


def test_model_selector_with_fallback():
    selector = make_selector(["claude", "openai", "deepseek"], enable_fallback=True)
    decision = selector.route()
    assert decision.primary == "claude"
    assert decision.fallbacks == ["openai", "deepseek"]


def test_model_selector_single_provider_no_fallbacks():
    selector = make_selector(["claude"], enable_fallback=True)
    decision = selector.route()
    assert decision.primary == "claude"
    assert decision.fallbacks == []


def test_model_selector_empty_priority_chain_raises():
    with pytest.raises(Exception):
        ModelSelector(priority_chain=[])
