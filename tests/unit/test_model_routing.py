from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from agent.models.model_routing.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitState,
    ProviderCircuitBreaker,
)
from agent.models.model_routing.provider_router import ModelSelector
from schemas.errors import LLMNormalizedError, LLMNormalizedErrorCode
from schemas.task import LLMProviderCapabilities


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make_config(**overrides) -> CircuitBreakerConfig:
    defaults = dict(
        enabled=True,
        immediate_open_codes=frozenset(["AUTH_FAILED", "PERMISSION_DENIED", "QUOTA_EXCEEDED", "CONFIG_ERROR"]),
        failure_threshold=2,
        failure_window_seconds=60,
        half_open_success_threshold=2,
        cooloff_jitter_ratio=0.0,   # zero jitter for deterministic tests
        default_cooloff_seconds=30,
        cooloff_seconds={
            "AUTH_FAILED": 3600,
            "PERMISSION_DENIED": 600,
            "QUOTA_EXCEEDED": 3600,
            "RATE_LIMITED": 60,
            "HTTP_5XX": 15,
            "PROVIDER_OVERLOADED": 30,
            "NETWORK_ERROR": 10,
            "TIMEOUT": 10,
            "CONTENT_FILTERED": 30,
            "INPUT_CONTENT_POLICY": 30,
            "RESPONSE_ERROR": 15,
            "RESPONSE_PARSE_ERROR": 10,
            "CONTEXT_TOO_LONG": 0,
            "OUTPUT_TOO_LONG": 0,
            "INVALID_REQUEST": 0,
            "CONFIG_ERROR": 3600,
        },
    )
    defaults.update(overrides)
    return CircuitBreakerConfig(**defaults)


def _make_error(code: LLMNormalizedErrorCode, retry_after: float | None = None) -> LLMNormalizedError:
    return LLMNormalizedError(code, "test error", retry_after=retry_after)


def _make_capabilities(names: list[str]) -> list[LLMProviderCapabilities]:
    return [
        LLMProviderCapabilities(
            name=n,
            cognitive_complexity=["L2", "L3"],
            best_scenarios=["general"],
            cost_tier="medium",
            latency_tier="medium",
            context_size=32000,
        )
        for n in names
    ]


def _make_selector(names: list[str], cb_cfg_dict: dict | None = None) -> ModelSelector:
    inner = cb_cfg_dict if cb_cfg_dict is not None else {
        "enabled": True,
        "immediate_open_codes": ["AUTH_FAILED", "PERMISSION_DENIED", "QUOTA_EXCEEDED", "CONFIG_ERROR"],
        "failure_threshold": 2,
        "failure_window_seconds": 60,
        "half_open_success_threshold": 2,
        "cooloff_jitter_ratio": 0.0,
        "default_cooloff_seconds": 30,
        "cooloff_seconds": {
            "AUTH_FAILED": 3600, "PERMISSION_DENIED": 600, "QUOTA_EXCEEDED": 3600,
            "RATE_LIMITED": 60, "HTTP_5XX": 15, "PROVIDER_OVERLOADED": 30,
            "NETWORK_ERROR": 10, "TIMEOUT": 10, "CONTENT_FILTERED": 30,
            "INPUT_CONTENT_POLICY": 30, "RESPONSE_ERROR": 15, "RESPONSE_PARSE_ERROR": 10,
            "CONTEXT_TOO_LONG": 0, "OUTPUT_TOO_LONG": 0, "INVALID_REQUEST": 0, "CONFIG_ERROR": 3600,
        },
    }
    cfg_data = {"model_selector": {"circuit_breaker": inner}}

    def _nested_get(key, default=None):
        parts = key.split(".")
        val = cfg_data
        for p in parts:
            if not isinstance(val, dict):
                return default
            if p not in val:
                return default
            val = val[p]
        return val

    config = MagicMock()
    config.get = _nested_get
    tracer = MagicMock()
    tracer.start_span.return_value.__enter__ = MagicMock(return_value=MagicMock())
    tracer.start_span.return_value.__exit__ = MagicMock(return_value=False)
    logger = MagicMock()
    return ModelSelector(
        config=config,
        logger=logger,
        tracer=tracer,
        event_bus=MagicMock(),
        provider_capabilities=_make_capabilities(names),
        enable_fallback=True,
    )


# ---------------------------------------------------------------------------
# ProviderCircuitBreaker unit tests
# ---------------------------------------------------------------------------

class TestCircuitBreakerStateTransitions:

    def test_immediate_open_on_auth_failed(self):
        cb = ProviderCircuitBreaker(provider_name="p1")
        cfg = _make_config()
        now = _now()
        cooloff = cb.record_failure(_make_error(LLMNormalizedErrorCode.AUTH_FAILED), cfg, now)
        assert cb.state == CircuitState.OPEN
        assert cooloff == pytest.approx(3600, rel=0.01)

    def test_threshold_open_on_network_error(self):
        cb = ProviderCircuitBreaker(provider_name="p1")
        cfg = _make_config(failure_threshold=2)
        now = _now()
        # first failure — should NOT open
        cooloff1 = cb.record_failure(_make_error(LLMNormalizedErrorCode.NETWORK_ERROR), cfg, now)
        assert cb.state == CircuitState.CLOSED
        assert cooloff1 == 0.0
        # second failure — should open
        cooloff2 = cb.record_failure(_make_error(LLMNormalizedErrorCode.NETWORK_ERROR), cfg, now)
        assert cb.state == CircuitState.OPEN
        assert cooloff2 == pytest.approx(10, rel=0.01)

    def test_failure_window_reset(self):
        cb = ProviderCircuitBreaker(provider_name="p1")
        cfg = _make_config(failure_threshold=2, failure_window_seconds=5)
        t0 = _now()
        cb.record_failure(_make_error(LLMNormalizedErrorCode.NETWORK_ERROR), cfg, t0)
        assert cb.failure_count == 1
        # second failure arrives after window expires — counter resets
        t1 = t0 + timedelta(seconds=10)
        cooloff = cb.record_failure(_make_error(LLMNormalizedErrorCode.NETWORK_ERROR), cfg, t1)
        assert cb.state == CircuitState.CLOSED
        assert cooloff == 0.0
        assert cb.failure_count == 1

    def test_cooloff_by_error_code(self):
        cfg = _make_config(failure_threshold=1)  # open on first failure for threshold codes
        now = _now()
        cases = [
            (LLMNormalizedErrorCode.AUTH_FAILED, 3600),   # immediate open
            (LLMNormalizedErrorCode.RATE_LIMITED, 60),    # threshold open (threshold=1)
            (LLMNormalizedErrorCode.HTTP_5XX, 15),        # threshold open (threshold=1)
        ]
        for code, expected in cases:
            cb = ProviderCircuitBreaker(provider_name="p")
            cooloff = cb.record_failure(_make_error(code), cfg, now)
            assert cooloff == pytest.approx(expected, rel=0.01), f"wrong cooloff for {code}"

    def test_context_too_long_does_not_open(self):
        cb = ProviderCircuitBreaker(provider_name="p1")
        cfg = _make_config()
        now = _now()
        cooloff = cb.record_failure(_make_error(LLMNormalizedErrorCode.CONTEXT_TOO_LONG), cfg, now)
        assert cb.state == CircuitState.CLOSED
        assert cooloff == 0.0

    def test_retry_after_overrides_cooloff(self):
        cb = ProviderCircuitBreaker(provider_name="p1")
        cfg = _make_config(failure_threshold=1)  # open on first failure
        now = _now()
        cooloff = cb.record_failure(
            _make_error(LLMNormalizedErrorCode.RATE_LIMITED, retry_after=120), cfg, now
        )
        assert cooloff == pytest.approx(120, rel=0.01)

    def test_jitter_within_bounds(self):
        cfg = _make_config(cooloff_jitter_ratio=0.15, failure_threshold=1)
        base = 60.0
        results = set()
        for _ in range(50):
            cb = ProviderCircuitBreaker(provider_name="p1")
            cooloff = cb.record_failure(_make_error(LLMNormalizedErrorCode.RATE_LIMITED), cfg, _now())
            assert base * 0.85 <= cooloff <= base * 1.15
            results.add(round(cooloff, 2))
        # with 50 samples and 15% jitter, we expect variation
        assert len(results) > 1

    def test_open_to_half_open_on_cooloff_expiry(self):
        cb = ProviderCircuitBreaker(provider_name="p1")
        cfg = _make_config()
        now = _now()
        cb.record_failure(_make_error(LLMNormalizedErrorCode.AUTH_FAILED), cfg, now)
        assert cb.state == CircuitState.OPEN
        assert not cb.is_available(now)
        # advance past cooloff
        future = now + timedelta(seconds=3601)
        assert cb.is_available(future)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_to_closed_requires_success_threshold(self):
        cb = ProviderCircuitBreaker(provider_name="p1")
        cfg = _make_config(half_open_success_threshold=2)
        now = _now()
        cb.record_failure(_make_error(LLMNormalizedErrorCode.AUTH_FAILED), cfg, now)
        cb.state = CircuitState.HALF_OPEN  # simulate cooloff expired
        # first success — not yet closed
        recovered = cb.record_success(cfg)
        assert not recovered
        assert cb.state == CircuitState.HALF_OPEN
        # second success — now closed
        recovered = cb.record_success(cfg)
        assert recovered
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_half_open_probe_failure_reopens(self):
        cb = ProviderCircuitBreaker(provider_name="p1")
        cfg = _make_config()
        now = _now()
        cb.record_failure(_make_error(LLMNormalizedErrorCode.HTTP_5XX), cfg, now)
        cb.state = CircuitState.HALF_OPEN
        cooloff = cb.record_probe_failure(_make_error(LLMNormalizedErrorCode.HTTP_5XX), cfg, now)
        assert cb.state == CircuitState.OPEN
        assert cooloff == pytest.approx(15, rel=0.01)
        assert cb.consecutive_success_count == 0

    def test_failure_with_none_error_uses_default_cooloff(self):
        cb = ProviderCircuitBreaker(provider_name="p1")
        cfg = _make_config(immediate_open_codes=frozenset(["UNKNOWN"]))
        now = _now()
        cooloff = cb.record_failure(None, cfg, now)
        assert cb.state == CircuitState.OPEN
        assert cooloff == pytest.approx(30, rel=0.01)

    def test_closed_success_resets_failure_count(self):
        cb = ProviderCircuitBreaker(provider_name="p1")
        cfg = _make_config(failure_threshold=3)
        now = _now()
        cb.record_failure(_make_error(LLMNormalizedErrorCode.NETWORK_ERROR), cfg, now)
        assert cb.failure_count == 1
        cb.record_success(cfg)
        assert cb.failure_count == 0


# ---------------------------------------------------------------------------
# ModelSelector circuit breaker integration tests
# ---------------------------------------------------------------------------

class TestModelSelectorCircuitBreaker:

    def test_get_next_available_skips_open_provider(self):
        sel = _make_selector(["p1", "p2", "p3"])
        now = _now()
        # open p2
        sel._breakers["p2"].record_failure(
            _make_error(LLMNormalizedErrorCode.AUTH_FAILED), sel._cb_config, now
        )
        result = sel.get_next_available_provider(["p1", "p2", "p3"], "p1")
        assert result == "p3"

    def test_get_next_available_returns_none_when_all_open(self):
        sel = _make_selector(["p1", "p2", "p3"])
        now = _now()
        for name in ["p2", "p3"]:
            sel._breakers[name].record_failure(
                _make_error(LLMNormalizedErrorCode.AUTH_FAILED), sel._cb_config, now
            )
        result = sel.get_next_available_provider(["p1", "p2", "p3"], "p1")
        assert result is None

    def test_get_best_recovered_returns_highest_priority(self):
        sel = _make_selector(["p1", "p2", "p3"])
        now = _now()
        # open p1, then simulate cooloff expired → HALF_OPEN
        sel._breakers["p1"].record_failure(
            _make_error(LLMNormalizedErrorCode.AUTH_FAILED), sel._cb_config, now
        )
        future = now + timedelta(seconds=3601)
        sel._breakers["p1"].is_available(future)  # triggers OPEN → HALF_OPEN
        with patch("agent.models.model_routing.provider_router._time_now", return_value=future):
            result = sel.get_best_recovered_provider(["p1", "p2", "p3"], "p2")
        assert result == "p1"

    def test_get_best_recovered_returns_none_when_current_is_primary(self):
        sel = _make_selector(["p1", "p2", "p3"])
        result = sel.get_best_recovered_provider(["p1", "p2", "p3"], "p1")
        assert result is None

    def test_record_provider_success_logs_only_on_half_open_recovery(self):
        sel = _make_selector(["p1"])
        now = _now()
        sel._breakers["p1"].record_failure(
            _make_error(LLMNormalizedErrorCode.HTTP_5XX), sel._cb_config, now
        )
        sel._breakers["p1"].state = CircuitState.HALF_OPEN
        sel._breakers["p1"].consecutive_success_count = 1  # one away from threshold
        sel.record_provider_success("p1")
        sel._logger.info.assert_called()

    def test_circuit_breaker_disabled_is_noop(self):
        sel = _make_selector(["p1", "p2"], cb_cfg_dict={"enabled": False})
        now = _now()
        # record_provider_failure should not change state
        sel.record_provider_failure("p1", _make_error(LLMNormalizedErrorCode.AUTH_FAILED))
        assert sel._breakers["p1"].state == CircuitState.CLOSED
        # get_best_recovered_provider should return None (disabled)
        result = sel.get_best_recovered_provider(["p1", "p2"], "p2")
        assert result is None
