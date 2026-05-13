from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from schemas.errors import LLMNormalizedError


class CircuitState(str, Enum):
    CLOSED    = "CLOSED"     # healthy, serving normally
    OPEN      = "OPEN"       # cooling off, rejecting requests
    HALF_OPEN = "HALF_OPEN"  # cooloff expired, allowing one probe


@dataclass
class CircuitBreakerConfig:
    enabled: bool
    immediate_open_codes: frozenset[str]
    failure_threshold: int
    failure_window_seconds: float
    half_open_success_threshold: int
    cooloff_jitter_ratio: float
    default_cooloff_seconds: float
    cooloff_seconds: dict[str, float]

    @classmethod
    def from_dict(cls, cfg: dict) -> CircuitBreakerConfig:
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            immediate_open_codes=frozenset(cfg.get("immediate_open_codes", [])),
            failure_threshold=int(cfg.get("failure_threshold", 2)),
            failure_window_seconds=float(cfg.get("failure_window_seconds", 60)),
            half_open_success_threshold=int(cfg.get("half_open_success_threshold", 2)),
            cooloff_jitter_ratio=float(cfg.get("cooloff_jitter_ratio", 0.15)),
            default_cooloff_seconds=float(cfg.get("default_cooloff_seconds", 30)),
            cooloff_seconds={k: float(v) for k, v in cfg.get("cooloff_seconds", {}).items()},
        )


@dataclass
class ProviderCircuitBreaker:
    provider_name: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    consecutive_success_count: int = 0
    last_error_code: str | None = None
    cooloff_until: datetime | None = None
    first_failure_at: datetime | None = None

    def record_failure(
        self,
        error: LLMNormalizedError | None,
        config: CircuitBreakerConfig,
        now: datetime,
    ) -> float:
        """Record a failure. Returns actual cooloff seconds (0 if circuit not opened)."""
        error_code = error.code.value if error and hasattr(error, "code") else "UNKNOWN"
        self.last_error_code = error_code
        self.consecutive_success_count = 0

        should_open = False

        if error_code in config.immediate_open_codes:
            should_open = True
        else:
            # Reset window if first failure is too old
            if self.first_failure_at is not None:
                elapsed = (now - self.first_failure_at).total_seconds()
                if elapsed > config.failure_window_seconds:
                    self.failure_count = 0
                    self.first_failure_at = None

            if self.first_failure_at is None:
                self.first_failure_at = now
            self.failure_count += 1

            if self.failure_count >= config.failure_threshold:
                should_open = True

        if not should_open:
            return 0.0

        return self._open_circuit(error, error_code, config, now)

    def record_probe_failure(
        self,
        error: LLMNormalizedError | None,
        config: CircuitBreakerConfig,
        now: datetime,
    ) -> float:
        """Called when a HALF_OPEN probe fails. Re-opens the circuit."""
        error_code = error.code.value if error and hasattr(error, "code") else "UNKNOWN"
        self.last_error_code = error_code
        self.consecutive_success_count = 0
        self.failure_count = 1
        self.first_failure_at = now
        return self._open_circuit(error, error_code, config, now)

    def record_success(self, config: CircuitBreakerConfig) -> bool:
        """Record a success. Returns True if HALF_OPEN → CLOSED transition occurred."""
        if self.state == CircuitState.HALF_OPEN:
            self.consecutive_success_count += 1
            if self.consecutive_success_count >= config.half_open_success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.consecutive_success_count = 0
                self.first_failure_at = None
                self.cooloff_until = None
                return True
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
            self.first_failure_at = None
        return False

    def is_available(self, now: datetime) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if self.cooloff_until is not None and now >= self.cooloff_until:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        # HALF_OPEN: allow probe
        return True

    def cooloff_remaining(self, now: datetime) -> float:
        if self.state != CircuitState.OPEN or self.cooloff_until is None:
            return 0.0
        return max(0.0, (self.cooloff_until - now).total_seconds())

    def _open_circuit(
        self,
        error: LLMNormalizedError | None,
        error_code: str,
        config: CircuitBreakerConfig,
        now: datetime,
    ) -> float:
        base = config.cooloff_seconds.get(error_code, config.default_cooloff_seconds)

        # retry_after from provider header takes precedence
        if error is not None and getattr(error, "retry_after", None):
            base = float(error.retry_after)

        # skip cooloff for codes with 0s (e.g. CONTEXT_TOO_LONG)
        if base <= 0:
            return 0.0

        jitter = config.cooloff_jitter_ratio * (random.random() * 2 - 1)
        cooloff = base * (1 + jitter)

        self.state = CircuitState.OPEN
        self.cooloff_until = now + timedelta(seconds=cooloff)
        self.failure_count = 0
        self.first_failure_at = None
        return cooloff
