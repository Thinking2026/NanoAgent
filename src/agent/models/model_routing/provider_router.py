from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from config.config import ConfigReader
from infra.observability.tracing.tracer import Tracer
from schemas.errors import CONFIG_ERROR, UNKNOWN_LOGIC_ERROR, build_config_error, LLMNormalizedError, build_pipeline_error
from schemas.task import (
    LLMProviderCapabilities,
    ModelRoutingDecision,
    Task,
    L1, L2, L3, L4,
)
from utils.log.log import Logger, zap
from utils.time.time import now as _time_now

from agent.models.model_routing.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitState,
    ProviderCircuitBreaker,
)

# Ordered from simplest to most complex; used to derive accepted tiers by level.
_COMPLEXITY_LEVELS = [L1, L2, L3, L4]

_COST_PREFERENCE_KEYWORDS    = {"cheap", "cost", "budget", "economy", "affordable"}
_SPEED_PREFERENCE_KEYWORDS   = {"fast", "speed", "quick", "low-latency", "realtime", "real-time"}
_TOOL_STRENGTH_KEYWORDS      = {"tool_use", "tool use", "function_calling", "function calling", "tool"}
_REASONING_STRENGTH_KEYWORDS = {"reasoning", "multi-step", "multi_step", "chain-of-thought", "cot"}


def _tier_label(level: int) -> str:
    """Convert a TaskComplexity.level (1-4) to the canonical tier label used by providers."""
    return f"L{level}"


# ---------------------------------------------------------------------------
# RoutingStrategy protocol — swap at any time
# ---------------------------------------------------------------------------

@runtime_checkable
class RoutingStrategy(Protocol):
    """Pluggable strategy: given a task + capabilities, return an ordered provider list."""

    def select(
        self,
        task: Task,
        candidates: list[LLMProviderCapabilities],
    ) -> list[str]:
        """Return provider tool_names in priority order (best first). Must be non-empty."""
        ...


# ---------------------------------------------------------------------------
# Built-in strategy 1: capability-match scoring (default)
# ---------------------------------------------------------------------------

class CapabilityMatchStrategy:
    """Score each provider against the task and rank by score.

    Task attributes used and their scoring contribution:
      task_type / intent tokens  → match against provider best_scenarios        (+3 each hit)
      complexity.use_cases       → match against provider best_scenarios        (+3 each hit)
      complexity.features        → match against provider top_strengths         (+2 each hit)
      complexity.level           → provider cognitive_complexity tier match     (+2)
      required_tools non-empty   → provider has tool-use strength               (+2)
      user preference "cost"     → penalise high cost_tier                      (-1)
      user preference "speed"    → penalise non-fast latency_tier               (-1)
    """

    def select(
        self,
        task: Task,
        candidates: list[LLMProviderCapabilities],
    ) -> list[str]:
        if not candidates:
            raise build_pipeline_error(LLM_CONFIG_ERROR, "no provider candidates available")
        if task is None:
            return [c.name for c in candidates]

        # --- derive signals from Task ---

        # Scenario tokens: task_type, intent, and the use_cases of the matched complexity level
        scenario_tokens: set[str] = set()
        if task.task_type:
            scenario_tokens.update(task.task_type.lower().split())
        if task.intent:
            scenario_tokens.update(task.intent.lower().split())
        for use_case in task.complexity.use_cases:
            scenario_tokens.update(use_case.lower().split())

        # Feature tokens: the features list of the matched complexity level
        # e.g. L3 → ["多步推理", "代码", "分析"]
        complexity_feature_tokens: set[str] = {f.lower() for f in task.complexity.features}

        # Accepted cognitive-complexity tiers: the task's level and all levels below it
        # (a provider that handles L3 can certainly handle L2 tasks)
        accepted_tiers: list[str] = [
            _tier_label(lvl.level)
            for lvl in _COMPLEXITY_LEVELS
            if lvl.level >= task.complexity.level
        ]

        needs_tools = bool(task.required_tools)

        # Derive cost/speed preference from user preference entries
        prefer_low_cost = False
        prefer_low_latency = False
        for pref_entry in task.related_user_preference_entries:
            keywords_lower = {kw.lower() for kw in pref_entry.entry.keywords}
            content_tokens = set(pref_entry.entry.content.lower().split())
            combined = keywords_lower | content_tokens
            if combined & _COST_PREFERENCE_KEYWORDS:
                prefer_low_cost = True
            if combined & _SPEED_PREFERENCE_KEYWORDS:
                prefer_low_latency = True

        # --- score each candidate ---
        scored: list[tuple[int, str]] = []
        for cap in candidates:
            score = 0
            cap_scenarios_lower = [s.lower() for s in cap.best_scenarios]
            cap_strengths_lower  = [s.lower() for s in cap.top_strengths]

            # Scenario match: task_type / intent / use_cases vs provider best_scenarios
            for scenario in cap_scenarios_lower:
                scenario_words = set(scenario.split())
                if scenario_words & scenario_tokens:
                    score += 3

            # Complexity feature match: complexity.features vs provider top_strengths
            for feature_token in complexity_feature_tokens:
                if any(feature_token in strength for strength in cap_strengths_lower):
                    score += 2

            # Cognitive complexity tier match
            if any(t in cap.cognitive_complexity for t in accepted_tiers):
                score += 2

            # Tool-use capability
            if needs_tools and any(
                kw in strength
                for kw in _TOOL_STRENGTH_KEYWORDS
                for strength in cap_strengths_lower
            ):
                score += 2

            # Cost/latency penalties from user preferences
            if prefer_low_cost and cap.cost_tier == "high":
                score -= 1
            if prefer_low_latency and cap.latency_tier != "fast":
                score -= 1

            scored.append((score, cap.name))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [name for _, name in scored]


# ---------------------------------------------------------------------------
# Built-in strategy 2: cost/latency first
# ---------------------------------------------------------------------------

class CostLatencyStrategy:
    """Rank providers by cost and latency tiers, ignoring task semantics.

    cost_tier rank:    low=0, medium=1, high=2
    latency_tier rank: fast=0, medium=1, slow=2
    """

    _COST_RANK    = {"low": 0, "medium": 1, "high": 2}
    _LATENCY_RANK = {"fast": 0, "medium": 1, "slow": 2}

    def __init__(self, weight_cost: float = 0.5, weight_latency: float = 0.5) -> None:
        self._wc = weight_cost
        self._wl = weight_latency

    def select(
        self,
        _task: Task | None,
        candidates: list[LLMProviderCapabilities],
    ) -> list[str]:
        def _rank(cap: LLMProviderCapabilities) -> float:
            c   = self._COST_RANK.get(cap.cost_tier, 1)
            lat = self._LATENCY_RANK.get(cap.latency_tier, 1)
            return self._wc * c + self._wl * lat

        return [c.name for c in sorted(candidates, key=_rank)]


# ---------------------------------------------------------------------------
# ModelSelector — held by Pipeline
# ---------------------------------------------------------------------------

class ModelSelector:
    """Selects the primary provider and fallback chain for a task.

    The selection algorithm is fully delegated to a RoutingStrategy, which can
    be replaced at any time via set_strategy().

    At runtime, circuit breakers track provider health and filter the chain
    returned by route() so that degraded providers are skipped automatically.
    """

    def __init__(
        self,
        config: ConfigReader,
        logger: Logger,
        tracer: Tracer,
        provider_capabilities: list[LLMProviderCapabilities],
        strategy: RoutingStrategy | None = None,
        enable_fallback: bool = False,
    ) -> None:
        if not provider_capabilities:
            raise build_pipeline_error(LLM_CONFIG_ERROR, "provider_capabilities cannot be empty")
        self._config = config
        self._logger = logger
        self._tracer = tracer
        self._capabilities = provider_capabilities
        self._strategy: RoutingStrategy = strategy or CapabilityMatchStrategy()
        self._enable_fallback = enable_fallback

        cb_cfg_dict = config.get("model_selector.circuit_breaker", {})
        if not isinstance(cb_cfg_dict, dict):
            cb_cfg_dict = {}
        self._cb_config = CircuitBreakerConfig.from_dict(cb_cfg_dict)
        self._breakers: dict[str, ProviderCircuitBreaker] = {
            cap.name: ProviderCircuitBreaker(provider_name=cap.name)
            for cap in provider_capabilities
        }

    def set_strategy(self, strategy: RoutingStrategy) -> None:
        """Replace the routing strategy at runtime."""
        self._strategy = strategy

    def route(
        self,
        task: Task,
        enable_fallback: bool | None = None,
        excluded_providers: set[str] | None = None,
    ) -> ModelRoutingDecision:
        """Return primary provider name and fallback chain."""
        use_fallback = enable_fallback if enable_fallback is not None else self._enable_fallback
        excluded = excluded_providers or set()
        candidates = [c for c in self._capabilities if c.name not in excluded]
        if not candidates:
            raise build_config_error(CONFIG_ERROR, "no available providers after applying exclusions")

        with self._tracer.start_span(
            "model.route",
            "routing",
            {
                "task_id": task.id if task else None,
                "task_type": task.task_type if task else None,
                "strategy": self._strategy.__class__.__name__,
                "candidate_count": len(candidates),
                "enable_fallback": use_fallback,
            },
        ) as span:
            ordered = self._strategy.select(task, candidates)
            if not ordered:
                raise build_pipeline_error(UNKNOWN_LOGIC_ERROR, "routing strategy returned an empty provider list")

            primary = ordered[0]
            fallbacks = ordered[1:] if use_fallback else []
            span.add_attributes({"primary": primary, "fallbacks": fallbacks})
        self._logger.info(
            "Model routing selected providers",
            zap.any("task_id", task.id if task else None),
            zap.any("task_type", task.task_type if task else None),
            zap.any("strategy", self._strategy.__class__.__name__),
            zap.any("candidate_count", len(candidates)),
            zap.any("primary", primary),
            zap.any("fallbacks", fallbacks),
        )
        return ModelRoutingDecision(primary=primary, fallbacks=fallbacks)

    # ------------------------------------------------------------------
    # Circuit breaker public API
    # ------------------------------------------------------------------

    def record_provider_failure(
        self,
        provider: str,
        error: LLMNormalizedError | None,
    ) -> None:
        """Record a provider failure and update circuit state."""
        if not self._cb_config.enabled:
            return
        breaker = self._breakers.get(provider)
        if breaker is None:
            return

        now = _time_now()
        state_before = breaker.state

        if breaker.state == CircuitState.HALF_OPEN:
            cooloff = breaker.record_probe_failure(error, self._cb_config, now)
        else:
            cooloff = breaker.record_failure(error, self._cb_config, now)

        error_code = error.code.value if error and hasattr(error, "code") else "UNKNOWN"
        self._logger.warning(
            "Provider failure recorded",
            zap.any("provider", provider),
            zap.any("error_code", error_code),
            zap.any("state_before", state_before.value),
            zap.any("state_after", breaker.state.value),
            zap.any("cooloff_seconds", round(cooloff, 1)),
            zap.any("failure_count", breaker.failure_count),
        )
        with self._tracer.start_span(
            "model.provider_failure",
            "routing",
            {
                "provider": provider,
                "error_code": error_code,
                "state_before": state_before.value,
                "state_after": breaker.state.value,
                "cooloff_seconds": round(cooloff, 1),
            },
        ):
            pass

    def record_provider_success(self, provider: str) -> None:
        """Record a provider success; logs only on HALF_OPEN → CLOSED recovery."""
        if not self._cb_config.enabled:
            return
        breaker = self._breakers.get(provider)
        if breaker is None:
            return

        recovered = breaker.record_success(self._cb_config)
        if recovered:
            self._logger.info(
                "Provider recovered from HALF_OPEN to CLOSED",
                zap.any("provider", provider),
            )
            with self._tracer.start_span(
                "model.provider_recovered",
                "routing",
                {"provider": provider},
            ):
                pass

    def get_next_available_provider(
        self,
        provider_chain: list[str],
        current: str,
    ) -> str | None:
        """Return the next provider after *current* that is available (not OPEN)."""
        now = _time_now()
        try:
            current_idx = provider_chain.index(current)
        except ValueError:
            current_idx = -1

        skipped: list[tuple[str, float]] = []
        for provider in provider_chain[current_idx + 1:]:
            breaker = self._breakers.get(provider)
            if breaker is None or breaker.is_available(now):
                if skipped:
                    self._logger.info(
                        "Skipped unavailable providers during fallback",
                        zap.any("skipped", [
                            {"provider": p, "cooloff_remaining_s": round(r, 1)}
                            for p, r in skipped
                        ]),
                        zap.any("selected", provider),
                    )
                return provider
            skipped.append((provider, breaker.cooloff_remaining(now)))

        if skipped:
            self._logger.warning(
                "All fallback providers unavailable",
                zap.any("current", current),
                zap.any("unavailable", [
                    {"provider": p, "cooloff_remaining_s": round(r, 1)}
                    for p, r in skipped
                ]),
            )
        return None

    def get_best_recovered_provider(
        self,
        provider_chain: list[str],
        current: str,
    ) -> str | None:
        """Return the highest-priority provider ahead of *current* that has recovered."""
        if not self._cb_config.enabled:
            return None
        now = _time_now()
        try:
            current_idx = provider_chain.index(current)
        except ValueError:
            return None

        checked = 0
        for provider in provider_chain[:current_idx]:
            checked += 1
            breaker = self._breakers.get(provider)
            if breaker is not None and breaker.is_available(now):
                with self._tracer.start_span(
                    "model.provider_recovery_check",
                    "routing",
                    {
                        "current": current,
                        "recovered": provider,
                        "checked_count": checked,
                    },
                ):
                    pass
                return provider
        return None
