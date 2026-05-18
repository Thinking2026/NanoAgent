from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from config.config import ConfigReader
from infra.observability.tracing.tracer import Tracer
from schemas.errors import (
    CONFIG_ERROR,
    UNKNOWN_LOGIC_ERROR,
    LLM_ALL_PROVIDERS_FAILED,
    build_config_error,
    build_pipeline_error,
    LLMNormalizedError,
    PipelineError,
)
from schemas.event_bus import EventBus
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

_COST_PREFERENCE_KEYWORDS  = {"cheap", "cost", "budget", "economy", "affordable"}
_SPEED_PREFERENCE_KEYWORDS = {"fast", "speed", "quick", "low-latency", "realtime", "real-time"}

# Jaccard similarity threshold for scenario phrase matching (tunable via config)
_DEFAULT_JACCARD_THRESHOLD = 0.3

# Tie-break rank: lower is better
_COST_RANK    = {"low": 0, "medium": 1, "high": 2}
_LATENCY_RANK = {"fast": 0, "medium": 1, "slow": 2, "high": 2}


def _jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two phrases."""
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


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
    """Score each provider against the task using a weighted multi-signal model.

    Scoring signals (positive):
      task.task_type exact match in preferred_task_types          → +8
      each routing_hints.required_capability_tags in cap tags     → +5 per tag
      cognitive_complexity tier covers task level                 → +3
      each complexity.use_cases phrase: Jaccard ≥ threshold vs
        provider best_scenarios                                   → +2 per match
      each complexity.features token in capability_tags           → +2 per hit
      required_tools non-empty AND "tool_use" in capability_tags  → +3

    Scoring signals (negative):
      estimated_context_tokens > context_size * 0.7              → -5
      routing_hints.latency_sensitive AND latency_tier != "fast"  → -3
      routing_hints.cost_sensitive AND cost_tier == "high"        → -3
      user preference "cost" AND cost_tier == "high"              → -2
      user preference "speed" AND latency_tier != "fast"          → -2

    Tie-breaking: lower cost_tier rank, then lower latency_tier rank.
    """

    def __init__(self, jaccard_threshold: float = _DEFAULT_JACCARD_THRESHOLD) -> None:
        self._jaccard_threshold = jaccard_threshold

    def select(
        self,
        task: Task,
        candidates: list[LLMProviderCapabilities],
    ) -> list[str]:
        if not candidates:
            raise build_pipeline_error(CONFIG_ERROR, "no provider candidates available")
        if task is None:
            return [c.name for c in candidates]

        hints = task.routing_hints
        task_type_lower = task.task_type.lower() if task.task_type else ""
        required_tags: set[str] = {t.lower() for t in hints.required_capability_tags}
        needs_tools = bool(task.required_tools)
        complexity_feature_tokens: set[str] = {f.lower() for f in task.complexity.features}
        use_cases_lower: list[str] = [uc.lower() for uc in task.complexity.use_cases]

        # Accepted cognitive-complexity tiers: task level and all higher tiers
        # (a provider that handles L4 can certainly handle L2 tasks)
        accepted_tiers: set[str] = {
            _tier_label(lvl.level)
            for lvl in _COMPLEXITY_LEVELS
            if lvl.level >= task.complexity.level
        }

        # Soft preferences from user preference text
        prefer_low_cost = False
        prefer_low_latency = False
        if task.related_user_preference:
            pref_tokens = set(task.related_user_preference.lower().split())
            prefer_low_cost = bool(pref_tokens & _COST_PREFERENCE_KEYWORDS)
            prefer_low_latency = bool(pref_tokens & _SPEED_PREFERENCE_KEYWORDS)

        scored: list[tuple[int, tuple[int, int], str]] = []
        for cap in candidates:
            score = 0
            cap_tags: set[str] = {t.lower() for t in cap.capability_tags}
            cap_preferred_types: set[str] = {t.lower() for t in cap.preferred_task_types}
            cap_scenarios_lower: list[str] = [s.lower() for s in cap.best_scenarios]

            # Signal 1: task_type exact match in preferred_task_types (+8)
            if task_type_lower and task_type_lower in cap_preferred_types:
                score += 8

            # Signal 2: required capability tags present in provider tags (+5 each)
            for tag in required_tags:
                if tag in cap_tags:
                    score += 5

            # Signal 3: cognitive complexity tier covers task level (+3)
            if cap.cognitive_complexity and accepted_tiers & set(cap.cognitive_complexity):
                score += 3

            # Signal 4: use_cases Jaccard match against best_scenarios (+2 each)
            for use_case in use_cases_lower:
                for scenario in cap_scenarios_lower:
                    if _jaccard(use_case, scenario) >= self._jaccard_threshold:
                        score += 2
                        break  # count each use_case at most once

            # Signal 5: complexity feature tokens in capability_tags (+2 each)
            for feat in complexity_feature_tokens:
                if feat in cap_tags:
                    score += 2

            # Signal 6: tool-use capability when tools are needed (+3)
            if needs_tools and "tool_use" in cap_tags:
                score += 3

            # Penalty 1: context window risk (-5)
            if hints.estimated_context_tokens > 0 and cap.context_size > 0:
                if hints.estimated_context_tokens > cap.context_size * 0.7:
                    score -= 5

            # Penalty 2: latency SLO violation (-3)
            if hints.latency_sensitive and cap.latency_tier != "fast":
                score -= 3

            # Penalty 3: cost constraint violation (-3)
            if hints.cost_sensitive and cap.cost_tier == "high":
                score -= 3

            # Penalty 4: soft cost preference (-2)
            if prefer_low_cost and cap.cost_tier == "high":
                score -= 2

            # Penalty 5: soft speed preference (-2)
            if prefer_low_latency and cap.latency_tier != "fast":
                score -= 2

            # Tie-break tuple: (score desc, cost_rank asc, latency_rank asc)
            tiebreak = (
                _COST_RANK.get(cap.cost_tier, 1),
                _LATENCY_RANK.get(cap.latency_tier, 1),
            )
            scored.append((score, tiebreak, cap.name))

        scored.sort(key=lambda x: (-x[0], x[1][0], x[1][1]))
        return [name for _, _, name in scored]


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
        event_bus: EventBus,
        provider_capabilities: list[LLMProviderCapabilities],
        strategy: RoutingStrategy | None = None,
        enable_fallback: bool = False,
    ) -> None:
        if not provider_capabilities:
            raise build_pipeline_error(CONFIG_ERROR, "provider_capabilities cannot be empty")
        self._config = config
        self._logger = logger
        self._tracer = tracer
        self._event_bus = event_bus
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
        self._priority_list: list[str] = []
        self._current_provider: str = ""

    def set_strategy(self, strategy: RoutingStrategy) -> None:
        """Replace the routing strategy at runtime."""
        self._strategy = strategy

    def initialize_routing(self, task: Task) -> None:
        """Build the priority list for *task* and set the initial current provider.

        Stores routing state internally; callers do not receive a return value.
        """
        candidates = list(self._capabilities)
        if not candidates:
            raise build_config_error(CONFIG_ERROR, "no available providers after applying exclusions")

        with self._tracer.start_span(
            "model.initialize_routing", "routing",
            {
                "task_id": task.id if task else None,
                "task_type": task.task_type if task else None,
                "strategy": self._strategy.__class__.__name__,
                "candidate_count": len(candidates),
                "enable_fallback": self._enable_fallback,
            },
        ) as span:
            ordered = self._strategy.select(task, candidates)
            if not ordered:
                raise build_pipeline_error(UNKNOWN_LOGIC_ERROR, "routing strategy returned an empty provider list")

            self._priority_list = ordered if self._enable_fallback else ordered[:1]
            self._current_provider = self._priority_list[0]
            span.add_attributes({"priority_list": self._priority_list, "current_provider": self._current_provider})

        self._logger.info(
            "Model routing initialized",
            zap.any("task_id", task.id if task else None),
            zap.any("strategy", self._strategy.__class__.__name__),
            zap.any("priority_list", self._priority_list),
            zap.any("current_provider", self._current_provider),
        )

    def get_current_provider(self) -> str:
        """Return the currently active provider name."""
        if not self._current_provider:
            raise build_pipeline_error(UNKNOWN_LOGIC_ERROR, "get_current_provider() called before initialize_routing()")
        return self._current_provider

    def advance_provider(self, error: LLMNormalizedError | None) -> str:
        """Record failure for current provider and advance to the next best one.

        Selection order:
          1. Check if a higher-priority provider has recovered (get_best_recovered_provider).
          2. Otherwise fall forward to the next available provider (get_next_available_provider).
        Raises PipelineError(LLM_ALL_PROVIDERS_FAILED) if all providers are exhausted.
        """
        if not self._priority_list:
            raise build_pipeline_error(UNKNOWN_LOGIC_ERROR, "advance_provider() called before initialize_routing()")

        failed = self._current_provider
        self.record_provider_failure(failed, error)

        recovered = self.get_best_recovered_provider(self._priority_list, failed)
        if recovered is not None:
            self._current_provider = recovered
            self._logger.info("Switched to recovered higher-priority provider",
                zap.any("from", failed), zap.any("to", recovered))
            return self._current_provider

        next_provider = self.get_next_available_provider(self._priority_list, failed)
        if next_provider is None:
            raise PipelineError(
                code=LLM_ALL_PROVIDERS_FAILED,
                message=f"All providers in priority list exhausted after failure on '{failed}'",
            )

        self._current_provider = next_provider
        self._logger.info("Advanced to next available provider",
            zap.any("from", failed), zap.any("to", next_provider))
        return self._current_provider

    def confirm_provider_success(self) -> None:
        """Record success for the current provider."""
        if self._current_provider:
            self.record_provider_success(self._current_provider)

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
