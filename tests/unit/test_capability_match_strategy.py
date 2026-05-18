from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent.models.model_routing.provider_router import CapabilityMatchStrategy
from schemas.task import (
    LLMProviderCapabilities,
    RoutingHints,
    Task,
    TaskComplexity,
    TaskStatus,
)
from schemas.ids import TaskId, UserId


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task_id() -> TaskId:
    return TaskId("test-task-1")


def _user_id() -> UserId:
    return UserId("test-user-1")


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make_task(
    task_type: str = "",
    complexity_level: int = 2,
    required_tools: list[str] | None = None,
    related_user_preference: str = "",
    routing_hints: RoutingHints | None = None,
) -> Task:
    return Task(
        id=_task_id(),
        user_id=_user_id(),
        description="test task",
        created_at=_now(),
        status=TaskStatus.CREATED,
        task_type=task_type,
        complexity=TaskComplexity(level=complexity_level),
        required_tools=required_tools or [],
        related_user_preference=related_user_preference,
        routing_hints=routing_hints or RoutingHints(),
    )


def _make_cap(
    name: str,
    capability_tags: list[str] | None = None,
    preferred_task_types: list[str] | None = None,
    cognitive_complexity: list[str] | None = None,
    best_scenarios: list[str] | None = None,
    top_strengths: list[str] | None = None,
    cost_tier: str = "medium",
    latency_tier: str = "medium",
    context_size: int = 200_000,
) -> LLMProviderCapabilities:
    return LLMProviderCapabilities(
        name=name,
        cognitive_complexity=cognitive_complexity or ["L2", "L3"],
        best_scenarios=best_scenarios or [],
        top_strengths=top_strengths or [],
        capability_tags=capability_tags or [],
        preferred_task_types=preferred_task_types or [],
        cost_tier=cost_tier,
        latency_tier=latency_tier,
        context_size=context_size,
    )


STRATEGY = CapabilityMatchStrategy()


# ---------------------------------------------------------------------------
# Signal 1: preferred_task_types exact match
# ---------------------------------------------------------------------------

class TestPreferredTaskTypeMatch:
    def test_exact_match_ranks_first(self):
        code_provider = _make_cap("code_expert", preferred_task_types=["code_generation"])
        generic = _make_cap("generic")
        task = _make_task(task_type="code_generation")
        result = STRATEGY.select(task, [generic, code_provider])
        assert result[0] == "code_expert"

    def test_no_match_does_not_boost(self):
        provider_a = _make_cap("a", preferred_task_types=["math"])
        provider_b = _make_cap("b", preferred_task_types=["code_generation"])
        task = _make_task(task_type="search")
        result = STRATEGY.select(task, [provider_a, provider_b])
        # neither matches; tie-break by cost/latency (both medium) → stable order
        assert set(result) == {"a", "b"}


# ---------------------------------------------------------------------------
# Signal 2: required_capability_tags
# ---------------------------------------------------------------------------

class TestRequiredCapabilityTags:
    def test_tag_match_boosts_score(self):
        code_cap = _make_cap("code_provider", capability_tags=["code", "reasoning"])
        no_code = _make_cap("no_code_provider", capability_tags=["writing"])
        hints = RoutingHints(required_capability_tags=["code"])
        task = _make_task(routing_hints=hints)
        result = STRATEGY.select(task, [no_code, code_cap])
        assert result[0] == "code_provider"

    def test_multiple_tags_accumulate(self):
        full_match = _make_cap("full", capability_tags=["code", "tool_use", "reasoning"])
        partial = _make_cap("partial", capability_tags=["code"])
        hints = RoutingHints(required_capability_tags=["code", "tool_use", "reasoning"])
        task = _make_task(routing_hints=hints)
        result = STRATEGY.select(task, [partial, full_match])
        assert result[0] == "full"


# ---------------------------------------------------------------------------
# Signal 3: cognitive complexity tier
# ---------------------------------------------------------------------------

class TestCognitiveComplexityTier:
    def test_l4_provider_preferred_for_l4_task(self):
        l4_provider = _make_cap("deep", cognitive_complexity=["L3", "L4"])
        l2_only = _make_cap("light", cognitive_complexity=["L1", "L2"])
        task = _make_task(complexity_level=4)
        result = STRATEGY.select(task, [l2_only, l4_provider])
        assert result[0] == "deep"

    def test_l2_provider_acceptable_for_l1_task(self):
        l2_provider = _make_cap("mid", cognitive_complexity=["L2", "L3"])
        l4_provider = _make_cap("heavy", cognitive_complexity=["L4"])
        task = _make_task(complexity_level=1)
        result = STRATEGY.select(task, [l4_provider, l2_provider])
        # both cover L1 (accepted_tiers = L1,L2,L3,L4); tie-break by cost/latency
        assert set(result) == {"mid", "heavy"}


# ---------------------------------------------------------------------------
# Signal 4: use_cases Jaccard match
# ---------------------------------------------------------------------------

class TestUseCasesJaccardMatch:
    def test_jaccard_match_boosts_score(self):
        # L3 use_cases include "代码审查", "数据分析", "报告生成"
        # provider best_scenarios has "data analysis" — Jaccard("数据分析", "data analysis") = 0
        # but "code review" vs "代码审查" also 0 for Chinese
        # Use English use_cases via a custom complexity
        from schemas.task import TaskComplexity
        task = Task(
            id=_task_id(), user_id=_user_id(), description="test",
            created_at=_now(), status=TaskStatus.CREATED,
            task_type="data_analysis",
            complexity=TaskComplexity(level=3, use_cases=["data analysis", "report generation"]),
            routing_hints=RoutingHints(),
        )
        data_provider = _make_cap("data_expert", best_scenarios=["data analysis", "visualization"])
        generic = _make_cap("generic", best_scenarios=["general purpose"])
        result = STRATEGY.select(task, [generic, data_provider])
        assert result[0] == "data_expert"


# ---------------------------------------------------------------------------
# Signal 6: tool_use capability
# ---------------------------------------------------------------------------

class TestToolUseCapability:
    def test_tool_use_tag_boosts_when_tools_needed(self):
        tool_provider = _make_cap("tool_expert", capability_tags=["tool_use", "code"])
        no_tool = _make_cap("no_tool", capability_tags=["code"])
        task = _make_task(required_tools=["search_tool", "calc_tool"])
        result = STRATEGY.select(task, [no_tool, tool_provider])
        assert result[0] == "tool_expert"

    def test_no_boost_when_no_tools_needed(self):
        tool_provider = _make_cap("tool_expert", capability_tags=["tool_use"])
        no_tool = _make_cap("no_tool", capability_tags=[])
        task = _make_task(required_tools=[])
        # no tool signal; both score 0 on this dimension
        result = STRATEGY.select(task, [no_tool, tool_provider])
        assert set(result) == {"tool_expert", "no_tool"}


# ---------------------------------------------------------------------------
# Penalty 1: context window risk
# ---------------------------------------------------------------------------

class TestContextWindowPenalty:
    def test_small_context_provider_penalized_for_large_task(self):
        small_ctx = _make_cap("small", context_size=128_000)
        large_ctx = _make_cap("large", context_size=1_000_000)
        # 200k tokens > 128k * 0.7 = 89.6k → penalty for small_ctx
        # 200k tokens < 1M * 0.7 = 700k  → no penalty for large_ctx
        hints = RoutingHints(estimated_context_tokens=200_000)
        task = _make_task(routing_hints=hints)
        result = STRATEGY.select(task, [small_ctx, large_ctx])
        assert result[0] == "large"

    def test_no_penalty_when_context_fits(self):
        small_ctx = _make_cap("small", context_size=128_000)
        large_ctx = _make_cap("large", context_size=1_000_000)
        hints = RoutingHints(estimated_context_tokens=50_000)
        task = _make_task(routing_hints=hints)
        result = STRATEGY.select(task, [small_ctx, large_ctx])
        # no penalty; tie-break by cost/latency (both medium)
        assert set(result) == {"small", "large"}


# ---------------------------------------------------------------------------
# Penalty 2 & 3: latency/cost SLO from routing_hints
# ---------------------------------------------------------------------------

class TestSLOPenalties:
    def test_latency_sensitive_penalizes_slow_providers(self):
        fast = _make_cap("fast_provider", latency_tier="fast")
        slow = _make_cap("slow_provider", latency_tier="high")
        hints = RoutingHints(latency_sensitive=True)
        task = _make_task(routing_hints=hints)
        result = STRATEGY.select(task, [slow, fast])
        assert result[0] == "fast_provider"

    def test_cost_sensitive_penalizes_expensive_providers(self):
        cheap = _make_cap("cheap_provider", cost_tier="low")
        expensive = _make_cap("expensive_provider", cost_tier="high")
        hints = RoutingHints(cost_sensitive=True)
        task = _make_task(routing_hints=hints)
        result = STRATEGY.select(task, [expensive, cheap])
        assert result[0] == "cheap_provider"


# ---------------------------------------------------------------------------
# Penalty 4 & 5: soft preferences from related_user_preference
# ---------------------------------------------------------------------------

class TestSoftPreferencePenalties:
    def test_cost_preference_penalizes_high_cost(self):
        cheap = _make_cap("cheap", cost_tier="low")
        expensive = _make_cap("expensive", cost_tier="high")
        task = _make_task(related_user_preference="please use a cheap and affordable option")
        result = STRATEGY.select(task, [expensive, cheap])
        assert result[0] == "cheap"

    def test_speed_preference_penalizes_slow(self):
        fast = _make_cap("fast", latency_tier="fast")
        slow = _make_cap("slow", latency_tier="high")
        task = _make_task(related_user_preference="I need a fast and quick response")
        result = STRATEGY.select(task, [slow, fast])
        assert result[0] == "fast"


# ---------------------------------------------------------------------------
# Tie-breaking
# ---------------------------------------------------------------------------

class TestTieBreaking:
    def test_tie_broken_by_cost_tier(self):
        low_cost = _make_cap("low_cost", cost_tier="low", latency_tier="medium")
        high_cost = _make_cap("high_cost", cost_tier="high", latency_tier="medium")
        task = _make_task()  # no signals → both score 0
        result = STRATEGY.select(task, [high_cost, low_cost])
        assert result[0] == "low_cost"

    def test_tie_broken_by_latency_when_cost_equal(self):
        fast = _make_cap("fast", cost_tier="medium", latency_tier="fast")
        slow = _make_cap("slow", cost_tier="medium", latency_tier="high")
        task = _make_task()
        result = STRATEGY.select(task, [slow, fast])
        assert result[0] == "fast"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_candidate_always_returned(self):
        cap = _make_cap("only_one")
        task = _make_task()
        result = STRATEGY.select(task, [cap])
        assert result == ["only_one"]

    def test_empty_candidates_raises(self):
        task = _make_task()
        with pytest.raises(Exception):
            STRATEGY.select(task, [])

    def test_none_task_returns_all_candidates(self):
        caps = [_make_cap("a"), _make_cap("b"), _make_cap("c")]
        result = STRATEGY.select(None, caps)
        assert set(result) == {"a", "b", "c"}

    def test_empty_capability_tags_no_crash(self):
        cap = _make_cap("bare", capability_tags=[], preferred_task_types=[])
        hints = RoutingHints(required_capability_tags=["code", "math"])
        task = _make_task(task_type="code_generation", routing_hints=hints)
        result = STRATEGY.select(task, [cap])
        assert result == ["bare"]
