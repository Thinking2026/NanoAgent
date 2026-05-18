from __future__ import annotations

import json
import tempfile

import pytest

from config.config import ConfigReader
from agent.models.context.budget.token_budget_manager import (
    DefaultTokenBudgetManager,
    RoleBasedTokenBudgetManager,
    TokenBudgetManagerFactory,
)
from schemas.errors import ConfigError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(data: dict) -> ConfigReader:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    return ConfigReader(path)


def default_config() -> ConfigReader:
    return make_config({})


def role_based_config(reserve_ratio: float, role_ratios: dict[str, float]) -> ConfigReader:
    return make_config({
        "context": {
            "budget": {
                "role_based": {
                    "reserve_ratio": reserve_ratio,
                    "role_ratios": role_ratios,
                }
            }
        }
    })


# ---------------------------------------------------------------------------
# RoleBasedTokenBudgetManager — defaults
# ---------------------------------------------------------------------------

def test_allocate_with_defaults():
    mgr = RoleBasedTokenBudgetManager(default_config())
    result = mgr.allocate(1000)
    assert result.strategy == "role_based"
    assert result.total_budget == 1000
    assert result.reserve_ratio == pytest.approx(0.20)
    assert result.reserved_tokens == 200
    assert result.available_tokens == 800


def test_allocate_role_budgets_present():
    mgr = RoleBasedTokenBudgetManager(default_config())
    result = mgr.allocate(1000)
    assert "system" in result.role_budgets
    assert "user" in result.role_budgets
    assert "assistant" in result.role_budgets
    assert "tool" in result.role_budgets


def test_allocate_role_budgets_sum_to_available():
    mgr = RoleBasedTokenBudgetManager(default_config())
    result = mgr.allocate(1000)
    total = sum(rb.token_budget for rb in result.role_budgets.values())
    # Due to int truncation, allow small difference
    assert abs(total - result.available_tokens) <= len(result.role_budgets)


def test_allocate_zero_budget():
    mgr = RoleBasedTokenBudgetManager(default_config())
    result = mgr.allocate(0)
    assert result.total_budget == 0
    assert result.reserved_tokens == 0
    assert result.available_tokens == 0


# ---------------------------------------------------------------------------
# RoleBasedTokenBudgetManager — custom config
# ---------------------------------------------------------------------------

def test_custom_reserve_ratio():
    cfg = role_based_config(
        reserve_ratio=0.10,
        role_ratios={"system": 0.25, "user": 0.25, "assistant": 0.25, "tool": 0.25},
    )
    mgr = RoleBasedTokenBudgetManager(cfg)
    result = mgr.allocate(1000)
    assert result.reserve_ratio == pytest.approx(0.10)
    assert result.reserved_tokens == 100
    assert result.available_tokens == 900


def test_custom_role_ratios():
    cfg = role_based_config(
        reserve_ratio=0.20,
        role_ratios={"system": 0.10, "user": 0.40, "assistant": 0.30, "tool": 0.20},
    )
    mgr = RoleBasedTokenBudgetManager(cfg)
    result = mgr.allocate(1000)
    user_budget = result.role_budgets["user"].token_budget
    assert user_budget == int(800 * 0.40)


# ---------------------------------------------------------------------------
# RoleBasedTokenBudgetManager — validation errors
# ---------------------------------------------------------------------------

def test_invalid_reserve_ratio_zero_raises():
    cfg = role_based_config(
        reserve_ratio=0.0,
        role_ratios={"system": 0.25, "user": 0.25, "assistant": 0.25, "tool": 0.25},
    )
    with pytest.raises(ConfigError, match="reserve_ratio"):
        RoleBasedTokenBudgetManager(cfg)


def test_invalid_reserve_ratio_one_raises():
    cfg = role_based_config(
        reserve_ratio=1.0,
        role_ratios={"system": 0.25, "user": 0.25, "assistant": 0.25, "tool": 0.25},
    )
    with pytest.raises(ConfigError, match="reserve_ratio"):
        RoleBasedTokenBudgetManager(cfg)


def test_role_ratios_not_summing_to_one_raises():
    cfg = role_based_config(
        reserve_ratio=0.20,
        role_ratios={"system": 0.10, "user": 0.10, "assistant": 0.10, "tool": 0.10},
    )
    with pytest.raises(ConfigError, match="role_ratios must sum to 1.0"):
        RoleBasedTokenBudgetManager(cfg)


# ---------------------------------------------------------------------------
# TokenBudgetManagerFactory
# ---------------------------------------------------------------------------

def test_factory_creates_role_based_manager():
    mgr = TokenBudgetManagerFactory.create("role_based", default_config())
    assert isinstance(mgr, RoleBasedTokenBudgetManager)


def test_factory_creates_default_manager():
    mgr = TokenBudgetManagerFactory.create("default", default_config())
    assert isinstance(mgr, DefaultTokenBudgetManager)


def test_factory_unknown_strategy_raises():
    with pytest.raises(ConfigError, match="Unsupported strategy"):
        TokenBudgetManagerFactory.create("unknown", default_config())
