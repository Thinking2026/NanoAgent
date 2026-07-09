"""Unit tests for AgentReachToolRegistrar.

All tests use mocked health snapshots and a temporary route config.
No network or filesystem side-effects beyond tmp_path.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.agent_reach.registrar import AgentReachToolRegistrar
from tools.agent_reach.native_tools import (
    ExaSearchTool,
    GithubSearchTool,
    JinaReaderTool,
    V2EXTool,
    YoutubeTool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, *, routes_path: str, snapshot_path: str, enabled: bool = True):
    from config.config import ConfigReader
    data = {
        "agent_reach": {
            "routes_path": routes_path,
            "dynamic_tools": {"enabled": enabled},
        },
        "jobs": {
            "agent_reach_health": {
                "output_path": snapshot_path,
            }
        },
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return ConfigReader(p)


def _write_routes(tmp_path: Path, routes: dict) -> Path:
    p = tmp_path / "routes.json"
    p.write_text(json.dumps(routes), encoding="utf-8")
    return p


def _write_snapshot(tmp_path: Path, snapshot: dict) -> Path:
    p = tmp_path / "health.json"
    p.write_text(json.dumps(snapshot), encoding="utf-8")
    return p


def _healthy_snapshot(channels: dict) -> dict:
    return {"success": True, "checked_at": 1234567890.0, "duration_ms": 500, "health": channels, "error": None}


def _make_registry() -> MagicMock:
    registry = MagicMock()
    registry.register = MagicMock()
    return registry


_FULL_ROUTES = {
    "github": {"tool": "agent_reach_github_search", "status_required": "ok", "backend": "gh CLI"},
    "youtube": {"tool": "agent_reach_youtube", "status_required": "ok", "backend": "yt-dlp"},
    "v2ex": {"tool": "agent_reach_v2ex", "status_required": "ok"},
    "web_search": {"tool": "agent_reach_exa_search", "status_required": "ok", "backend": "mcporter"},
    "web": {"tool": "agent_reach_jina_reader", "status_required": "ok", "backend": "Jina Reader"},
}

_ALL_HEALTHY = {
    "github": {"status": "ok", "active_backend": "gh CLI", "backends": ["gh CLI"]},
    "youtube": {"status": "ok", "active_backend": "yt-dlp", "backends": ["yt-dlp"]},
    "v2ex": {"status": "ok", "active_backend": None, "backends": []},
    "web_search": {"status": "ok", "active_backend": "mcporter", "backends": ["mcporter"]},
    "web": {"status": "ok", "active_backend": "Jina Reader", "backends": ["Jina Reader"]},
}


# ---------------------------------------------------------------------------
# Registration logic
# ---------------------------------------------------------------------------

class TestRegistrarRegistration:
    def test_all_healthy_registers_all_tools(self, tmp_path):
        routes_path = _write_routes(tmp_path, _FULL_ROUTES)
        snapshot_path = _write_snapshot(tmp_path, _healthy_snapshot(_ALL_HEALTHY))
        cfg = _make_config(tmp_path, routes_path=str(routes_path), snapshot_path=str(snapshot_path))

        registrar = AgentReachToolRegistrar(cfg)
        registry = _make_registry()
        registered = registrar.register_healthy_tools(registry)

        assert len(registered) == 5
        assert set(registered) == {
            "agent_reach_github_search",
            "agent_reach_youtube",
            "agent_reach_v2ex",
            "agent_reach_exa_search",
            "agent_reach_jina_reader",
        }
        assert registry.register.call_count == 5

    def test_disabled_registers_nothing(self, tmp_path):
        routes_path = _write_routes(tmp_path, _FULL_ROUTES)
        snapshot_path = _write_snapshot(tmp_path, _healthy_snapshot(_ALL_HEALTHY))
        cfg = _make_config(tmp_path, routes_path=str(routes_path), snapshot_path=str(snapshot_path), enabled=False)

        registrar = AgentReachToolRegistrar(cfg)
        registry = _make_registry()
        registered = registrar.register_healthy_tools(registry)

        assert registered == []
        registry.register.assert_not_called()

    def test_missing_snapshot_registers_nothing(self, tmp_path):
        routes_path = _write_routes(tmp_path, _FULL_ROUTES)
        cfg = _make_config(
            tmp_path,
            routes_path=str(routes_path),
            snapshot_path=str(tmp_path / "nonexistent.json"),
        )

        registrar = AgentReachToolRegistrar(cfg)
        registry = _make_registry()
        registered = registrar.register_healthy_tools(registry)

        assert registered == []
        registry.register.assert_not_called()

    def test_failed_snapshot_registers_nothing(self, tmp_path):
        routes_path = _write_routes(tmp_path, _FULL_ROUTES)
        snapshot_path = _write_snapshot(tmp_path, {
            "success": False,
            "error": "agent-reach timed out",
            "health": {},
        })
        cfg = _make_config(tmp_path, routes_path=str(routes_path), snapshot_path=str(snapshot_path))

        registrar = AgentReachToolRegistrar(cfg)
        registry = _make_registry()
        registered = registrar.register_healthy_tools(registry)

        assert registered == []
        registry.register.assert_not_called()

    def test_corrupted_snapshot_registers_nothing(self, tmp_path):
        routes_path = _write_routes(tmp_path, _FULL_ROUTES)
        snapshot_path = tmp_path / "health.json"
        snapshot_path.write_text("{invalid json{{", encoding="utf-8")
        cfg = _make_config(tmp_path, routes_path=str(routes_path), snapshot_path=str(snapshot_path))

        registrar = AgentReachToolRegistrar(cfg)
        registry = _make_registry()
        registered = registrar.register_healthy_tools(registry)

        assert registered == []

    def test_channel_off_not_registered(self, tmp_path):
        routes_path = _write_routes(tmp_path, {"github": _FULL_ROUTES["github"]})
        snapshot_path = _write_snapshot(tmp_path, _healthy_snapshot({
            "github": {"status": "off", "active_backend": None, "backends": []},
        }))
        cfg = _make_config(tmp_path, routes_path=str(routes_path), snapshot_path=str(snapshot_path))

        registrar = AgentReachToolRegistrar(cfg)
        registry = _make_registry()
        registered = registrar.register_healthy_tools(registry)

        assert registered == []
        registry.register.assert_not_called()

    def test_channel_warn_not_registered_when_ok_required(self, tmp_path):
        routes_path = _write_routes(tmp_path, {"github": _FULL_ROUTES["github"]})
        snapshot_path = _write_snapshot(tmp_path, _healthy_snapshot({
            "github": {"status": "warn", "active_backend": "gh CLI", "backends": ["gh CLI"]},
        }))
        cfg = _make_config(tmp_path, routes_path=str(routes_path), snapshot_path=str(snapshot_path))

        registrar = AgentReachToolRegistrar(cfg)
        registry = _make_registry()
        registered = registrar.register_healthy_tools(registry)

        assert registered == []

    def test_wrong_backend_not_registered(self, tmp_path):
        routes_path = _write_routes(tmp_path, {
            "github": {"tool": "agent_reach_github_search", "status_required": "ok", "backend": "gh CLI"},
        })
        snapshot_path = _write_snapshot(tmp_path, _healthy_snapshot({
            "github": {"status": "ok", "active_backend": "some-other-cli", "backends": ["some-other-cli"]},
        }))
        cfg = _make_config(tmp_path, routes_path=str(routes_path), snapshot_path=str(snapshot_path))

        registrar = AgentReachToolRegistrar(cfg)
        registry = _make_registry()
        registered = registrar.register_healthy_tools(registry)

        assert registered == []

    def test_no_backend_requirement_matches_any_backend(self, tmp_path):
        routes_path = _write_routes(tmp_path, {
            "v2ex": {"tool": "agent_reach_v2ex", "status_required": "ok"},
        })
        snapshot_path = _write_snapshot(tmp_path, _healthy_snapshot({
            "v2ex": {"status": "ok", "active_backend": "any-backend", "backends": []},
        }))
        cfg = _make_config(tmp_path, routes_path=str(routes_path), snapshot_path=str(snapshot_path))

        registrar = AgentReachToolRegistrar(cfg)
        registry = _make_registry()
        registered = registrar.register_healthy_tools(registry)

        assert "agent_reach_v2ex" in registered

    def test_partial_health_registers_only_ok_channels(self, tmp_path):
        routes_path = _write_routes(tmp_path, _FULL_ROUTES)
        partial_health = {
            "github": {"status": "ok", "active_backend": "gh CLI", "backends": ["gh CLI"]},
            "youtube": {"status": "off", "active_backend": None, "backends": []},
            "v2ex": {"status": "ok", "active_backend": None, "backends": []},
            "web_search": {"status": "off", "active_backend": None, "backends": []},
            "web": {"status": "ok", "active_backend": "Jina Reader", "backends": ["Jina Reader"]},
        }
        snapshot_path = _write_snapshot(tmp_path, _healthy_snapshot(partial_health))
        cfg = _make_config(tmp_path, routes_path=str(routes_path), snapshot_path=str(snapshot_path))

        registrar = AgentReachToolRegistrar(cfg)
        registry = _make_registry()
        registered = registrar.register_healthy_tools(registry)

        assert set(registered) == {"agent_reach_github_search", "agent_reach_v2ex", "agent_reach_jina_reader"}
        assert registry.register.call_count == 3


# ---------------------------------------------------------------------------
# Route loading
# ---------------------------------------------------------------------------

class TestRegistrarRouteLoading:
    def test_missing_route_file_registers_nothing(self, tmp_path):
        snapshot_path = _write_snapshot(tmp_path, _healthy_snapshot(_ALL_HEALTHY))
        cfg = _make_config(
            tmp_path,
            routes_path=str(tmp_path / "nonexistent_routes.json"),
            snapshot_path=str(snapshot_path),
        )

        registrar = AgentReachToolRegistrar(cfg)
        registry = _make_registry()
        registered = registrar.register_healthy_tools(registry)

        assert registered == []

    def test_invalid_route_json_registers_nothing(self, tmp_path):
        routes_path = tmp_path / "routes.json"
        routes_path.write_text("{not json{", encoding="utf-8")
        snapshot_path = _write_snapshot(tmp_path, _healthy_snapshot(_ALL_HEALTHY))
        cfg = _make_config(tmp_path, routes_path=str(routes_path), snapshot_path=str(snapshot_path))

        registrar = AgentReachToolRegistrar(cfg)
        registry = _make_registry()
        registered = registrar.register_healthy_tools(registry)

        assert registered == []

    def test_unknown_tool_name_skipped(self, tmp_path):
        routes_path = _write_routes(tmp_path, {
            "github": {"tool": "agent_reach_nonexistent_tool", "status_required": "ok"},
        })
        snapshot_path = _write_snapshot(tmp_path, _healthy_snapshot({
            "github": {"status": "ok", "active_backend": "gh CLI", "backends": ["gh CLI"]},
        }))
        cfg = _make_config(tmp_path, routes_path=str(routes_path), snapshot_path=str(snapshot_path))

        registrar = AgentReachToolRegistrar(cfg)
        registry = _make_registry()
        registered = registrar.register_healthy_tools(registry)

        assert registered == []

    def test_empty_routes_registers_nothing(self, tmp_path):
        routes_path = _write_routes(tmp_path, {})
        snapshot_path = _write_snapshot(tmp_path, _healthy_snapshot(_ALL_HEALTHY))
        cfg = _make_config(tmp_path, routes_path=str(routes_path), snapshot_path=str(snapshot_path))

        registrar = AgentReachToolRegistrar(cfg)
        registry = _make_registry()
        registered = registrar.register_healthy_tools(registry)

        assert registered == []


# ---------------------------------------------------------------------------
# Tool factory instances
# ---------------------------------------------------------------------------

class TestRegistrarToolInstances:
    def test_registers_correct_tool_types(self, tmp_path):
        routes_path = _write_routes(tmp_path, _FULL_ROUTES)
        snapshot_path = _write_snapshot(tmp_path, _healthy_snapshot(_ALL_HEALTHY))
        cfg = _make_config(tmp_path, routes_path=str(routes_path), snapshot_path=str(snapshot_path))

        registrar = AgentReachToolRegistrar(cfg)
        registered_tools = []

        class CapturingRegistry:
            def register(self, tool):
                registered_tools.append(tool)

        registrar.register_healthy_tools(CapturingRegistry())

        tool_types = {type(t) for t in registered_tools}
        assert ExaSearchTool in tool_types
        assert JinaReaderTool in tool_types
        assert GithubSearchTool in tool_types
        assert V2EXTool in tool_types
        assert YoutubeTool in tool_types
