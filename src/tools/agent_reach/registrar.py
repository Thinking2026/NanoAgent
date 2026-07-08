from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import ConfigReader
from tools.agent_reach.health_store import AgentReachHealthStore
from tools.agent_reach.native_tools import (
    ExaSearchTool,
    GithubSearchTool,
    JinaReaderTool,
    V2EXTool,
    YoutubeTool,
)
from tools.tool_registry import ToolRegistry
from utils.env_util.runtime_env import get_project_root
from utils.log.log import Logger, zap


_TOOL_FACTORIES = {
    "agent_reach_exa_search": ExaSearchTool,
    "agent_reach_jina_reader": JinaReaderTool,
    "agent_reach_youtube": YoutubeTool,
    "agent_reach_github_search": GithubSearchTool,
    "agent_reach_v2ex": V2EXTool,
}


class AgentReachToolRegistrar:
    def __init__(self, config: ConfigReader, logger: Logger | None = None) -> None:
        self._config = config
        self._logger = logger or Logger.get_instance()
        self._project_root = get_project_root()
        self._health_store = AgentReachHealthStore(self._health_snapshot_path())
        self._routes = self._load_routes()

    def register_healthy_tools(self, registry: ToolRegistry) -> list[str]:
        if not self._enabled():
            return []

        snapshot = self._health_store.read()
        if not snapshot or not snapshot.get("success"):
            self._logger.warning(
                "Agent Reach health snapshot unavailable; skipping dynamic tool registration",
                zap.any("path", str(self._health_store.path)),
                zap.any("error", None if not snapshot else snapshot.get("error")),
            )
            return []

        health = snapshot.get("health")
        if not isinstance(health, dict):
            return []

        registered: list[str] = []
        for channel, route in self._routes.items():
            if not isinstance(route, dict):
                continue
            tool_name = str(route.get("tool", ""))
            factory = _TOOL_FACTORIES.get(tool_name)
            if factory is None:
                continue
            if not self._route_is_healthy(channel, route, health):
                continue
            registry.register(factory())
            registered.append(tool_name)

        self._logger.info(
            "Agent Reach healthy tools registered",
            zap.any("count", len(registered)),
            zap.any("tools", registered),
        )
        return registered

    def _route_is_healthy(self, channel: str, route: dict[str, Any], health: dict[str, Any]) -> bool:
        status_required = str(route.get("status_required", "ok"))
        backend_required = route.get("backend")
        channel_health = health.get(channel)
        if not isinstance(channel_health, dict):
            return False
        if channel_health.get("status") != status_required:
            return False
        if backend_required:
            active_backend = str(channel_health.get("active_backend") or "")
            backends = channel_health.get("backends") or []
            if active_backend != backend_required and backend_required not in backends:
                return False
        return True

    def _load_routes(self) -> dict[str, Any]:
        path = self._route_config_path()
        if not path.exists():
            self._logger.warning("Agent Reach route config missing", zap.any("path", str(path)))
            return {}
        try:
            routes = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self._logger.error("Agent Reach route config is invalid", zap.any("path", str(path)), zap.any("error", exc))
            return {}
        return routes if isinstance(routes, dict) else {}

    def _route_config_path(self) -> Path:
        configured = self._config.get("agent_reach.routes_path", "config/agent_reach_routes.json")
        path = Path(str(configured))
        if not path.is_absolute():
            path = self._project_root / path
        return path

    def _health_snapshot_path(self) -> Path:
        configured = self._config.get("jobs.agent_reach_health.output_path", "var/state/agent_reach_health.json")
        path = Path(str(configured))
        if not path.is_absolute():
            path = self._project_root / path
        return path

    def _enabled(self) -> bool:
        return bool(self._config.get("agent_reach.dynamic_tools.enabled", True))
