from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from config import ConfigReader
from jobs.job_base import JobResult
from tools.agent_reach.health_store import AgentReachHealthStore
from utils.env_util.runtime_env import get_project_root


class AgentReachHealthJob:
    name = "agent_reach_health"

    def __init__(self, config: ConfigReader) -> None:
        self._config = config
        self._project_root = get_project_root()
        self._store = AgentReachHealthStore(self._snapshot_path())

    def run_once(self) -> JobResult:
        timeout = self._config.positive_int("jobs.agent_reach_health.timeout_seconds", 45)
        started = time.time()
        try:
            completed = subprocess.run(
                ["agent-reach", "doctor", "--json"],
                capture_output=True,
                text=True,
                cwd=self._project_root,
                timeout=timeout,
                env=self._subprocess_env(),
            )
        except subprocess.TimeoutExpired:
            message = f"agent-reach doctor timed out after {timeout} seconds"
            self._store.write_failure(message=message, checked_at=started)
            return JobResult(success=False, message=message)
        except Exception as exc:
            message = f"agent-reach doctor failed to start: {exc}"
            self._store.write_failure(message=message, checked_at=started)
            return JobResult(success=False, message=message)

        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or f"agent-reach doctor exited {completed.returncode}").strip()
            self._store.write_failure(message=message, checked_at=started)
            return JobResult(success=False, message=message)

        try:
            health = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            message = f"agent-reach doctor returned invalid JSON: {exc}"
            self._store.write_failure(message=message, checked_at=started)
            return JobResult(success=False, message=message)

        if not isinstance(health, dict):
            message = "agent-reach doctor JSON must be an object"
            self._store.write_failure(message=message, checked_at=started)
            return JobResult(success=False, message=message)

        snapshot = self._store.write_success(
            health=health,
            checked_at=started,
            duration_ms=int((time.time() - started) * 1000),
        )
        ok_count = sum(1 for item in health.values() if isinstance(item, dict) and item.get("status") == "ok")
        return JobResult(
            success=True,
            message=f"agent-reach health refreshed ({ok_count} ok channels)",
            data={"snapshot_path": str(self._store.path), "snapshot": snapshot},
        )

    def _snapshot_path(self) -> Path:
        configured = self._config.get("jobs.agent_reach_health.output_path", "var/state/agent_reach_health.json")
        path = Path(str(configured))
        if not path.is_absolute():
            path = self._project_root / path
        return path

    @staticmethod
    def _subprocess_env() -> dict[str, str]:
        env = dict(os.environ)
        path_parts = [str(Path(sys.prefix) / "bin"), str(Path(sys.executable).resolve().parent)]

        # also probe sibling venv/bin so agent-reach is found regardless of which venv is active
        project_root = get_project_root()
        for venv_name in ("venv", ".venv"):
            candidate = project_root / venv_name / "bin"
            if candidate.is_dir():
                path_parts.append(str(candidate))

        nvm_node = Path.home() / ".nvm" / "versions" / "node"
        if nvm_node.exists():
            path_parts.extend(str(path) for path in sorted(nvm_node.glob("v*/bin"), reverse=True))
        path_parts.append(env.get("PATH", ""))
        env["PATH"] = os.pathsep.join(part for part in path_parts if part)

        try:
            import certifi

            ca_bundle = certifi.where()
            env.setdefault("SSL_CERT_FILE", ca_bundle)
            env.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)
        except Exception:
            pass

        return env
