"""Integration tests for AgentReachHealthJob.

These tests actually invoke `agent-reach doctor --json` via subprocess.
They require the venv to be active and AgentReach installed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jobs.agent_reach_health_job import AgentReachHealthJob
from tools.agent_reach.health_store import AgentReachHealthStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(data: dict, tmp_path: Path):
    from config.config import ConfigReader
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return ConfigReader(p)


def _minimal_config(tmp_path: Path, *, timeout: int = 60, output_path: str | None = None):
    cfg_data: dict = {
        "jobs": {
            "agent_reach_health": {
                "timeout_seconds": timeout,
            }
        }
    }
    if output_path:
        cfg_data["jobs"]["agent_reach_health"]["output_path"] = output_path
    return _make_config(cfg_data, tmp_path)


# ---------------------------------------------------------------------------
# Unit tests — subprocess mocked
# ---------------------------------------------------------------------------

class TestAgentReachHealthJobUnit:
    def test_run_once_success(self, tmp_path):
        snapshot_path = tmp_path / "state" / "health.json"
        cfg = _minimal_config(tmp_path, output_path=str(snapshot_path))
        job = AgentReachHealthJob(cfg)

        fake_health = {
            "github": {"status": "ok", "message": "ok"},
            "rss": {"status": "ok", "message": "ok"},
            "exa_search": {"status": "ok", "message": "ok"},
        }

        mock_completed = MagicMock()
        mock_completed.returncode = 0
        mock_completed.stdout = json.dumps(fake_health)
        mock_completed.stderr = ""

        with patch("subprocess.run", return_value=mock_completed):
            result = job.run_once()

        assert result.success
        assert "3 ok" in result.message
        assert snapshot_path.exists()

        snapshot = json.loads(snapshot_path.read_text())
        assert snapshot["success"] is True
        assert snapshot["health"] == fake_health

    def test_run_once_non_zero_exit(self, tmp_path):
        snapshot_path = tmp_path / "health.json"
        cfg = _minimal_config(tmp_path, output_path=str(snapshot_path))
        job = AgentReachHealthJob(cfg)

        mock_completed = MagicMock()
        mock_completed.returncode = 1
        mock_completed.stdout = ""
        mock_completed.stderr = "command not found"

        with patch("subprocess.run", return_value=mock_completed):
            result = job.run_once()

        assert not result.success
        assert "command not found" in result.message
        snapshot = json.loads(snapshot_path.read_text())
        assert snapshot["success"] is False

    def test_run_once_invalid_json(self, tmp_path):
        snapshot_path = tmp_path / "health.json"
        cfg = _minimal_config(tmp_path, output_path=str(snapshot_path))
        job = AgentReachHealthJob(cfg)

        mock_completed = MagicMock()
        mock_completed.returncode = 0
        mock_completed.stdout = "not json {{"
        mock_completed.stderr = ""

        with patch("subprocess.run", return_value=mock_completed):
            result = job.run_once()

        assert not result.success
        assert "invalid JSON" in result.message

    def test_run_once_json_not_object(self, tmp_path):
        snapshot_path = tmp_path / "health.json"
        cfg = _minimal_config(tmp_path, output_path=str(snapshot_path))
        job = AgentReachHealthJob(cfg)

        mock_completed = MagicMock()
        mock_completed.returncode = 0
        mock_completed.stdout = json.dumps([1, 2, 3])
        mock_completed.stderr = ""

        with patch("subprocess.run", return_value=mock_completed):
            result = job.run_once()

        assert not result.success
        assert "must be an object" in result.message

    def test_run_once_timeout(self, tmp_path):
        snapshot_path = tmp_path / "health.json"
        cfg = _minimal_config(tmp_path, output_path=str(snapshot_path))
        job = AgentReachHealthJob(cfg)

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="agent-reach", timeout=45)):
            result = job.run_once()

        assert not result.success
        assert "timed out" in result.message

    def test_run_once_command_not_found(self, tmp_path):
        snapshot_path = tmp_path / "health.json"
        cfg = _minimal_config(tmp_path, output_path=str(snapshot_path))
        job = AgentReachHealthJob(cfg)

        with patch("subprocess.run", side_effect=FileNotFoundError("No such file")):
            result = job.run_once()

        assert not result.success
        assert "failed to start" in result.message

    def test_snapshot_written_on_success(self, tmp_path):
        snapshot_path = tmp_path / "nested" / "dir" / "health.json"
        cfg = _minimal_config(tmp_path, output_path=str(snapshot_path))
        job = AgentReachHealthJob(cfg)

        mock_completed = MagicMock()
        mock_completed.returncode = 0
        mock_completed.stdout = json.dumps({"github": {"status": "ok"}})
        mock_completed.stderr = ""

        with patch("subprocess.run", return_value=mock_completed):
            result = job.run_once()

        assert result.success
        assert snapshot_path.exists()

    def test_result_data_contains_snapshot_path(self, tmp_path):
        snapshot_path = tmp_path / "health.json"
        cfg = _minimal_config(tmp_path, output_path=str(snapshot_path))
        job = AgentReachHealthJob(cfg)

        mock_completed = MagicMock()
        mock_completed.returncode = 0
        mock_completed.stdout = json.dumps({"exa_search": {"status": "ok"}})
        mock_completed.stderr = ""

        with patch("subprocess.run", return_value=mock_completed):
            result = job.run_once()

        assert "snapshot_path" in result.data
        assert str(snapshot_path) == result.data["snapshot_path"]


# ---------------------------------------------------------------------------
# Integration test — real agent-reach CLI
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAgentReachHealthJobIntegration:
    """Runs the real `agent-reach doctor --json` command.

    Requires AgentReach installed in the venv. Skip if binary not found.
    """

    def test_run_once_real_doctor(self, tmp_path):
        # Verify agent-reach is available before running
        venv_bin = Path(sys.prefix) / "bin" / "agent-reach"
        if not venv_bin.exists():
            pytest.skip("agent-reach CLI not found in venv")

        snapshot_path = tmp_path / "health.json"
        cfg = _minimal_config(tmp_path, timeout=60, output_path=str(snapshot_path))
        job = AgentReachHealthJob(cfg)

        result = job.run_once()

        # Job should succeed (agent-reach doctor exits 0)
        assert result.success, f"run_once failed: {result.message}"

        # Snapshot must exist and contain valid data
        assert snapshot_path.exists()
        snapshot = json.loads(snapshot_path.read_text())
        assert snapshot["success"] is True
        assert isinstance(snapshot["health"], dict)
        assert len(snapshot["health"]) > 0

        # At least one "ok" channel (github/rss/exa are reliable)
        ok_channels = [k for k, v in snapshot["health"].items() if isinstance(v, dict) and v.get("status") == "ok"]
        assert len(ok_channels) > 0, f"Expected at least one ok channel; got: {snapshot['health']}"
