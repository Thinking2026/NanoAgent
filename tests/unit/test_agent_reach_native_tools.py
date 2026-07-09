"""Integration tests for AgentReach native tools.

Tests in TestNativeToolsIntegration make real network calls.
Tests in TestNativeToolsUnit use mocked subprocess to stay fast and offline.
"""
from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from schemas.errors import SHELL_COMMAND_FAILED, SHELL_TIMEOUT, TOOL_ARGUMENT_ERROR
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

def _make_ok_proc(stdout: str) -> MagicMock:
    m = MagicMock()
    m.returncode = 0
    m.stdout = stdout
    m.stderr = ""
    return m


def _make_fail_proc(stderr: str = "error", code: int = 1) -> MagicMock:
    m = MagicMock()
    m.returncode = code
    m.stdout = ""
    m.stderr = stderr
    return m


# ---------------------------------------------------------------------------
# ExaSearchTool — unit
# ---------------------------------------------------------------------------

class TestExaSearchToolUnit:
    def setup_method(self):
        self.tool = ExaSearchTool()

    def test_name(self):
        assert self.tool.name == "agent_reach_exa_search"

    def test_missing_query(self):
        result = self.tool.run({})
        assert not result.success
        assert result.error.code == TOOL_ARGUMENT_ERROR

    def test_empty_query(self):
        result = self.tool.run({"query": "   "})
        assert not result.success
        assert result.error.code == TOOL_ARGUMENT_ERROR

    def test_query_too_long(self):
        result = self.tool.run({"query": "x" * 501})
        assert not result.success
        assert result.error.code == TOOL_ARGUMENT_ERROR

    def test_success_returns_stdout(self):
        fake_output = json.dumps({"results": [{"title": "Test", "url": "https://example.com"}]})
        with patch("subprocess.run", return_value=_make_ok_proc(fake_output)):
            result = self.tool.run({"query": "Python asyncio"})
        assert result.success
        data = json.loads(result.output)["data"]
        assert fake_output in data["stdout"]

    def test_command_failure(self):
        with patch("subprocess.run", return_value=_make_fail_proc("mcporter: command not found")):
            result = self.tool.run({"query": "test"})
        assert not result.success
        assert result.error.code == SHELL_COMMAND_FAILED

    def test_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="mcporter", timeout=30)):
            result = self.tool.run({"query": "test"})
        assert not result.success
        assert result.error.code == SHELL_TIMEOUT

    def test_limit_clamped_to_max(self):
        captured = {}

        def _capture(cmd, **_kwargs):
            captured["cmd"] = cmd
            return _make_ok_proc("ok")

        with patch("subprocess.run", side_effect=_capture):
            self.tool.run({"query": "test", "limit": 999})

        cmd_str = " ".join(captured["cmd"])
        assert "numResults: 10" in cmd_str

    def test_limit_clamped_to_min(self):
        captured = {}

        def _capture(cmd, **_kwargs):
            captured["cmd"] = cmd
            return _make_ok_proc("ok")

        with patch("subprocess.run", side_effect=_capture):
            self.tool.run({"query": "test", "limit": 0})

        cmd_str = " ".join(captured["cmd"])
        assert "numResults: 1" in cmd_str

    def test_output_truncation(self):
        long_output = "x" * 20000
        with patch("subprocess.run", return_value=_make_ok_proc(long_output)):
            result = self.tool.run({"query": "test"})
        assert result.success
        data = json.loads(result.output)["data"]
        assert data["truncated"] is True
        assert len(data["stdout"]) < 20000


# ---------------------------------------------------------------------------
# JinaReaderTool — unit
# ---------------------------------------------------------------------------

class TestJinaReaderToolUnit:
    def setup_method(self):
        self.tool = JinaReaderTool()

    def test_name(self):
        assert self.tool.name == "agent_reach_jina_reader"

    def test_missing_url(self):
        result = self.tool.run({})
        assert not result.success
        assert result.error.code == TOOL_ARGUMENT_ERROR

    def test_non_http_url(self):
        result = self.tool.run({"url": "ftp://example.com"})
        assert not result.success
        assert result.error.code == TOOL_ARGUMENT_ERROR

    def test_url_without_scheme(self):
        result = self.tool.run({"url": "example.com"})
        assert not result.success
        assert result.error.code == TOOL_ARGUMENT_ERROR

    def test_success(self):
        fake_content = "# Example Domain\nThis domain is for illustrative examples."
        with patch("subprocess.run", return_value=_make_ok_proc(fake_content)):
            result = self.tool.run({"url": "https://example.com"})
        assert result.success
        data = json.loads(result.output)["data"]
        assert fake_content in data["stdout"]

    def test_curl_failure(self):
        with patch("subprocess.run", return_value=_make_fail_proc("curl: (6) Could not resolve host")):
            result = self.tool.run({"url": "https://example.com"})
        assert not result.success
        assert result.error.code == SHELL_COMMAND_FAILED

    def test_sigint_kill_reports_signal_name(self):
        with patch("subprocess.run", return_value=_make_fail_proc("", code=-2)):
            result = self.tool.run({"url": "https://example.com"})
        assert not result.success
        assert result.error.code == SHELL_COMMAND_FAILED
        assert "SIGINT" in result.error.message

    def test_jina_url_constructed_correctly(self):
        captured = {}

        def _capture(cmd, **_kwargs):
            captured["cmd"] = cmd
            return _make_ok_proc("content")

        with patch("subprocess.run", side_effect=_capture):
            self.tool.run({"url": "https://example.com/page"})

        url_arg = captured["cmd"][-1]
        assert url_arg == "https://r.jina.ai/https://example.com/page"

    def test_fallback_to_direct_fetch_on_connection_error(self):
        calls = []

        def _side_effect(cmd, **kwargs):  # noqa: ARG001
            calls.append(cmd)
            if "r.jina.ai" in " ".join(cmd):
                return _make_fail_proc("curl: (7) Failed to connect to r.jina.ai port 443: Couldn't connect to server", code=7)
            return _make_ok_proc("<html>direct content</html>")

        with patch("subprocess.run", side_effect=_side_effect):
            result = self.tool.run({"url": "https://example.com"})

        assert result.success
        assert len(calls) == 2
        assert "r.jina.ai" in " ".join(calls[0])
        assert "example.com" in calls[1][-1]

    def test_fallback_to_direct_fetch_on_timeout(self):
        calls = []

        def _side_effect(cmd, **kwargs):  # noqa: ARG001
            calls.append(cmd)
            if "r.jina.ai" in " ".join(cmd):
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=30)
            return _make_ok_proc("<html>direct content</html>")

        with patch("subprocess.run", side_effect=_side_effect):
            result = self.tool.run({"url": "https://example.com"})

        assert result.success
        assert len(calls) == 2

    def test_no_fallback_on_non_connection_error(self):
        calls = []

        def _side_effect(cmd, **kwargs):  # noqa: ARG001
            calls.append(cmd)
            return _make_fail_proc("curl: (22) The requested URL returned error: 404", code=22)

        with patch("subprocess.run", side_effect=_side_effect):
            result = self.tool.run({"url": "https://example.com"})

        assert not result.success
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# GithubSearchTool — unit
# ---------------------------------------------------------------------------

class TestGithubSearchToolUnit:
    def setup_method(self):
        self.tool = GithubSearchTool()

    def test_name(self):
        assert self.tool.name == "agent_reach_github_search"

    def test_missing_query(self):
        result = self.tool.run({})
        assert not result.success
        assert result.error.code == TOOL_ARGUMENT_ERROR

    def test_success_returns_json_repos(self):
        fake_repos = json.dumps([
            {"name": "fastapi", "owner": {"login": "tiangolo"}, "description": "FastAPI framework",
             "url": "https://github.com/tiangolo/fastapi", "stargazersCount": 70000, "updatedAt": "2026-07-01T00:00:00Z"},
        ])
        with patch("subprocess.run", return_value=_make_ok_proc(fake_repos)):
            result = self.tool.run({"query": "fastapi"})
        assert result.success
        data = json.loads(result.output)["data"]
        repos = json.loads(data["stdout"])
        assert len(repos) == 1
        assert repos[0]["name"] == "fastapi"

    def test_gh_command_used(self):
        captured = {}

        def _capture(cmd, **_kwargs):
            captured["cmd"] = cmd
            return _make_ok_proc("[]")

        with patch("subprocess.run", side_effect=_capture):
            self.tool.run({"query": "pytorch"})

        assert captured["cmd"][0] == "gh"
        assert "repos" in captured["cmd"]
        assert "pytorch" in captured["cmd"]

    def test_limit_passed_to_command(self):
        captured = {}

        def _capture(cmd, **_kwargs):
            captured["cmd"] = cmd
            return _make_ok_proc("[]")

        with patch("subprocess.run", side_effect=_capture):
            self.tool.run({"query": "numpy", "limit": 3})

        assert "--limit" in captured["cmd"]
        limit_idx = captured["cmd"].index("--limit")
        assert captured["cmd"][limit_idx + 1] == "3"


# ---------------------------------------------------------------------------
# V2EXTool — unit
# ---------------------------------------------------------------------------

class TestV2EXToolUnit:
    def setup_method(self):
        self.tool = V2EXTool()

    def test_name(self):
        assert self.tool.name == "agent_reach_v2ex"

    def test_no_required_args(self):
        with patch("subprocess.run", return_value=_make_ok_proc("[]")):
            result = self.tool.run({})
        assert result.success

    def test_success_returns_topics(self):
        fake_topics = json.dumps([
            {"id": 1, "title": "Python 3.14 released", "url": "https://v2ex.com/t/1"},
        ])
        with patch("subprocess.run", return_value=_make_ok_proc(fake_topics)):
            result = self.tool.run({})
        assert result.success
        data = json.loads(result.output)["data"]
        topics = json.loads(data["stdout"])
        assert len(topics) == 1
        assert topics[0]["title"] == "Python 3.14 released"

    def test_curl_command_targets_v2ex(self):
        captured = {}

        def _capture(cmd, **_kwargs):
            captured["cmd"] = cmd
            return _make_ok_proc("[]")

        with patch("subprocess.run", side_effect=_capture):
            self.tool.run({})

        cmd_str = " ".join(captured["cmd"])
        assert "v2ex.com/api/topics/hot.json" in cmd_str

    def test_network_error(self):
        with patch("subprocess.run", return_value=_make_fail_proc("curl: (6) Could not resolve host")):
            result = self.tool.run({})
        assert not result.success
        assert result.error.code == SHELL_COMMAND_FAILED


# ---------------------------------------------------------------------------
# YoutubeTool — unit
# ---------------------------------------------------------------------------

class TestYoutubeToolUnit:
    def setup_method(self):
        self.tool = YoutubeTool()

    def test_name(self):
        assert self.tool.name == "agent_reach_youtube"

    def test_invalid_mode(self):
        result = self.tool.run({"mode": "download"})
        assert not result.success
        assert result.error.code == TOOL_ARGUMENT_ERROR

    def test_info_mode_requires_url(self):
        result = self.tool.run({"mode": "info"})
        assert not result.success
        assert result.error.code == TOOL_ARGUMENT_ERROR

    def test_search_mode_requires_query(self):
        result = self.tool.run({"mode": "search"})
        assert not result.success
        assert result.error.code == TOOL_ARGUMENT_ERROR

    def test_info_mode_non_http_url(self):
        result = self.tool.run({"mode": "info", "url": "not-a-url"})
        assert not result.success
        assert result.error.code == TOOL_ARGUMENT_ERROR

    def test_info_mode_success(self):
        fake_meta = json.dumps({"id": "abc123", "title": "Test Video", "duration": 300})
        with patch("subprocess.run", return_value=_make_ok_proc(fake_meta)):
            result = self.tool.run({"mode": "info", "url": "https://www.youtube.com/watch?v=abc123"})
        assert result.success

    def test_search_mode_success(self):
        fake_meta = json.dumps({"id": "xyz", "title": "Python Tutorial"})
        with patch("subprocess.run", return_value=_make_ok_proc(fake_meta)):
            result = self.tool.run({"mode": "search", "query": "Python asyncio tutorial"})
        assert result.success

    def test_search_mode_command_uses_ytsearch(self):
        captured = {}

        def _capture(cmd, **_kwargs):
            captured["cmd"] = cmd
            return _make_ok_proc("{}")

        with patch("subprocess.run", side_effect=_capture):
            self.tool.run({"mode": "search", "query": "numpy tutorial", "limit": 3})

        cmd_str = " ".join(captured["cmd"])
        assert "ytsearch3:numpy tutorial" in cmd_str


# ---------------------------------------------------------------------------
# Integration tests — real network calls
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestNativeToolsIntegration:
    """Makes real network/CLI calls. Skipped unless -m integration is passed."""

    def test_jina_reader_real(self):
        # r.jina.ai may be unreachable from restricted networks
        tool = JinaReaderTool()
        result = tool.run({"url": "https://example.com", "timeout": 60})
        if not result.success:
            pytest.skip(f"Jina Reader unreachable from this network: {json.loads(result.output)['error']}")
        data = json.loads(result.output)["data"]
        assert len(data["stdout"]) > 50

    def test_github_search_real(self):
        tool = GithubSearchTool()
        result = tool.run({"query": "fastapi", "limit": 3, "timeout": 30})
        assert result.success, f"GithubSearch failed: {json.loads(result.output)}"
        data = json.loads(result.output)["data"]
        repos = json.loads(data["stdout"])
        assert len(repos) > 0
        assert "name" in repos[0]
        assert "stargazersCount" in repos[0]

    def test_v2ex_real(self):
        # v2ex.com may be unreachable from restricted networks
        tool = V2EXTool()
        result = tool.run({"timeout": 60})
        if not result.success:
            pytest.skip(f"V2EX unreachable from this network: {json.loads(result.output)['error']}")
        data = json.loads(result.output)["data"]
        if data.get("truncated"):
            # Large JSON response was cut at MAX_OUTPUT_CHARS — structural validity can't be checked
            assert len(data["stdout"]) > 100
            return
        topics = json.loads(data["stdout"])
        assert isinstance(topics, list)
        assert len(topics) > 0

    def test_exa_search_real(self):
        tool = ExaSearchTool()
        result = tool.run({"query": "Python asyncio tutorial", "limit": 3, "timeout": 60})
        assert result.success, f"ExaSearch failed: {json.loads(result.output)}"
        data = json.loads(result.output)["data"]
        assert len(data["stdout"]) > 20

    @pytest.mark.slow
    def test_youtube_search_real(self):
        tool = YoutubeTool()
        result = tool.run({"mode": "search", "query": "Python tutorial", "limit": 2, "timeout": 60})
        assert result.success, f"YouTube failed: {json.loads(result.output)}"
