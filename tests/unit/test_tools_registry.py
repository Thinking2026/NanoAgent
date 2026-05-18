from __future__ import annotations

from typing import Any

import pytest

from schemas.types import ToolCall, ToolResult
from schemas.errors import PipelineError, TOOL_NOT_FOUND, TOOL_TIMEOUT, TOOL_EXECUTION_ERROR
from tools.tool_base import BaseTool, build_tool_output
from tools.tool_registry import (
    ToolRegistry,
    ToolChainRouter,
    ToolHandlerNode,
    FallbackToolHandler,
)


# ---------------------------------------------------------------------------
# Concrete test tool
# ---------------------------------------------------------------------------

class EchoTool(BaseTool):
    name = "echo"
    description = "Echoes the input"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(output=arguments.get("text", ""), success=True)


class FailTool(BaseTool):
    name = "fail"
    description = "Always fails"
    parameters = {"type": "object", "properties": {}}

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        raise RuntimeError("intentional failure")


class TimeoutTool(BaseTool):
    name = "timeout_tool"
    description = "Raises TimeoutError"
    parameters = {"type": "object", "properties": {}}

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        raise TimeoutError("timed out")


class AgentTimeoutTool(BaseTool):
    name = "agent_timeout_tool"
    description = "Raises AgentError with TIMEOUT code"
    parameters = {"type": "object", "properties": {}}

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        raise PipelineError(code="TOOL_TIMEOUT", message="agent timeout")


# ---------------------------------------------------------------------------
# build_tool_output
# ---------------------------------------------------------------------------

def test_build_tool_output_success():
    import json
    output = build_tool_output(success=True, data={"result": 42})
    parsed = json.loads(output)
    assert parsed["success"] is True
    assert parsed["data"]["result"] == 42
    assert parsed["error"] is None


def test_build_tool_output_failure():
    import json
    err = PipelineError(code="ERR", message="bad")
    output = build_tool_output(success=False, error=err)
    parsed = json.loads(output)
    assert parsed["success"] is False
    assert parsed["error"]["code"] == "ERR"
    assert parsed["error"]["message"] == "bad"


def test_build_tool_output_no_data():
    import json
    output = build_tool_output(success=True)
    parsed = json.loads(output)
    assert parsed["data"] is None


# ---------------------------------------------------------------------------
# BaseTool
# ---------------------------------------------------------------------------

def test_base_tool_can_handle():
    tool = EchoTool()
    assert tool.can_handle("echo") is True
    assert tool.can_handle("other") is False


def test_base_tool_schema():
    tool = EchoTool()
    schema = tool.schema()
    assert schema["name"] == "echo"
    assert schema["description"] == "Echoes the input"
    assert "properties" in schema["parameters"]


def test_base_tool_reset_does_nothing():
    tool = EchoTool()
    tool.reset()  # should not raise


# ---------------------------------------------------------------------------
# FallbackToolHandler
# ---------------------------------------------------------------------------

def test_fallback_handler_returns_not_found():
    handler = FallbackToolHandler()
    tc = ToolCall(name="unknown", arguments={})
    result = handler.handle(tc)
    assert not result.success
    assert result.error.code == TOOL_NOT_FOUND


# ---------------------------------------------------------------------------
# ToolHandlerNode
# ---------------------------------------------------------------------------

def test_tool_handler_node_success():
    node = ToolHandlerNode(EchoTool())
    tc = ToolCall(name="echo", arguments={"text": "hello"}, llm_raw_tool_call_id="tc1")
    result = node.process(tc)
    assert result.success
    assert result.output == "hello"
    assert result.llm_raw_tool_call_id == "tc1"


def test_tool_handler_node_unexpected_exception():
    node = ToolHandlerNode(FailTool())
    tc = ToolCall(name="fail", arguments={})
    result = node.process(tc)
    assert not result.success
    assert result.error.code == TOOL_EXECUTION_ERROR


def test_tool_handler_node_timeout_error():
    node = ToolHandlerNode(TimeoutTool())
    tc = ToolCall(name="timeout_tool", arguments={})
    result = node.process(tc)
    assert not result.success
    assert result.error.code == TOOL_TIMEOUT


def test_tool_handler_node_pipeline_error():
    node = ToolHandlerNode(AgentTimeoutTool())
    tc = ToolCall(name="agent_timeout_tool", arguments={})
    result = node.process(tc)
    assert not result.success


def test_tool_handler_node_delegates_to_next():
    echo = ToolHandlerNode(EchoTool())
    fallback = FallbackToolHandler()
    echo.set_next(fallback)
    tc = ToolCall(name="unknown", arguments={})
    result = echo.handle(tc)
    assert not result.success
    assert result.error.code == TOOL_NOT_FOUND


# ---------------------------------------------------------------------------
# ToolChainRouter
# ---------------------------------------------------------------------------

def test_chain_router_routes_to_correct_tool():
    router = ToolChainRouter([EchoTool()])
    tc = ToolCall(name="echo", arguments={"text": "hi"})
    result = router.route(tc)
    assert result.success
    assert result.output == "hi"


def test_chain_router_unknown_tool_returns_not_found():
    router = ToolChainRouter([EchoTool()])
    tc = ToolCall(name="nonexistent", arguments={})
    result = router.route(tc)
    assert not result.success
    assert result.error.code == TOOL_NOT_FOUND


def test_chain_router_empty_tools():
    router = ToolChainRouter([])
    tc = ToolCall(name="any", arguments={})
    result = router.route(tc)
    assert not result.success


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

def test_registry_execute_success():
    registry = ToolRegistry(tools=[EchoTool()], timeout_retry_max_attempts=1)
    result = registry.execute("echo", {"text": "world"})
    assert result.success
    assert result.output == "world"


def test_registry_execute_unknown_tool():
    registry = ToolRegistry(tools=[], timeout_retry_max_attempts=1)
    result = registry.execute("unknown", {})
    assert not result.success
    assert result.error.code == TOOL_NOT_FOUND


def test_registry_register_and_execute():
    registry = ToolRegistry(timeout_retry_max_attempts=1)
    registry.register(EchoTool())
    result = registry.execute("echo", {"text": "test"})
    assert result.success


def test_registry_get_tool_schemas():
    registry = ToolRegistry(tools=[EchoTool()], timeout_retry_max_attempts=1)
    schemas = registry.get_tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "echo"


def test_registry_reset_all():
    registry = ToolRegistry(tools=[EchoTool()], timeout_retry_max_attempts=1)
    registry.reset_all()  # should not raise


def test_registry_execute_sets_tool_call_id():
    registry = ToolRegistry(tools=[EchoTool()], timeout_retry_max_attempts=1)
    result = registry.execute("echo", {"text": "x"}, llm_raw_tool_call_id="tc_99")
    assert result.llm_raw_tool_call_id == "tc_99"


def test_registry_retries_timeout_error(monkeypatch):
    tool = EchoTool()
    calls = {"count": 0}

    def flaky_run(arguments: dict[str, Any]) -> ToolResult:
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError("temporary")
        return ToolResult(output="ok", success=True)

    monkeypatch.setattr(tool, "run", flaky_run)
    monkeypatch.setattr("tools.tool_registry.time.sleep", lambda _: None)
    registry = ToolRegistry(
        tools=[tool],
        timeout_retry_max_attempts=2,
        timeout_retry_delays=(0.001,),
    )
    result = registry.execute("echo", {})
    assert result.success
    assert result.output == "ok"
    assert calls["count"] == 2


def test_registry_validate_arguments_reports_missing_required():
    registry = ToolRegistry(tools=[EchoTool()], timeout_retry_max_attempts=1)
    missing = registry.validate_arguments(ToolCall(name="echo", arguments={}))
    assert missing == ["text"]


def test_registry_get_tool_schemas_for_filters_by_name():
    registry = ToolRegistry(tools=[EchoTool(), FailTool()], timeout_retry_max_attempts=1)
    schemas = registry.get_tool_schemas_for(["fail"])
    assert [schema["name"] for schema in schemas] == ["fail"]


# ---------------------------------------------------------------------------
# ToolRegistry._normalize_retry_delays
# ---------------------------------------------------------------------------

def test_normalize_retry_delays_pads():
    delays = ToolRegistry._normalize_retry_delays((1.0,), max_attempts=4)
    assert len(delays) == 3
    assert delays[0] == pytest.approx(1.0)
    assert delays[1] == pytest.approx(2.0)
    assert delays[2] == pytest.approx(4.0)


def test_normalize_retry_delays_truncates():
    delays = ToolRegistry._normalize_retry_delays((1.0, 2.0, 4.0, 8.0), max_attempts=3)
    assert len(delays) == 2


def test_normalize_retry_delays_max_attempts_one():
    delays = ToolRegistry._normalize_retry_delays((1.0,), max_attempts=1)
    assert delays == ()


def test_normalize_retry_delays_empty_input_uses_default():
    delays = ToolRegistry._normalize_retry_delays((), max_attempts=3)
    assert len(delays) == 2
    assert delays[0] == pytest.approx(1.0)
