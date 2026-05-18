from __future__ import annotations

from unittest.mock import MagicMock
from types import SimpleNamespace

import pytest

from agent.models.reasoning.decision import NextDecision, NextDecisionType
from agent.models.reasoning.impl.react.message_formatter import MessageFormatter
from agent.models.reasoning.impl.react.react_strategy import ReActStrategy
from schemas.types import LLMMessage, UnifiedLLMRequest, LLMResponse, ToolCall


# ---------------------------------------------------------------------------
# MessageFormatter
# ---------------------------------------------------------------------------

def test_formatter_build_request():
    fmt = MessageFormatter()
    msgs = [LLMMessage(role="user", content="hello")]
    tools = [{"name": "calc"}]
    req = fmt.build_request(system_prompt="sys", conversation=msgs, tools=tools)
    assert isinstance(req, UnifiedLLMRequest)
    assert req.system_prompt == "sys"
    assert req.messages is msgs
    assert req.tool_schemas is tools


def test_formatter_format_tool_observation_success():
    fmt = MessageFormatter()
    msg = fmt.format_tool_observation(
        tool_name="calc",
        output='{"result": 42}',
        success=True,
        llm_raw_tool_call_id="tc1",
    )
    assert msg.role == "tool"
    assert msg.content == '{"result": 42}'
    assert msg.metadata["tool_name"] == "calc"
    assert msg.metadata["success"] is True
    assert msg.metadata["llm_raw_tool_call_id"] == "tc1"


def test_formatter_format_tool_observation_failure():
    fmt = MessageFormatter()
    msg = fmt.format_tool_observation(
        tool_name="shell",
        output="error",
        success=False,
        llm_raw_tool_call_id=None,
    )
    assert msg.metadata["success"] is False
    assert msg.metadata["llm_raw_tool_call_id"] is None


def test_formatter_parse_response_passthrough():
    fmt = MessageFormatter()
    resp = LLMResponse(assistant_message=LLMMessage(role="assistant", content="done"))
    assert fmt.parse_response(resp) is resp


# ---------------------------------------------------------------------------
# ReActStrategy — parse_llm_response
# ---------------------------------------------------------------------------

def make_strategy() -> ReActStrategy:
    return ReActStrategy()


def make_response(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    finish_reason: str = "stop",
) -> LLMResponse:
    return LLMResponse(
        assistant_message=LLMMessage(role="assistant", content=content),
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
    )


def test_parse_final_answer():
    strategy = make_strategy()
    resp = make_response(content="The answer is 42.")
    decision = strategy.parse_llm_response(resp)
    assert isinstance(decision, NextDecision)
    assert decision.decision_type == NextDecisionType.FINAL_ANSWER
    assert decision.answer == "The answer is 42."


def test_parse_invoke_tools():
    strategy = make_strategy()
    tc = ToolCall(name="calc", arguments={"expression": "1+1"}, llm_raw_tool_call_id="tc1")
    resp = make_response(tool_calls=[tc])
    decision = strategy.parse_llm_response(resp)
    assert isinstance(decision, NextDecision)
    assert decision.decision_type == NextDecisionType.TOOL_CALL
    assert len(decision.tool_calls) == 1
    assert decision.tool_calls[0].name == "calc"
    assert decision.assistant_message is not None
    assert decision.assistant_message.role == "assistant"


def test_parse_response_truncated():
    strategy = make_strategy()
    resp = make_response(content="partial...", finish_reason="length")
    decision = strategy.parse_llm_response(resp)
    assert isinstance(decision, NextDecision)
    assert decision.decision_type == NextDecisionType.FINAL_ANSWER
    assert decision.answer == "partial..."


def test_parse_multiple_tool_calls():
    strategy = make_strategy()
    tcs = [
        ToolCall(name="calc", arguments={"expression": "1+1"}, llm_raw_tool_call_id="tc1"),
        ToolCall(name="file", arguments={"action": "read", "path": "f.txt"}, llm_raw_tool_call_id="tc2"),
    ]
    resp = make_response(tool_calls=tcs)
    decision = strategy.parse_llm_response(resp)
    assert decision.decision_type == NextDecisionType.TOOL_CALL
    assert len(decision.tool_calls) == 2


# ---------------------------------------------------------------------------
# ReActStrategy — build_llm_request
# ---------------------------------------------------------------------------

def test_build_llm_request():
    strategy = make_strategy()
    req_in = UnifiedLLMRequest(
        system_prompt="base context",
        messages=[LLMMessage(role="user", content="hi")],
        tool_schemas=[{"name": "calc"}],
    )

    req = strategy.build_llm_request(req_in)
    assert isinstance(req, UnifiedLLMRequest)
    assert req.system_prompt.startswith(ReActStrategy.SYSTEM_PROMPT)
    assert len(req.messages) == 1
    assert req.tool_schemas == [{"name": "calc"}]


# ---------------------------------------------------------------------------
# ReActStrategy — format_tool_observation
# ---------------------------------------------------------------------------

def test_format_tool_observation_success():
    strategy = make_strategy()
    tc = ToolCall(name="calc", arguments={}, llm_raw_tool_call_id="tc1")
    from schemas.types import ToolResult
    result = ToolResult(output='{"result": 5}', success=True)
    msg = strategy.format_tool_observation(tc, result)
    assert msg.role == "tool"
    assert msg.metadata["tool_name"] == "calc"
    assert msg.metadata["success"] is True
    assert msg.metadata["llm_raw_tool_call_id"] == "tc1"


def test_format_tool_observation_failure():
    strategy = make_strategy()
    tc = ToolCall(name="shell", arguments={}, llm_raw_tool_call_id="tc2")
    from schemas.types import ToolResult
    result = ToolResult(output="error", success=False)
    msg = strategy.format_tool_observation(tc, result)
    assert msg.metadata["success"] is False


# ---------------------------------------------------------------------------
# NextDecision dataclass
# ---------------------------------------------------------------------------

def test_next_decision_final_answer():
    d = NextDecision(decision_type=NextDecisionType.FINAL_ANSWER, answer="done")
    assert d.decision_type == NextDecisionType.FINAL_ANSWER
    assert d.answer == "done"
    assert d.tool_calls == []


def test_next_decision_tool_call():
    tc = ToolCall(name="calc", arguments={})
    d = NextDecision(decision_type=NextDecisionType.TOOL_CALL, tool_calls=[tc])
    assert d.decision_type == NextDecisionType.TOOL_CALL
    assert len(d.tool_calls) == 1


def test_next_decision_continue():
    d = NextDecision(decision_type=NextDecisionType.CONTINUE, message="thinking...")
    assert d.decision_type == NextDecisionType.CONTINUE
    assert d.message == "thinking..."


def test_next_decision_clarification():
    d = NextDecision(decision_type=NextDecisionType.CLARIFICATION_NEEDED, message="what do you mean?")
    assert d.decision_type == NextDecisionType.CLARIFICATION_NEEDED
