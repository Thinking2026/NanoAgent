from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from agent.models.reasoning.impl.react.message_formatter import MessageFormatter
from agent.models.context.context_manager import (
    ContextMessage,
    SummaryMetadata,
    ToolResultMetadata,
    ToolUseMetadata,
)
from agent.models.context.estimator.token_estimator import ClaudeTokenEstimator
from agent.models.context.truncation.token_truncation import (
    DefaultContextTruncator,
    TruncationConfig,
    ToolCallMessageUnit,
    _parse_tool_call_message_units,
    _unit_to_messages,
)
from schemas.types import BudgetResult, LLMMessage, RoleBudget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_unit(tool_name: str, arg_val: str, result: str, success: bool = True) -> ToolCallMessageUnit:
    tc_id = f"tc_{tool_name}"
    assistant = ContextMessage(
        id=str(uuid4()),
        role="assistant",
        content="",
        tool_use=ToolUseMetadata(
            tool_call_id=tc_id,
            tool_name=tool_name,
            tool_arguments={"q": arg_val},
        ),
    )
    tool = ContextMessage(
        id=str(uuid4()),
        role="tool",
        content=result,
        tool_result=ToolResultMetadata(
            tool_call_id=tc_id,
            tool_name=tool_name,
            success=success,
        ),
    )
    return ToolCallMessageUnit(assistant_msg=assistant, tool_msgs=[tool])


def units_to_messages(units: list[ToolCallMessageUnit]) -> list[ContextMessage]:
    return [m for u in units for m in _unit_to_messages(u)]


def make_truncator(
    cfg: TruncationConfig | None = None,
    assistant_budget: int = 500,
    tool_budget: int = 500,
) -> DefaultContextTruncator:
    logger = MagicMock()
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: default
    tracer = MagicMock()
    llm_gateway = MagicMock()
    estimator = MagicMock()
    t = DefaultContextTruncator(
        logger=logger,
        config=config,
        tracer=tracer,
        llm_gateway=llm_gateway,
        estimator=estimator,
    )
    if cfg is not None:
        t._trunc_cfg = cfg
    t._assistant_budget = assistant_budget
    t._tool_budget = tool_budget
    return t


def make_fits(truncator: DefaultContextTruncator, estimator: ClaudeTokenEstimator):
    budget = BudgetResult(
        strategy="react",
        total_budget=2000,
        reserve_ratio=0.2,
        reserved_tokens=400,
        available_tokens=1600,
        role_budgets={
            "assistant": RoleBudget("assistant", 0.35, truncator._assistant_budget),
            "tool": RoleBudget("tool", 0.40, truncator._tool_budget),
        },
    )
    return truncator._make_fits_fn(budget, estimator)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = TruncationConfig()
    assert cfg.keep_first_units == 1
    assert cfg.keep_last_units == 3
    assert cfg.summary_ratio == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# _parse_reasoning_units
# ---------------------------------------------------------------------------

def test_parse_reasoning_units_groups_correctly():
    u1 = make_unit("search", "query1", "result1")
    u2 = make_unit("read", "file.txt", "content")
    msgs = units_to_messages([u1, u2])
    units = _parse_tool_call_message_units(msgs)
    assert len(units) == 2
    assert units[0].assistant_msg is u1.assistant_msg
    assert units[1].assistant_msg is u2.assistant_msg


def test_parse_reasoning_units_skips_non_tool_assistant():
    plain_assistant = ContextMessage(id=str(uuid4()), role="assistant", content="plain reply")
    u1 = make_unit("search", "q", "r")
    msgs = [plain_assistant] + units_to_messages([u1])
    units = _parse_tool_call_message_units(msgs)
    assert len(units) == 1


# ---------------------------------------------------------------------------
# _get_middle_units
# ---------------------------------------------------------------------------

def test_get_middle_units_standard():
    cfg = TruncationConfig(keep_first_units=1, keep_last_units=3)
    t = make_truncator(cfg)
    units = [make_unit(f"t{i}", "a", "r") for i in range(6)]
    head, middle, tail = t._get_middle_units(units)
    assert len(head) == 1
    assert len(middle) == 2   # units[1:3]
    assert len(tail) == 3


def test_get_middle_units_empty_when_too_few():
    cfg = TruncationConfig(keep_first_units=1, keep_last_units=3)
    t = make_truncator(cfg)
    units = [make_unit(f"t{i}", "a", "r") for i in range(4)]
    _, middle, _ = t._get_middle_units(units)
    assert middle == []


# ---------------------------------------------------------------------------
# Strategy B
# ---------------------------------------------------------------------------

def test_strategy_b_removes_only_middle_failed():
    cfg = TruncationConfig(keep_first_units=1, keep_last_units=3)
    t = make_truncator(cfg)
    # 5 units: index 0 (head), 1 (middle, failed), 2-4 (tail, protected)
    units = [
        make_unit("t0", "a", "r", success=True),
        make_unit("t1", "a", "r", success=False),   # middle, should be removed
        make_unit("t2", "a", "r", success=False),   # tail, protected
        make_unit("t3", "a", "r", success=True),
        make_unit("t4", "a", "r", success=True),
    ]
    msgs = units_to_messages(units)
    result = t._strategy_b_remove_failed(msgs)
    # unit[1] removed (2 msgs), unit[2] kept (tail)
    assert len(result) == len(msgs) - 2
    removed_ids = {id(m) for m in _unit_to_messages(units[1])}
    for m in result:
        assert id(m) not in removed_ids


def test_strategy_b_empty_middle_returns_unchanged():
    cfg = TruncationConfig(keep_first_units=1, keep_last_units=3)
    t = make_truncator(cfg)
    units = [make_unit(f"t{i}", "a", "r", success=False) for i in range(4)]
    msgs = units_to_messages(units)
    result = t._strategy_b_remove_failed(msgs)
    assert result is msgs  # unchanged


def test_strategy_b_recognizes_failed_tool_message_from_formatter():
    cfg = TruncationConfig(keep_first_units=1, keep_last_units=1)
    t = make_truncator(cfg)
    tc_id = "tc_mid"

    head = make_unit("head", "a", "ok", success=True)
    assistant = ContextMessage(
        id=str(uuid4()),
        role="assistant",
        content="",
        tool_use=ToolUseMetadata(
            tool_call_id=tc_id,
            tool_name="mid",
            tool_arguments={"q": "a"},
        ),
    )
    failed_tool_msg = ContextMessage(
        id=str(uuid4()),
        role="tool",
        content="failed",
        tool_result=ToolResultMetadata(
            tool_call_id=tc_id,
            tool_name="mid",
            success=False,
        ),
    )
    middle = ToolCallMessageUnit(assistant_msg=assistant, tool_msgs=[failed_tool_msg])
    tail = make_unit("tail", "a", "ok", success=True)

    msgs = units_to_messages([head, middle, tail])
    result = t._strategy_b_remove_failed(msgs)

    removed_ids = {id(m) for m in _unit_to_messages(middle)}
    assert len(result) == len(msgs) - 2
    for m in result:
        assert id(m) not in removed_ids


# ---------------------------------------------------------------------------
# Strategy C
# ---------------------------------------------------------------------------

def test_strategy_c_trims_only_middle_args():
    long_arg = "x" * 400
    cfg = TruncationConfig(keep_first_units=1, keep_last_units=1, tool_arg_max_chars=10)
    t = make_truncator(cfg)
    units = [
        make_unit("head", long_arg, "r"),    # head — protected
        make_unit("mid", long_arg, "r"),     # middle — should be trimmed
        make_unit("tail", long_arg, "r"),    # tail — protected
    ]
    msgs = units_to_messages(units)
    result = t._strategy_c_trim_args(msgs)

    # Find the assistant messages by position
    result_units = _parse_tool_call_message_units(result)
    head_args = result_units[0].assistant_msg.tool_use.tool_arguments["q"]
    mid_args = result_units[1].assistant_msg.tool_use.tool_arguments["q"]
    tail_args = result_units[2].assistant_msg.tool_use.tool_arguments["q"]

    assert head_args == long_arg          # untouched
    assert "(trimmed because too long)" in mid_args
    assert tail_args == long_arg          # untouched


# ---------------------------------------------------------------------------
# Strategy D
# ---------------------------------------------------------------------------

def test_strategy_d_trims_only_middle_results():
    long_result = "y" * 600
    cfg = TruncationConfig(keep_first_units=1, keep_last_units=1, tool_result_max_chars=10)
    t = make_truncator(cfg)
    units = [
        make_unit("head", "a", long_result),
        make_unit("mid", "a", long_result),
        make_unit("tail", "a", long_result),
    ]
    msgs = units_to_messages(units)
    result = t._strategy_d_trim_results(msgs)

    result_units = _parse_tool_call_message_units(result)
    head_content = result_units[0].tool_msgs[0].content
    mid_content = result_units[1].tool_msgs[0].content
    tail_content = result_units[2].tool_msgs[0].content

    assert head_content == long_result
    assert "(trimmed because too long)" in mid_content
    assert tail_content == long_result


# ---------------------------------------------------------------------------
# Strategy E
# ---------------------------------------------------------------------------

def test_strategy_e_full_binary_search_range():
    # Build 12 units; with keep_first=1, keep_last=3 → 8 middle units
    # Use a very tight budget so old 10% cap (max 1 drop) would fail
    # but dropping more middle units succeeds
    estimator = ClaudeTokenEstimator()
    cfg = TruncationConfig(keep_first_units=1, keep_last_units=3)
    # Each unit has ~100 chars of tool result → ~29 tokens each (ClaudeTokenEstimator: chars/3.5)
    # head+tail (4 units) = ~116 tool tokens; set budget=200 so dropping 6+ middle units fits
    t = make_truncator(cfg, assistant_budget=2000, tool_budget=200)
    units = [make_unit(f"t{i}", "a", "x" * 100) for i in range(12)]
    msgs = units_to_messages(units)
    fits = make_fits(t, estimator)
    result = t._strategy_e_binary_drop(msgs, fits)
    assert result is not None
    assert fits(result)


def test_strategy_e_returns_none_when_no_solution():
    estimator = ClaudeTokenEstimator()
    cfg = TruncationConfig(keep_first_units=1, keep_last_units=3)
    # Budget so tight that even dropping all middle units won't help
    # (head+tail alone exceed budget)
    t = make_truncator(cfg, assistant_budget=1, tool_budget=1)
    units = [make_unit(f"t{i}", "a", "x" * 100) for i in range(6)]
    msgs = units_to_messages(units)
    fits = make_fits(t, estimator)
    result = t._strategy_e_binary_drop(msgs, fits)
    assert result is None


def test_strategy_e_empty_middle_returns_none():
    estimator = ClaudeTokenEstimator()
    cfg = TruncationConfig(keep_first_units=1, keep_last_units=3)
    t = make_truncator(cfg)
    units = [make_unit(f"t{i}", "a", "r") for i in range(4)]
    msgs = units_to_messages(units)
    fits = make_fits(t, estimator)
    result = t._strategy_e_binary_drop(msgs, fits)
    assert result is None


# ---------------------------------------------------------------------------
# Strategy F
# ---------------------------------------------------------------------------

def _setup_mock_llm(truncator: DefaultContextTruncator, summary_text: str = "summary") -> None:
    mock_response = MagicMock()
    mock_response.assistant_message = LLMMessage(role="assistant", content=summary_text)
    truncator._llm_gateway.generate.return_value = mock_response


def test_strategy_f_uses_summary_ratio():
    cfg = TruncationConfig(keep_first_units=1, keep_last_units=3, summary_ratio=0.5)
    t = make_truncator(cfg)
    _setup_mock_llm(t, "summary text")
    # 6 units: 1 head, 2 middle, 3 tail → summary_ratio=0.5 → summarize 1 of 2 middle units
    units = [make_unit(f"t{i}", "a", "r") for i in range(6)]
    msgs = units_to_messages(units)
    result = t._strategy_f_summarize(msgs)
    assert result is not None
    # The summarized unit (1 assistant + 1 tool = 2 msgs) replaced by 1 summary msg
    assert len(result) == len(msgs) - 1
    summary_msgs = [m for m in result if m.summary is not None]
    assert len(summary_msgs) == 1


def test_strategy_f_minimum_one_unit():
    cfg = TruncationConfig(keep_first_units=1, keep_last_units=3, summary_ratio=0.01)
    t = make_truncator(cfg)
    _setup_mock_llm(t)
    # 6 units → 2 middle; ratio=0.01 → int(2*0.01)=0 → max(1,0)=1 unit summarized
    units = [make_unit(f"t{i}", "a", "r") for i in range(6)]
    msgs = units_to_messages(units)
    result = t._strategy_f_summarize(msgs)
    assert result is not None
    summary_msgs = [m for m in result if m.summary is not None]
    assert len(summary_msgs) == 1


def test_strategy_f_empty_middle_returns_none():
    cfg = TruncationConfig(keep_first_units=1, keep_last_units=3)
    t = make_truncator(cfg)
    units = [make_unit(f"t{i}", "a", "r") for i in range(4)]
    msgs = units_to_messages(units)
    result = t._strategy_f_summarize(msgs)
    assert result is None


# ---------------------------------------------------------------------------
# _call_summary_llm logs response
# ---------------------------------------------------------------------------

def test_call_summary_llm_logs_response():
    cfg = TruncationConfig()
    t = make_truncator(cfg)
    _setup_mock_llm(t, "the summary content")
    msgs = [LLMMessage(role="assistant", content="step1"), LLMMessage(role="tool", content="res1")]
    result = t._call_summary_llm(msgs)
    assert result is not None
    assert result.content == "the summary content"
    t._logger.info.assert_called_with(
        "Strategy F: summary LLM response", content="the summary content"
    )
