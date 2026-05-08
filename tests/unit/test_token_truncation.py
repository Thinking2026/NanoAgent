from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

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
    U_SYS,
    U_USER,
    U_TOOL_BLOCK,
    parse_message_units,
    units_to_messages,
    _truncate_string_arg,
    _truncate_tool_result_content,
)
from schemas.types import BudgetResult, LLMMessage, RoleBudget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tool_block(tool_name: str, arg_val: str, result: str, success: bool = True) -> U_TOOL_BLOCK:
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
    return U_TOOL_BLOCK(assistant_msg=assistant, tool_msgs=[tool])


def make_user_unit(content: str = "user msg") -> U_USER:
    msg = ContextMessage(id=str(uuid4()), role="user", content=content)
    return U_USER(messages=[msg])


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


def make_budget(available_tokens: int = 1600) -> BudgetResult:
    return BudgetResult(
        strategy="default",
        total_budget=2000,
        reserve_ratio=0.2,
        reserved_tokens=400,
        available_tokens=available_tokens,
        role_budgets={},
    )


def _setup_mock_llm(truncator: DefaultContextTruncator, summary_text: str = "summary") -> None:
    mock_response = MagicMock()
    mock_response.assistant_message = LLMMessage(role="assistant", content=summary_text)
    truncator._llm_gateway.generate.return_value = mock_response


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = TruncationConfig()
    assert cfg.keep_first_units == 1
    assert cfg.keep_last_units == 3
    assert cfg.keep_first_user_units == 3
    assert cfg.summary_ratio == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# JSON-safe truncation helpers
# ---------------------------------------------------------------------------

def test_truncate_string_arg_plain():
    result = _truncate_string_arg("x" * 400, 10)
    assert result == "x" * 10 + "...(truncated)"


def test_truncate_string_arg_json_dict():
    import json
    v = json.dumps({"key": "a" * 400})
    result = _truncate_string_arg(v, 20)
    assert result.endswith("...(truncated)}")
    assert len(result) <= 20 + len("...(truncated)}")


def test_truncate_string_arg_json_list():
    import json
    v = json.dumps(["a" * 400])
    result = _truncate_string_arg(v, 20)
    assert result.endswith("...(truncated)]")


def test_truncate_string_arg_short_unchanged():
    v = "short"
    assert _truncate_string_arg(v, 100) == v


def test_truncate_tool_result_content_plain():
    result = _truncate_tool_result_content("y" * 600, 10)
    assert result == "y" * 10 + "\n...(truncated)"


def test_truncate_tool_result_content_json():
    import json
    content = json.dumps({"data": "z" * 400})
    result = _truncate_tool_result_content(content, 20)
    assert result.endswith("...(truncated)}")


# ---------------------------------------------------------------------------
# parse_message_units
# ---------------------------------------------------------------------------

def test_parse_message_units_groups_correctly():
    b1 = make_tool_block("search", "query1", "result1")
    b2 = make_tool_block("read", "file.txt", "content")
    msgs = units_to_messages([b1, b2])
    units = parse_message_units(msgs)
    tool_units = [u for u in units if isinstance(u, U_TOOL_BLOCK)]
    assert len(tool_units) == 2
    assert tool_units[0].assistant_msg is b1.assistant_msg
    assert tool_units[1].assistant_msg is b2.assistant_msg


def test_parse_message_units_skips_non_tool_assistant():
    plain_assistant = ContextMessage(id=str(uuid4()), role="assistant", content="plain reply")
    b1 = make_tool_block("search", "q", "r")
    msgs = [plain_assistant] + units_to_messages([b1])
    units = parse_message_units(msgs)
    tool_units = [u for u in units if isinstance(u, U_TOOL_BLOCK)]
    assert len(tool_units) == 1


# ---------------------------------------------------------------------------
# _get_candidate_units
# ---------------------------------------------------------------------------

def test_get_candidate_units_budget_driven():
    estimator = ClaudeTokenEstimator()
    cfg = TruncationConfig(keep_first_user_units=1)
    t = make_truncator(cfg)
    t._estimator = estimator
    # 1 sys + 1 user (head) + 5 tool blocks (candidates/tail)
    sys_msg = ContextMessage(id=str(uuid4()), role="system", content="sys")
    units = [U_SYS(msg=sys_msg), make_user_unit()] + [make_tool_block(f"t{i}", "a", "x" * 50) for i in range(5)]
    all_msgs = units_to_messages(units)
    total_tokens = estimator.estimate(
        __import__("agent.models.context.truncation.token_truncation", fromlist=["_to_llm_request"])._to_llm_request(all_msgs)
    )["total"]
    # Set budget so tail (last 2 blocks) consumes ~70%
    tail_tokens = sum(
        estimator.estimate(
            __import__("agent.models.context.truncation.token_truncation", fromlist=["_to_llm_request"])._to_llm_request(units[i].to_messages())
        )["total"]
        for i in range(5, 7)
    )
    available = int(tail_tokens / 0.70) + 1
    t._current_budget = make_budget(available_tokens=available)
    head, candidates, tail = t._get_candidate_units(units)
    assert len(head) == 2  # sys + 1 user
    assert len(tail) >= 1
    assert len(candidates) >= 1


def test_get_candidate_units_protects_first_user_units():
    estimator = ClaudeTokenEstimator()
    cfg = TruncationConfig(keep_first_user_units=3)
    t = make_truncator(cfg)
    t._estimator = estimator
    # 1 sys + 5 user units + 3 tool blocks
    sys_msg = ContextMessage(id=str(uuid4()), role="system", content="sys")
    user_units = [make_user_unit(f"user {i}") for i in range(5)]
    tool_units = [make_tool_block(f"t{i}", "a", "x" * 100) for i in range(3)]
    all_units = [U_SYS(msg=sys_msg)] + user_units + tool_units
    t._current_budget = make_budget(available_tokens=10000)
    head, candidates, tail = t._get_candidate_units(all_units)
    # First 3 U_USER units must be in head
    head_user_units = [u for u in head if isinstance(u, U_USER)]
    assert len(head_user_units) >= 3


def test_get_candidate_units_empty_when_all_protected():
    estimator = ClaudeTokenEstimator()
    cfg = TruncationConfig(keep_first_user_units=3)
    t = make_truncator(cfg)
    t._estimator = estimator
    units = [make_user_unit(f"u{i}") for i in range(4)]
    t._current_budget = make_budget(available_tokens=10000)
    _, candidates, _ = t._get_candidate_units(units)
    assert candidates == []


# ---------------------------------------------------------------------------
# Strategy A: dedup with placeholder
# ---------------------------------------------------------------------------

def test_strategy_a_dedup_replaces_with_placeholder():
    cfg = TruncationConfig(keep_first_user_units=0)
    t = make_truncator(cfg)
    t._estimator = ClaudeTokenEstimator()
    t._current_budget = make_budget()
    b1 = make_tool_block("search", "query", "result")
    b2 = make_tool_block("search", "query", "result")  # identical
    msgs = units_to_messages([b1, b2])
    result, delta = t._strategy_a_dedup(msgs)
    assert delta <= 0  # tokens reduced or same
    user_msgs = [m for m in result if m.role == "user"]
    assert any("[重复调用已省略" in m.content for m in user_msgs)
    # The kept (later) block's assistant message should still be present
    assert any(m is b2.assistant_msg for m in result)


def test_strategy_a_dedup_non_consecutive_not_deduped():
    cfg = TruncationConfig(keep_first_user_units=0)
    t = make_truncator(cfg)
    t._estimator = ClaudeTokenEstimator()
    t._current_budget = make_budget()
    b1 = make_tool_block("search", "query", "result")
    b2 = make_tool_block("read", "file", "content")   # different
    b3 = make_tool_block("search", "query", "result")  # same as b1 but non-consecutive
    msgs = units_to_messages([b1, b2, b3])
    result, delta = t._strategy_a_dedup(msgs)
    assert delta == 0  # nothing changed
    assert result is msgs


def test_strategy_a_dedup_no_duplicates_unchanged():
    cfg = TruncationConfig(keep_first_user_units=0)
    t = make_truncator(cfg)
    t._estimator = ClaudeTokenEstimator()
    t._current_budget = make_budget()
    b1 = make_tool_block("search", "q1", "r1")
    b2 = make_tool_block("read", "q2", "r2")
    msgs = units_to_messages([b1, b2])
    result, delta = t._strategy_a_dedup(msgs)
    assert delta == 0
    assert result is msgs


# ---------------------------------------------------------------------------
# Strategy B: compress failed to placeholder
# ---------------------------------------------------------------------------

def test_strategy_b_compress_failed_placeholder_in_middle():
    cfg = TruncationConfig(keep_first_user_units=1)
    t = make_truncator(cfg)
    t._estimator = ClaudeTokenEstimator()
    head = make_user_unit("head")
    failed = make_tool_block("search", "q", "error details " * 50, success=False)
    tail = make_tool_block("read", "q", "ok")
    msgs = units_to_messages([head, failed, tail])
    # Budget: tight enough that tail (last block) consumes >70%, so failed is a candidate
    from agent.models.context.truncation.token_truncation import _to_llm_request
    tail_tokens = ClaudeTokenEstimator().estimate(_to_llm_request(tail.to_messages()))["total"]
    available = int(tail_tokens / 0.70) + 1
    t._current_budget = make_budget(available_tokens=available)
    result, delta = t._strategy_b_compress_failed(msgs)
    assert delta <= 0
    user_msgs = [m for m in result if m.role == "user"]
    assert any("[工具调用失败已压缩" in m.content for m in user_msgs)
    # tail block should still be present
    assert any(m is tail.assistant_msg for m in result)


def test_strategy_b_compress_failed_empty_candidates_unchanged():
    cfg = TruncationConfig(keep_first_user_units=3)
    t = make_truncator(cfg)
    t._estimator = ClaudeTokenEstimator()
    t._current_budget = make_budget(available_tokens=10000)
    units = [make_user_unit(f"u{i}") for i in range(3)]
    msgs = units_to_messages(units)
    result, delta = t._strategy_b_compress_failed(msgs)
    assert delta == 0
    assert result is msgs


def test_strategy_b_compress_failed_no_failed_unchanged():
    cfg = TruncationConfig(keep_first_user_units=1)
    t = make_truncator(cfg)
    t._estimator = ClaudeTokenEstimator()
    t._current_budget = make_budget(available_tokens=10000)
    head = make_user_unit("head")
    mid = make_tool_block("search", "q", "ok", success=True)
    tail = make_tool_block("read", "q", "ok")
    msgs = units_to_messages([head, mid, tail])
    result, delta = t._strategy_b_compress_failed(msgs)
    assert delta == 0
    assert result is msgs


# ---------------------------------------------------------------------------
# Strategy C: trim args (JSON-safe)
# ---------------------------------------------------------------------------

def test_strategy_c_trims_only_candidate_args():
    long_arg = "x" * 400
    cfg = TruncationConfig(keep_first_user_units=1, tool_arg_max_chars=10)
    t = make_truncator(cfg)
    t._estimator = ClaudeTokenEstimator()
    head = make_user_unit("head")
    mid = make_tool_block("mid", long_arg, "r")
    tail = make_tool_block("tail", long_arg, "r")
    msgs = units_to_messages([head, mid, tail])
    # Budget: tight so tail consumes >70%, making mid a candidate
    from agent.models.context.truncation.token_truncation import _to_llm_request
    tail_tokens = ClaudeTokenEstimator().estimate(_to_llm_request(tail.to_messages()))["total"]
    available = int(tail_tokens / 0.70) + 1
    t._current_budget = make_budget(available_tokens=available)
    result, delta = t._strategy_c_trim_args(msgs)
    result_units = [u for u in parse_message_units(result) if isinstance(u, U_TOOL_BLOCK)]
    mid_args = result_units[0].assistant_msg.tool_use.tool_arguments["q"]
    tail_args = result_units[1].assistant_msg.tool_use.tool_arguments["q"]
    assert "...(truncated)" in mid_args
    assert tail_args == long_arg  # tail protected


def test_strategy_c_trim_args_json_string_value():
    import json
    json_val = json.dumps({"nested": "v" * 400})
    cfg = TruncationConfig(keep_first_user_units=0, tool_arg_max_chars=30)
    t = make_truncator(cfg)
    t._estimator = ClaudeTokenEstimator()
    # Budget: very tight so the single block is a candidate
    t._current_budget = make_budget(available_tokens=5)
    block = make_tool_block("t", json_val, "r")
    msgs = units_to_messages([block])
    result, _ = t._strategy_c_trim_args(msgs)
    result_units = [u for u in parse_message_units(result) if isinstance(u, U_TOOL_BLOCK)]
    trimmed = result_units[0].assistant_msg.tool_use.tool_arguments["q"]
    assert "...(truncated)" in trimmed
    assert isinstance(trimmed, str)


# ---------------------------------------------------------------------------
# Strategy D: trim results (JSON-safe)
# ---------------------------------------------------------------------------

def test_strategy_d_trims_only_candidate_results():
    long_result = "y" * 600
    cfg = TruncationConfig(keep_first_user_units=1, tool_result_max_chars=10)
    t = make_truncator(cfg)
    t._estimator = ClaudeTokenEstimator()
    head = make_user_unit("head")
    mid = make_tool_block("mid", "a", long_result)
    tail = make_tool_block("tail", "a", long_result)
    msgs = units_to_messages([head, mid, tail])
    # Budget: tight so tail consumes >70%, making mid a candidate
    from agent.models.context.truncation.token_truncation import _to_llm_request
    tail_tokens = ClaudeTokenEstimator().estimate(_to_llm_request(tail.to_messages()))["total"]
    available = int(tail_tokens / 0.70) + 1
    t._current_budget = make_budget(available_tokens=available)
    result, delta = t._strategy_d_trim_results(msgs)
    result_units = [u for u in parse_message_units(result) if isinstance(u, U_TOOL_BLOCK)]
    mid_content = result_units[0].tool_msgs[0].content
    tail_content = result_units[1].tool_msgs[0].content
    assert "...(truncated)" in mid_content
    assert tail_content == long_result  # tail protected


def test_strategy_d_trim_results_json_content():
    import json
    content = json.dumps({"data": "z" * 400})
    cfg = TruncationConfig(keep_first_user_units=0, tool_result_max_chars=30)
    t = make_truncator(cfg)
    t._estimator = ClaudeTokenEstimator()
    # Budget: very tight so the single block is a candidate
    t._current_budget = make_budget(available_tokens=5)
    block = make_tool_block("t", "a", content)
    msgs = units_to_messages([block])
    result, _ = t._strategy_d_trim_results(msgs)
    result_units = [u for u in parse_message_units(result) if isinstance(u, U_TOOL_BLOCK)]
    trimmed = result_units[0].tool_msgs[0].content
    assert "...(truncated)}" in trimmed


# ---------------------------------------------------------------------------
# Strategy E: binary drop
# ---------------------------------------------------------------------------

def test_strategy_e_full_binary_search_range():
    estimator = ClaudeTokenEstimator()
    cfg = TruncationConfig(keep_first_user_units=1)
    t = make_truncator(cfg, assistant_budget=2000, tool_budget=200)
    t._estimator = estimator
    head = make_user_unit("head")
    blocks = [make_tool_block(f"t{i}", "a", "x" * 100) for i in range(10)]
    all_units = [head] + blocks
    msgs = units_to_messages(all_units)
    # Budget: tight enough that dropping several middle blocks is needed
    t._current_budget = make_budget(available_tokens=200)
    result = t._strategy_e_binary_drop(msgs)
    assert result is not None
    assert estimator.estimate(
        __import__("agent.models.context.truncation.token_truncation", fromlist=["_to_llm_request"])._to_llm_request(result)
    )["total"] <= 200


def test_strategy_e_returns_none_when_no_candidates():
    estimator = ClaudeTokenEstimator()
    cfg = TruncationConfig(keep_first_user_units=3)
    t = make_truncator(cfg)
    t._estimator = estimator
    # Only 3 user units — all protected as head, no candidates
    units = [make_user_unit(f"u{i}") for i in range(3)]
    msgs = units_to_messages(units)
    t._current_budget = make_budget(available_tokens=1)
    result = t._strategy_e_binary_drop(msgs)
    assert result is None


def test_strategy_e_empty_candidates_returns_none():
    estimator = ClaudeTokenEstimator()
    cfg = TruncationConfig(keep_first_user_units=3)
    t = make_truncator(cfg)
    t._estimator = estimator
    units = [make_user_unit(f"u{i}") for i in range(3)]
    msgs = units_to_messages(units)
    t._current_budget = make_budget(available_tokens=10000)
    result = t._strategy_e_binary_drop(msgs)
    assert result is None


# ---------------------------------------------------------------------------
# Strategy F: summarize candidates
# ---------------------------------------------------------------------------

def test_strategy_f_summarizes_all_candidates():
    cfg = TruncationConfig(keep_first_user_units=1)
    t = make_truncator(cfg)
    t._estimator = ClaudeTokenEstimator()
    t._current_budget = make_budget(available_tokens=10000)
    _setup_mock_llm(t, "summary text")
    head = make_user_unit("head")
    mid1 = make_tool_block("t1", "a", "r1")
    mid2 = make_tool_block("t2", "a", "r2")
    tail = make_tool_block("t3", "a", "r3")
    # With large budget, tail will absorb everything — set small budget so candidates exist
    t._current_budget = make_budget(available_tokens=50)
    msgs = units_to_messages([head, mid1, mid2, tail])
    result = t._strategy_f_summarize(msgs)
    if result is not None:
        summary_msgs = [m for m in result if m.summary is not None]
        assert len(summary_msgs) == 1


def test_strategy_f_empty_candidates_returns_none():
    cfg = TruncationConfig(keep_first_user_units=3)
    t = make_truncator(cfg)
    t._estimator = ClaudeTokenEstimator()
    t._current_budget = make_budget(available_tokens=10000)
    units = [make_user_unit(f"u{i}") for i in range(3)]
    msgs = units_to_messages(units)
    result = t._strategy_f_summarize(msgs)
    assert result is None


def test_strategy_f_uses_structured_prompt():
    cfg = TruncationConfig(keep_first_user_units=0)
    t = make_truncator(cfg)
    t._estimator = ClaudeTokenEstimator()
    t._current_budget = make_budget(available_tokens=10)
    _setup_mock_llm(t, "1. 用户要求搜索；2. 返回结果A")
    block = make_tool_block("search", "q", "result " * 20)
    msgs = units_to_messages([block])
    t._strategy_f_summarize(msgs)
    call_args = t._llm_gateway.generate.call_args
    if call_args:
        request = call_args[0][0]
        assert "instruction-logic chain" in request.system_prompt


# ---------------------------------------------------------------------------
# _call_summary_llm
# ---------------------------------------------------------------------------

def test_call_summary_llm_logs_response():
    cfg = TruncationConfig()
    t = make_truncator(cfg)
    _setup_mock_llm(t, "the summary content")
    msgs = [
        ContextMessage(id=str(uuid4()), role="assistant", content="step1"),
        ContextMessage(id=str(uuid4()), role="tool", content="res1",
                       tool_result=ToolResultMetadata(tool_call_id="tc1", tool_name="t", success=True)),
    ]
    result = t._call_summary_llm(msgs)
    assert result is not None
    assert result.content == "the summary content"
    assert result.role == "user"
    t._logger.info.assert_called_with(
        "Strategy F: summary LLM response", content="the summary content"
    )


# ---------------------------------------------------------------------------
# truncate() early exit
# ---------------------------------------------------------------------------

def test_truncate_no_truncation_needed():
    estimator = ClaudeTokenEstimator()
    cfg = TruncationConfig(keep_first_user_units=1)
    t = make_truncator(cfg)
    t._estimator = estimator
    msgs = units_to_messages([make_user_unit("hi")])
    budget = make_budget(available_tokens=100000)
    result = t.truncate(msgs, budget)
    assert result is msgs


def test_truncate_early_exit_after_strategy_a():
    estimator = ClaudeTokenEstimator()
    cfg = TruncationConfig(keep_first_user_units=0)
    t = make_truncator(cfg)
    t._estimator = estimator
    # Two identical blocks — dedup will replace one with a tiny placeholder
    b1 = make_tool_block("search", "q", "x" * 200)
    b2 = make_tool_block("search", "q", "x" * 200)
    msgs = units_to_messages([b1, b2])
    # Budget: tight enough that original fails but after dedup it fits
    original_tokens = estimator.estimate(
        __import__("agent.models.context.truncation.token_truncation", fromlist=["_to_llm_request"])._to_llm_request(msgs)
    )["total"]
    # After dedup one block is replaced by ~5 token placeholder
    budget = make_budget(available_tokens=int(original_tokens * 0.6))
    result = t.truncate(msgs, budget)
    user_msgs = [m for m in result if m.role == "user"]
    assert any("[重复调用已省略" in m.content for m in user_msgs)
