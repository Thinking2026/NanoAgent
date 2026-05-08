from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from agent.models.context.context_manager import ContextManager, ContextMessage
from schemas.types import LLMMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ctx() -> ContextManager:
    logger = MagicMock()
    tracer = MagicMock()
    return ContextManager(logger=logger, tracer=tracer)


def make_llm_msg(role: str, content: str) -> LLMMessage:
    return LLMMessage(role=role, content=content)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def test_set_and_get_system_prompt():
    ctx = make_ctx()
    ctx.set_system_prompt("You are an agent.")
    assert ctx.get_system_prompt() == "You are an agent."


def test_append_system_prompt():
    ctx = make_ctx()
    ctx.set_system_prompt("Hello")
    ctx.append_system_prompt(" World")
    assert ctx.get_system_prompt() == "Hello World"


def test_append_system_prompt_line():
    ctx = make_ctx()
    ctx.set_system_prompt("Line1")
    ctx.append_system_prompt_line("Line2")
    assert ctx.get_system_prompt() == "Line1\nLine2"


def test_default_system_prompt_is_empty():
    ctx = make_ctx()
    assert ctx.get_system_prompt() == ""


# ---------------------------------------------------------------------------
# add_message / get_conversation_history
# ---------------------------------------------------------------------------

def test_add_and_get_history():
    ctx = make_ctx()
    ctx.add_message("user", "hello")
    ctx.add_message("assistant", "hi")
    history = ctx.get_conversation_history()
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"


def test_add_message_returns_uuid():
    ctx = make_ctx()
    msg_id = ctx.add_message("user", "test")
    assert isinstance(msg_id, str) and len(msg_id) == 36


def test_add_message_with_name_and_tool_call_id():
    ctx = make_ctx()
    ctx.add_message("tool", "result", name="my_tool", tool_call_id="call-123")
    # Verify the ContextMessage stored the fields
    assert ctx._ctx_window[0].tool_name == "my_tool"
    assert ctx._ctx_window[0].tool_call_id == "call-123"


def test_add_message_caches_token_count():
    ctx = make_ctx()
    ctx.add_message("user", "hello world")
    msg: ContextMessage = ctx._ctx_window[0]
    assert msg.token_count is not None and msg.token_count > 0


def test_add_message_has_timestamp():
    ctx = make_ctx()
    ctx.add_message("user", "hi")
    assert ctx._ctx_window[0].timestamp is not None


def test_get_conversation_history_returns_copies():
    ctx = make_ctx()
    ctx.add_message("user", "original")
    history = ctx.get_conversation_history()
    # LLMMessage is not frozen, but mutating the returned list should not affect internal state
    history.clear()
    assert len(ctx.get_conversation_history()) == 1


# ---------------------------------------------------------------------------
# replace_conversation_history
# ---------------------------------------------------------------------------

def test_replace_conversation_history():
    ctx = make_ctx()
    ctx.add_message("user", "old")
    new_msgs = [make_llm_msg("user", "new1"), make_llm_msg("assistant", "new2")]
    ctx.replace_conversation_history(new_msgs)
    history = ctx.get_conversation_history()
    assert len(history) == 2
    assert history[0].content == "new1"


def test_replace_conversation_history_resets_stages():
    ctx = make_ctx()
    ctx.begin_stage(0)
    ctx.add_message("user", "stage msg")
    ctx.replace_conversation_history([make_llm_msg("user", "fresh")])
    assert ctx._stage_records == []
    assert ctx._active_stage_index is None


def test_replace_conversation_history_updates_token_count():
    ctx = make_ctx()
    ctx.replace_conversation_history([make_llm_msg("user", "hello world")])
    assert ctx._current_token_count > 0


# ---------------------------------------------------------------------------
# Stage lifecycle
# ---------------------------------------------------------------------------

def test_begin_stage_sets_active_index():
    ctx = make_ctx()
    ctx.begin_stage(0)
    assert ctx._active_stage_index == 0


def test_begin_stage_stores_plan_step_order():
    ctx = make_ctx()
    ctx.begin_stage(0, plan_step_order=3)
    assert ctx._stage_records[0].plan_step_order == 3


def test_messages_tagged_to_active_stage():
    ctx = make_ctx()
    ctx.begin_stage(0)
    msg_id = ctx.add_message("user", "hello")
    assert ctx._message_id_to_stage[msg_id] == 0


def test_end_stage_clears_active_index():
    ctx = make_ctx()
    ctx.begin_stage(0)
    ctx.add_message("user", "msg")
    ctx.end_stage(0, success=False)
    assert ctx._active_stage_index is None


def test_drop_stage_removes_messages_from_window():
    ctx = make_ctx()
    ctx.begin_stage(0)
    ctx.add_message("user", "stage0 msg")
    ctx.end_stage(0, success=False)
    ctx.drop_stage(0)
    assert ctx._ctx_window == []


def test_drop_stage_preserves_history():
    ctx = make_ctx()
    ctx.begin_stage(0)
    ctx.add_message("user", "stage0 msg")
    ctx.end_stage(0, success=False)
    ctx.drop_stage(0)
    assert len(ctx.get_conversation_history()) == 1


def test_drop_stage_updates_token_count():
    ctx = make_ctx()
    ctx.begin_stage(0)
    ctx.add_message("user", "hello world")
    before = ctx._current_token_count
    ctx.end_stage(0, success=False)
    ctx.drop_stage(0)
    assert ctx._current_token_count < before


def test_summarize_stage_replaces_messages():
    ctx = make_ctx()
    ctx.begin_stage(0)
    ctx.add_message("user", "msg1")
    ctx.add_message("assistant", "msg2")
    ctx.end_stage(0, success=False)
    ctx.summarize_stage(0, "Summary of stage 0")
    window = ctx._ctx_window
    assert len(window) == 1
    assert window[0].content == "Summary of stage 0"
    assert window[0].metadata.get("summarized") is True


def test_summarize_stage_updates_token_count():
    ctx = make_ctx()
    ctx.begin_stage(0)
    ctx.add_message("user", "a" * 100)
    ctx.add_message("assistant", "b" * 100)
    ctx.end_stage(0, success=False)
    ctx.summarize_stage(0, "short")
    assert ctx._current_token_count > 0
    assert ctx._current_token_count < 60  # much less than 200 chars / 3.5


def test_get_stage_messages_returns_empty_for_dropped():
    ctx = make_ctx()
    ctx.begin_stage(0)
    ctx.add_message("user", "msg")
    ctx.end_stage(0, success=False)
    ctx.drop_stage(0)
    assert ctx.get_stage_messages(0) == []


# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------

def test_current_token_count_increments_on_add():
    ctx = make_ctx()
    assert ctx._current_token_count == 0
    ctx.add_message("user", "hello world")
    assert ctx._current_token_count > 0


def test_current_token_count_resets_on_reset():
    ctx = make_ctx()
    ctx.add_message("user", "hello")
    ctx.reset()
    assert ctx._current_token_count == 0


def test_context_pressure_callback_fires():
    ctx = make_ctx()
    fired: list[float] = []
    ctx.set_context_pressure_callback(lambda r: fired.append(r), threshold=0.0)
    ctx.add_message("user", "any message")
    assert len(fired) > 0
    assert fired[0] >= 0.0


def test_context_pressure_callback_not_fired_below_threshold():
    ctx = make_ctx()
    fired: list[float] = []
    ctx.set_context_pressure_callback(lambda r: fired.append(r), threshold=1.1)
    ctx.add_message("user", "short")
    assert fired == []


# ---------------------------------------------------------------------------
# Streaming messages
# ---------------------------------------------------------------------------

def test_streaming_message_basic():
    ctx = make_ctx()
    sid = ctx.begin_streaming_message("assistant")
    ctx.append_streaming_chunk(sid, "Hello")
    ctx.append_streaming_chunk(sid, " world")
    msg_id = ctx.end_streaming_message(sid)
    assert msg_id is not None
    assert ctx._ctx_window[-1].content == "Hello world"
    assert ctx._ctx_window[-1].role == "assistant"


def test_streaming_message_committed_to_history():
    ctx = make_ctx()
    sid = ctx.begin_streaming_message("user")
    ctx.append_streaming_chunk(sid, "streamed")
    ctx.end_streaming_message(sid)
    assert len(ctx.get_conversation_history()) == 1


def test_streaming_message_unknown_id_returns_none():
    ctx = make_ctx()
    result = ctx.end_streaming_message("nonexistent-id")
    assert result is None


def test_streaming_buffers_cleared_after_end():
    ctx = make_ctx()
    sid = ctx.begin_streaming_message("user")
    ctx.append_streaming_chunk(sid, "data")
    ctx.end_streaming_message(sid)
    assert sid not in ctx._streaming_buffers


# ---------------------------------------------------------------------------
# _repair_tool_pairs
# ---------------------------------------------------------------------------

def _make_ctx_msg(role, content, metadata=None, tool_call_id=None):
    from agent.models.context.context_manager import ContextMessage
    from datetime import datetime, timezone
    return ContextMessage(
        id=str(__import__("uuid").uuid4()),
        role=role,
        content=content,
        tool_call_id=tool_call_id,
        metadata=metadata or {},
    )


def test_repair_removes_trailing_unmatched_assistant():
    from agent.models.context.context_manager import ContextManager
    msgs = [
        _make_ctx_msg("user", "q"),
        _make_ctx_msg("assistant", "thinking", metadata={
            "tool_calls": [{"llm_raw_tool_call_id": "call-1", "name": "search", "arguments": {}}]
        }),
    ]
    repaired = ContextManager._repair_tool_pairs(msgs)
    assert len(repaired) == 1
    assert repaired[0].role == "user"


def test_repair_keeps_matched_tool_pair():
    from agent.models.context.context_manager import ContextManager
    msgs = [
        _make_ctx_msg("user", "q"),
        _make_ctx_msg("assistant", "thinking", metadata={
            "tool_calls": [{"llm_raw_tool_call_id": "call-1", "name": "search", "arguments": {}}]
        }),
        _make_ctx_msg("tool", "result", tool_call_id="call-1"),
    ]
    repaired = ContextManager._repair_tool_pairs(msgs)
    assert len(repaired) == 3


def test_repair_removes_orphaned_tool_result():
    from agent.models.context.context_manager import ContextManager
    # assistant with call-1 is present, but tool result references call-99 (orphan)
    msgs = [
        _make_ctx_msg("user", "q"),
        _make_ctx_msg("assistant", "thinking", metadata={
            "tool_calls": [{"llm_raw_tool_call_id": "call-1", "name": "search", "arguments": {}}]
        }),
        _make_ctx_msg("tool", "result1", tool_call_id="call-1"),
        _make_ctx_msg("tool", "orphan", tool_call_id="call-99"),
    ]
    repaired = ContextManager._repair_tool_pairs(msgs)
    tool_msgs = [m for m in repaired if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "call-1"


# ---------------------------------------------------------------------------
# reset / release
# ---------------------------------------------------------------------------

def test_reset_clears_window_and_stages():
    ctx = make_ctx()
    ctx.begin_stage(0)
    ctx.add_message("user", "msg")
    ctx.reset()
    assert ctx._ctx_window == []
    assert ctx._stage_records == []
    assert ctx._current_token_count == 0


def test_reset_preserves_history():
    ctx = make_ctx()
    ctx.add_message("user", "msg")
    ctx.reset()
    assert len(ctx.get_conversation_history()) == 1


def test_release_clears_everything():
    ctx = make_ctx()
    ctx.set_system_prompt("prompt")
    ctx.add_message("user", "msg")
    ctx.release()
    assert ctx.get_system_prompt() == ""
    assert ctx.get_conversation_history() == []
    assert ctx._current_token_count == 0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

def test_concurrent_add_messages():
    ctx = make_ctx()
    errors: list[Exception] = []

    def add_messages():
        try:
            for i in range(50):
                ctx.add_message("user", f"msg{i}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=add_messages) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(ctx.get_conversation_history()) == 200
