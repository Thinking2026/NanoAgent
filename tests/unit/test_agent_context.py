from __future__ import annotations

import threading

import pytest

from agent.models.context.context_manager import ContextManager
from schemas.types import LLMMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_msg(role: str, content: str) -> LLMMessage:
    return LLMMessage(role=role, content=content)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def test_set_and_get_system_prompt():
    ctx = ContextManager()
    ctx.set_system_prompt("You are an agent.")
    assert ctx.get_system_prompt() == "You are an agent."


def test_append_system_prompt():
    ctx = ContextManager()
    ctx.set_system_prompt("Hello")
    ctx.append_system_prompt(" World")
    assert ctx.get_system_prompt() == "Hello World"


def test_append_system_prompt_line():
    ctx = ContextManager()
    ctx.set_system_prompt("Line1")
    ctx.append_system_prompt_line("Line2")
    assert ctx.get_system_prompt() == "Line1\nLine2"


def test_default_system_prompt_is_empty():
    ctx = ContextManager()
    assert ctx.get_system_prompt() == ""


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

def test_append_and_get_conversation():
    ctx = ContextManager()
    ctx.append_conversation_message(make_msg("user", "hello"))
    ctx.append_conversation_message(make_msg("assistant", "hi"))
    history = ctx.get_conversation_history()
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"


def test_get_conversation_history_returns_copies():
    ctx = ContextManager()
    ctx.append_conversation_message(make_msg("user", "original"))
    history = ctx.get_conversation_history()
    history[0].content = "modified"
    # Internal state should be unchanged
    assert ctx.get_conversation_history()[0].content == "original"


def test_clear_conversation_history():
    ctx = ContextManager()
    ctx.append_conversation_message(make_msg("user", "msg"))
    ctx.clear_conversation_history()
    assert ctx.get_conversation_history() == []


def test_clear_current_task():
    ctx = ContextManager()
    ctx.append_conversation_message(make_msg("user", "msg"))
    ctx.clear_current_task()
    assert ctx.get_current_task_messages() == []


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------

def test_archive_current_task():
    ctx = ContextManager()
    ctx.append_conversation_message(make_msg("user", "task1"))
    ctx.archive_current_task()
    assert ctx.get_current_task_messages() == []
    archived = ctx.get_archived_tasks()
    assert len(archived) == 1
    assert archived[0][0].content == "task1"


def test_archive_empty_task_does_nothing():
    ctx = ContextManager()
    ctx.archive_current_task()
    assert ctx.get_archived_tasks() == []


def test_archived_tasks_included_in_history():
    ctx = ContextManager()
    ctx.append_conversation_message(make_msg("user", "task1"))
    ctx.archive_current_task()
    ctx.append_conversation_message(make_msg("user", "task2"))
    history = ctx.get_conversation_history()
    assert len(history) == 2
    assert history[0].content == "task1"
    assert history[1].content == "task2"


def test_multiple_archives():
    ctx = ContextManager()
    for i in range(3):
        ctx.append_conversation_message(make_msg("user", f"task{i}"))
        ctx.archive_current_task()
    assert len(ctx.get_archived_tasks()) == 3
    assert len(ctx.get_conversation_history()) == 3


# ---------------------------------------------------------------------------
# replace_conversation_history
# ---------------------------------------------------------------------------

def test_replace_conversation_history():
    ctx = ContextManager()
    ctx.append_conversation_message(make_msg("user", "old"))
    ctx.archive_current_task()
    new_msgs = [make_msg("user", "new1"), make_msg("assistant", "new2")]
    ctx.replace_conversation_history(new_msgs)
    history = ctx.get_conversation_history()
    assert len(history) == 2
    assert history[0].content == "new1"
    assert ctx.get_archived_tasks() == []


def test_replace_conversation_history_returns_copies():
    ctx = ContextManager()
    msgs = [make_msg("user", "original")]
    ctx.replace_conversation_history(msgs)
    msgs[0].content = "modified"
    assert ctx.get_conversation_history()[0].content == "original"


# ---------------------------------------------------------------------------
# release
# ---------------------------------------------------------------------------

def test_release_clears_everything():
    ctx = ContextManager()
    ctx.set_system_prompt("prompt")
    ctx.append_conversation_message(make_msg("user", "msg"))
    ctx.archive_current_task()
    ctx.release()
    assert ctx.get_system_prompt() == ""
    assert ctx.get_conversation_history() == []
    assert ctx.get_archived_tasks() == []


# ---------------------------------------------------------------------------
# Thread safety (basic smoke test)
# ---------------------------------------------------------------------------

def test_concurrent_appends():
    ctx = ContextManager()
    errors = []

    def append_messages():
        try:
            for i in range(50):
                ctx.append_conversation_message(make_msg("user", f"msg{i}"))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=append_messages) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(ctx.get_conversation_history()) == 200
