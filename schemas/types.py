from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentEvent:
    event_type: str
    payload: dict[str, Any]
    timestamp: str = field(default_factory=utc_now_iso)


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass(slots=True)
class ToolResult:
    call_id: str
    output: str
    success: bool = True
    error: str | None = None


@dataclass(slots=True)
class LLMRequest:
    system_prompt: str
    messages: list[ChatMessage]
    tools: list[dict[str, Any]]
    context: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class LLMResponse:
    assistant_message: ChatMessage
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = "stop"
