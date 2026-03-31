# LLM Response Schema

## Purpose

This project normalizes all provider responses into one internal schema so upper layers (`ReActAgent`, tools, message queue) can stay provider-agnostic.

The canonical interface is:

- `BaseLLMClient.generate(request: LLMRequest) -> LLMResponse`

Code references:

- `llm/llm_api.py`
- `schemas/types.py`
- `llm/impl/openai_api.py`
- `llm/impl/claude_api.py`

## Canonical Types

### `LLMResponse`

```python
@dataclass(slots=True)
class LLMResponse:
    assistant_message: ChatMessage
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = "stop"
```

### `ChatMessage`

```python
@dataclass(slots=True)
class ChatMessage:
    role: Literal["user", "assistant", "conversation"]
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

For `LLMResponse.assistant_message`, `role` should be `"assistant"` in normal cases.

### `ToolCall`

```python
@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    llm_raw_tool_call_id: str | None = None
```

## Normalization Contract

Any provider client implementation MUST:

1. Return `LLMResponse` on success.
2. Put parsed tool calls into `tool_calls` (not only raw payload).
3. Keep provider payload in `raw_response` for debugging/traceability.
4. Fill `finish_reason` with provider stop reason (or `"stop"` if unavailable).
5. Raise project errors (`build_error(...)`) on malformed or failed responses.

## Field Semantics

- `assistant_message.content`
  - Human-readable assistant output (may be empty when tool calls are returned).
- `assistant_message.metadata`
  - Provider-specific supplemental data (recommended keys: `tool_calls_count`, `tool_calls`).
- `tool_calls`
  - Structured, validated tool requests for agent runtime execution.
  - `llm_raw_tool_call_id` preserves the raw tool-call id returned by the upstream LLM provider.
- `raw_response`
  - Original provider JSON object (for logs and troubleshooting).
- `finish_reason`
  - Provider completion status, normalized to string.

## Minimal Success Example

```json
{
  "assistant_message": {
    "role": "assistant",
    "content": "I can help with that.",
    "metadata": {}
  },
  "tool_calls": [],
  "raw_response": {},
  "finish_reason": "stop"
}
```

## Tool Call Example

```json
{
  "assistant_message": {
    "role": "assistant",
    "content": "",
    "metadata": {
      "tool_calls_count": 1
    }
  },
  "tool_calls": [
    {
      "name": "current_time",
      "arguments": {
        "timezone": "Asia/Shanghai"
      },
      "llm_raw_tool_call_id": "call_123"
    }
  ],
  "raw_response": {
    "...": "provider payload"
  },
  "finish_reason": "tool_calls"
}
```

## Error Handling Guidance

Provider clients should use these error classes/codes:

- `LLM_HTTP_ERROR`
- `LLM_NETWORK_ERROR`
- `LLM_TIMEOUT`
- `LLM_RESPONSE_PARSE_ERROR`
- `LLM_RESPONSE_ERROR`

For retry exhaustion or fallback-chain total failure, wrapper client may raise:

- `LLM_ALL_PROVIDERS_FAILED`

Provider fallback is optional and controlled by config:

- `llm.retry.*` always controls per-provider exponential backoff retries.
- `llm.enable_provider_fallback` decides whether to continue to the next provider in `llm.priority_chain` after retries are exhausted.
- `llm.context_trimming.enabled` enables request-side conversation trimming before messages are sent to the provider.
- `llm.context_trimming.max_messages` keeps only the most recent conversation messages. The system prompt remains separate and is always preserved.

## Compatibility Note

`deepseek_api.py` and `qwen_api.py` currently inherit OpenAI-compatible parsing/serialization from `OpenAILLMClient`, so they follow this schema implicitly through the shared implementation.
