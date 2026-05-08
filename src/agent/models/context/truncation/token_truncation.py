from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable
from uuid import uuid4

from agent.models.context.estimator.token_estimator import BaseTokenEstimator
from agent.models.context.context_manager import (
    ContextMessage,
    SummaryMetadata,
    ToolCallEntry,
    ToolUseMetadata,
)
from infra.observability.tracing.tracer import Tracer
from llm.llm_gateway import LLMGateway
from schemas.types import BudgetResult, LLMMessage, UnifiedLLMRequest
from utils.log.log import Logger, zap

if TYPE_CHECKING:
    from config import ConfigReader

# ===========================================================================
# Base class
# ===========================================================================

class ContextTruncator(ABC):
    @abstractmethod
    def truncate(
        self,
        messages: list[ContextMessage],
        budget: BudgetResult,
    ) -> list[ContextMessage]:
        ...

# ===========================================================================
# Message units
# ===========================================================================

@dataclass
class U_SYS:
    """The leading system message, if present."""
    msg: ContextMessage

    def to_messages(self) -> list[ContextMessage]:
        return [self.msg]


@dataclass
class U_USER:
    """A single user message plus any immediately-following plain assistant replies."""
    messages: list[ContextMessage] = field(default_factory=list)

    def to_messages(self) -> list[ContextMessage]:
        return list(self.messages)


@dataclass
class U_TOOL_BLOCK:
    """One assistant tool-call + all parallel tool results + any trailing plain assistant replies."""
    assistant_msg: ContextMessage
    tool_msgs: list[ContextMessage] = field(default_factory=list)
    trailing_msgs: list[ContextMessage] = field(default_factory=list)

    def to_messages(self) -> list[ContextMessage]:
        return [self.assistant_msg, *self.tool_msgs, *self.trailing_msgs]


Unit = U_SYS | U_USER | U_TOOL_BLOCK


def parse_message_units(messages: list[ContextMessage]) -> list[Unit]:
    """
    Partition a flat message list into structured units:
      - U_SYS   : the leading system message (at most one, at index 0)
      - U_USER  : a single user message + any trailing plain-text assistant replies
      - U_TOOL_BLOCK : one assistant tool-call + its tool results + any trailing plain-text assistant replies

    Plain-text assistant messages (no tool_use) are never standalone units; they are
    appended to the tail of the nearest preceding U_USER or U_TOOL_BLOCK.
    """
    units: list[Unit] = []
    i = 0

    if messages and messages[0].role == "system":
        units.append(U_SYS(msg=messages[0]))
        i = 1

    while i < len(messages):
        msg = messages[i]

        if msg.role == "user":
            units.append(U_USER(messages=[msg]))
            i += 1

        elif msg.role == "assistant" and msg.tool_use is not None:
            tool_call_ids: set[str] = set(msg.tool_use.all_call_ids())
            block = U_TOOL_BLOCK(assistant_msg=msg)
            i += 1
            while i < len(messages) and messages[i].role == "tool":
                tid = messages[i].tool_result.tool_call_id if messages[i].tool_result else None
                if tid in tool_call_ids:
                    block.tool_msgs.append(messages[i])
                    i += 1
                else:
                    break
            units.append(block)

        elif msg.role == "assistant":
            # Plain-text assistant reply — merge into the tail of the preceding unit.
            if units and isinstance(units[-1], U_USER):
                units[-1].messages.append(msg)
            elif units and isinstance(units[-1], U_TOOL_BLOCK):
                units[-1].trailing_msgs.append(msg)
            else:
                units.append(U_USER(messages=[msg]))
            i += 1

        else:
            i += 1

    return units


def units_to_messages(units: list[Unit]) -> list[ContextMessage]:
    return [m for u in units for m in u.to_messages()]

# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TruncationConfig:
    tool_arg_max_chars: int = 300
    tool_result_max_chars: int = 500
    summary_provider: str = "deepseek"
    keep_last_units: int = 3
    keep_first_user_units: int = 3


def _tool_block_signature(block: U_TOOL_BLOCK) -> str:
    """Stable string key for dedup: tool names + sorted arguments."""
    parts = []
    for entry in block.assistant_msg.tool_use.all_calls():
        parts.append(f"{entry.tool_name}:{json.dumps(entry.tool_arguments, sort_keys=True)}")
    return "|".join(parts)


def _has_failed_tool(block: U_TOOL_BLOCK) -> bool:
    return any(
        m.tool_result is not None and not m.tool_result.success
        for m in block.tool_msgs
    )


def _to_llm_request(messages: list[ContextMessage]) -> UnifiedLLMRequest:
    """Convert ContextMessages to a bare UnifiedLLMRequest for token estimation."""
    llm_msgs = []
    for m in messages:
        metadata: dict = {}
        if m.tool_use is not None:
            all_calls = [
                {
                    "name": entry.tool_name,
                    "llm_raw_tool_call_id": entry.tool_call_id,
                    "arguments": entry.tool_arguments,
                }
                for entry in m.tool_use.all_calls()
            ]
            metadata["tool_calls"] = all_calls
            metadata["tool_calls_count"] = len(all_calls)
        elif m.tool_result is not None:
            metadata["llm_raw_tool_call_id"] = m.tool_result.tool_call_id
            metadata["tool_name"] = m.tool_result.tool_name
            metadata["success"] = m.tool_result.success
        elif m.summary is not None:
            metadata["summarized"] = True
            metadata["stage_index"] = m.summary.stage_index
        llm_msgs.append(LLMMessage(role=m.role, content=m.content, metadata=metadata))
    return UnifiedLLMRequest(messages=llm_msgs)


# ---------------------------------------------------------------------------
# JSON-safe truncation helpers
# ---------------------------------------------------------------------------

def _truncate_string_arg(v: str, limit: int) -> str:
    """
    Truncate a tool argument string value to at most `limit` chars.
    If the value is a JSON object or array, truncate the serialized form and
    close the structure so the result is still a readable (though invalid-JSON)
    string. If it is a plain string, truncate with a suffix marker.
    The returned value is always a Python str safe to store in tool_arguments.
    """
    try:
        parsed = json.loads(v)
    except (json.JSONDecodeError, ValueError):
        parsed = None

    if isinstance(parsed, dict):
        serialized = json.dumps(parsed, ensure_ascii=False)
        if len(serialized) <= limit:
            return v
        return serialized[:limit] + "...(truncated)}"
    if isinstance(parsed, list):
        serialized = json.dumps(parsed, ensure_ascii=False)
        if len(serialized) <= limit:
            return v
        return serialized[:limit] + "...(truncated)]"
    if len(v) <= limit:
        return v
    return v[:limit] + "...(truncated)"


def _truncate_tool_result_content(content: str, limit: int) -> str:
    """
    Truncate tool result content to at most `limit` chars.
    Applies the same JSON-aware logic as _truncate_string_arg.
    Plain text fallback appends a newline + marker.
    """
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        parsed = None

    if isinstance(parsed, dict):
        serialized = json.dumps(parsed, ensure_ascii=False)
        if len(serialized) <= limit:
            return content
        return serialized[:limit] + "...(truncated)}"
    if isinstance(parsed, list):
        serialized = json.dumps(parsed, ensure_ascii=False)
        if len(serialized) <= limit:
            return content
        return serialized[:limit] + "...(truncated)]"
    if len(content) <= limit:
        return content
    return content[:limit] + "\n...(truncated)"


# ===========================================================================
# DefaultContextTruncator
# ===========================================================================

_SUMMARY_SYSTEM_PROMPT = (
    "You are a context compressor for an AI agent. "
    "The messages below are a segment of the agent's execution history that must be compressed into a single summary message.\n\n"
    "Your goal is to preserve the **instruction-logic chain** — the reader of your summary must be able to understand:\n"
    "1. What the user originally asked or instructed in this segment\n"
    "2. What actions the agent took (tool calls, decisions, retries)\n"
    "3. What each tool returned (key facts, data, errors)\n"
    "4. The outcome: what succeeded, what failed, and what was concluded\n\n"
    "Write the summary as a numbered list of key events. Be concrete and specific — include tool names, key values, and outcomes. "
    "Do NOT write vague phrases like \"some tools were called\". Example format:\n\n"
    "1. 用户要求使用方案A搜索数据集；\n"
    "2. search_tool(\"dataset_A\") 返回了数据集B和C，共12条记录；\n"
    "3. read_tool(\"file.csv\") 第二次调用因超时失败，已切换至接口D；\n"
    "4. 当前已完成数据加载，待执行步骤：数据清洗。\n\n"
    "Output only the summary list. No preamble, no explanation. Write in the same language as the input."
)


class DefaultContextTruncator(ContextTruncator):
    def __init__(
        self,
        logger: Logger,
        config: ConfigReader,
        tracer: Tracer,
        llm_gateway: LLMGateway,
        estimator: BaseTokenEstimator,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._llm_gateway = llm_gateway
        self._config = config
        self._estimator = estimator
        self._current_budget: BudgetResult | None = None

        self._trunc_cfg = TruncationConfig(
            tool_arg_max_chars=int(config.get("context.truncation.default.tool_arg_max_chars", 300)) if config is not None else 300,
            tool_result_max_chars=int(config.get("context.truncation.default.tool_result_max_chars", 500)) if config is not None else 500,
            summary_provider=(config.get("llm.summary_provider", "deepseek") if config is not None else "deepseek"),
            keep_last_units=int(config.get("context.truncation.default.keep_last_units", 3)) if config is not None else 3,
            keep_first_user_units=int(config.get("context.truncation.default.keep_first_user_units", 3)) if config is not None else 3,
        )

    def truncate(
        self,
        messages: list[ContextMessage],
        budget: BudgetResult,
    ) -> list[ContextMessage]:
        if (self._estimator is None) or (messages is None):
            raise ValueError("Effective estimator, messages, and total budget must be provided and non-zero")

        self._current_budget = budget
        available = budget.available_tokens

        initial_est = self._estimator.estimate(_to_llm_request(messages))
        current_tokens: int = initial_est["total"]

        self._logger.info(
            "Truncation check",
            total_tokens=current_tokens,
            available_tokens=available,
        )

        if current_tokens <= available:
            self._logger.info("No truncation needed, context within budget.")
            return messages

        tokens_before = current_tokens
        msgs_before = len(messages)

        def fits(t: int) -> bool:
            return t <= available

        def log_result(strategy: str, msgs_after: list[ContextMessage], tokens_after: int) -> None:
            ratio = (tokens_before - tokens_after) / tokens_before if tokens_before > 0 else 0.0
            self._logger.info(
                f"{strategy} resolved budget",
                msgs_before=msgs_before,
                msgs_after=len(msgs_after),
                msgs_dropped=msgs_before - len(msgs_after),
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                tokens_dropped=tokens_before - tokens_after,
                truncation_ratio=f"{ratio:.2%}",
            )

        # Strategy A: dedup
        msgs, delta = self._strategy_a_dedup(messages)
        current_tokens += delta
        if fits(current_tokens):
            log_result("Strategy A (dedup)", msgs, current_tokens)
            return msgs
        self._logger.info("Strategy A insufficient, trying B")

        # Strategy B: compress failed
        msgs, delta = self._strategy_b_compress_failed(msgs)
        current_tokens += delta
        if fits(current_tokens):
            log_result("Strategy B (compress failed)", msgs, current_tokens)
            return msgs
        self._logger.info("Strategy B insufficient, trying C/D")

        # Strategy C/D: trim args + results
        msgs, delta = self._strategy_c_trim_args(msgs)
        current_tokens += delta
        msgs, delta = self._strategy_d_trim_results(msgs)
        current_tokens += delta
        if fits(current_tokens):
            log_result("Strategy C/D (trim args/results)", msgs, current_tokens)
            return msgs
        self._logger.info("Strategy C/D insufficient, trying E")

        # Strategy E: binary drop
        dropped = self._strategy_e_binary_drop(msgs)
        if dropped is not None:
            msgs = dropped
            current_tokens = self._estimator.estimate(_to_llm_request(msgs))["total"]
            if fits(current_tokens):
                log_result("Strategy E (binary drop)", msgs, current_tokens)
                return msgs
        self._logger.info("Strategy E insufficient, trying F")

        # Strategy F: summarize candidates
        summarized = self._strategy_f_summarize(msgs)
        if summarized is not None:
            msgs = summarized
            current_tokens = self._estimator.estimate(_to_llm_request(msgs))["total"]
            if fits(current_tokens):
                log_result("Strategy F (summarize)", msgs, current_tokens)
                return msgs
            self._logger.info("Strategy F insufficient, looping E")

            # Loop E after F until fits or no more candidates
            while not fits(current_tokens):
                dropped = self._strategy_e_binary_drop(msgs)
                if dropped is None or dropped is msgs:
                    break
                msgs = dropped
                current_tokens = self._estimator.estimate(_to_llm_request(msgs))["total"]

        if not fits(current_tokens):
            ratio = (tokens_before - current_tokens) / tokens_before if tokens_before > 0 else 0.0
            self._logger.warning(
                "All truncation strategies exhausted but context is still over budget",
                msgs_before=msgs_before,
                msgs_after=len(msgs),
                tokens_before=tokens_before,
                tokens_after=current_tokens,
                truncation_ratio=f"{ratio:.2%}",
            )

        return msgs

    # ------------------------------------------------------------------
    # Strategy A: dedup consecutive identical tool blocks with placeholder
    # ------------------------------------------------------------------

    def _strategy_a_dedup(self, messages: list[ContextMessage]) -> tuple[list[ContextMessage], int]:
        units = parse_message_units(messages)
        tool_blocks = [u for u in units if isinstance(u, U_TOOL_BLOCK)]
        if len(tool_blocks) < 2:
            return messages, 0

        dup_ids: set[int] = set()
        for i in range(len(tool_blocks) - 1):
            if _tool_block_signature(tool_blocks[i]) == _tool_block_signature(tool_blocks[i + 1]):
                dup_ids.add(id(tool_blocks[i]))

        if not dup_ids:
            return messages, 0

        delta = 0
        result_units: list[Unit] = []
        for u in units:
            if id(u) in dup_ids and isinstance(u, U_TOOL_BLOCK):
                calls = u.assistant_msg.tool_use.all_calls()
                if len(calls) == 1:
                    args_summary = json.dumps(calls[0].tool_arguments, ensure_ascii=False)
                    if len(args_summary) > 60:
                        args_summary = args_summary[:60] + "..."
                    label = f"{calls[0].tool_name}({args_summary})"
                else:
                    label = ", ".join(c.tool_name for c in calls)
                placeholder = ContextMessage(
                    id=str(uuid4()),
                    role="user",
                    content=f"[重复调用已省略: {label}]",
                    token_count=None,
                )
                # Compute delta: tokens removed = original block tokens - placeholder tokens
                orig_tokens = self._estimator.estimate(_to_llm_request(u.to_messages()))["total"]
                ph_tokens = self._estimator.estimate(_to_llm_request([placeholder]))["total"]
                delta -= (orig_tokens - ph_tokens)
                result_units.append(U_USER(messages=[placeholder]))
            else:
                result_units.append(u)

        return units_to_messages(result_units), delta

    # ------------------------------------------------------------------
    # Strategy B: compress failed tool blocks to placeholder (candidates only)
    # ------------------------------------------------------------------

    def _strategy_b_compress_failed(self, messages: list[ContextMessage]) -> tuple[list[ContextMessage], int]:
        units = parse_message_units(messages)
        _, candidates, _ = self._get_candidate_units(units)
        if not candidates:
            return messages, 0

        failed_ids = {id(u) for u in candidates if isinstance(u, U_TOOL_BLOCK) and _has_failed_tool(u)}
        if not failed_ids:
            return messages, 0

        delta = 0
        result_units: list[Unit] = []
        for u in units:
            if id(u) in failed_ids and isinstance(u, U_TOOL_BLOCK):
                failed_name = next(
                    (m.tool_result.tool_name for m in u.tool_msgs
                     if m.tool_result is not None and not m.tool_result.success),
                    "unknown",
                )
                placeholder = ContextMessage(
                    id=str(uuid4()),
                    role="user",
                    content=f"[工具调用失败已压缩: {failed_name} - 错误已省略]",
                    token_count=None,
                )
                orig_tokens = self._estimator.estimate(_to_llm_request(u.to_messages()))["total"]
                ph_tokens = self._estimator.estimate(_to_llm_request([placeholder]))["total"]
                delta -= (orig_tokens - ph_tokens)
                result_units.append(U_USER(messages=[placeholder]))
            else:
                result_units.append(u)

        return units_to_messages(result_units), delta

    # ------------------------------------------------------------------
    # Strategy C: trim oversized tool call arguments (candidates only)
    # ------------------------------------------------------------------

    def _strategy_c_trim_args(self, messages: list[ContextMessage]) -> tuple[list[ContextMessage], int]:
        units = parse_message_units(messages)
        _, candidates, _ = self._get_candidate_units(units)
        if not candidates:
            return messages, 0

        candidate_ids = {id(u) for u in candidates}
        limit = self._trunc_cfg.tool_arg_max_chars

        result_units: list[Unit] = []
        any_changed = False
        for unit in units:
            if id(unit) not in candidate_ids or not isinstance(unit, U_TOOL_BLOCK):
                result_units.append(unit)
                continue
            changed = False
            new_entries: list[ToolCallEntry] = []
            for entry in unit.assistant_msg.tool_use.all_calls():
                new_args = {}
                for k, v in entry.tool_arguments.items():
                    if isinstance(v, str) and len(v) > limit:
                        new_args[k] = _truncate_string_arg(v, limit)
                        changed = True
                    else:
                        new_args[k] = v
                new_entries.append(ToolCallEntry(
                    tool_call_id=entry.tool_call_id,
                    tool_name=entry.tool_name,
                    tool_arguments=new_args,
                ))
            if changed:
                any_changed = True
                primary = new_entries[0]
                new_tool_use = ToolUseMetadata(
                    tool_call_id=primary.tool_call_id,
                    tool_name=primary.tool_name,
                    tool_arguments=primary.tool_arguments,
                    extra_calls=tuple(new_entries[1:]),
                )
                new_assistant = ContextMessage(
                    id=str(uuid4()),
                    role=unit.assistant_msg.role,
                    content=unit.assistant_msg.content,
                    token_count=unit.assistant_msg.token_count,
                    tool_use=new_tool_use,
                )
                result_units.append(U_TOOL_BLOCK(
                    assistant_msg=new_assistant,
                    tool_msgs=unit.tool_msgs,
                    trailing_msgs=unit.trailing_msgs,
                ))
            else:
                result_units.append(unit)

        if not any_changed:
            return messages, 0

        result_msgs = units_to_messages(result_units)
        before_tokens = self._estimator.estimate(_to_llm_request(messages))["total"]
        after_tokens = self._estimator.estimate(_to_llm_request(result_msgs))["total"]
        return result_msgs, after_tokens - before_tokens

    # ------------------------------------------------------------------
    # Strategy D: trim oversized tool results (candidates only)
    # ------------------------------------------------------------------

    def _strategy_d_trim_results(self, messages: list[ContextMessage]) -> tuple[list[ContextMessage], int]:
        units = parse_message_units(messages)
        _, candidates, _ = self._get_candidate_units(units)
        if not candidates:
            return messages, 0

        candidate_ids = {id(u) for u in candidates}
        limit = self._trunc_cfg.tool_result_max_chars

        result_units: list[Unit] = []
        any_changed = False
        for unit in units:
            if id(unit) not in candidate_ids or not isinstance(unit, U_TOOL_BLOCK):
                result_units.append(unit)
                continue
            new_tool_msgs = []
            changed = False
            for msg in unit.tool_msgs:
                if len(msg.content) > limit:
                    new_content = _truncate_tool_result_content(msg.content, limit)
                    changed = True
                    new_tool_msgs.append(ContextMessage(
                        id=str(uuid4()),
                        role=msg.role,
                        content=new_content,
                        token_count=msg.token_count,
                        tool_result=msg.tool_result,
                    ))
                else:
                    new_tool_msgs.append(msg)
            if changed:
                any_changed = True
            result_units.append(U_TOOL_BLOCK(
                assistant_msg=unit.assistant_msg,
                tool_msgs=new_tool_msgs,
                trailing_msgs=unit.trailing_msgs,
            ))

        if not any_changed:
            return messages, 0

        result_msgs = units_to_messages(result_units)
        before_tokens = self._estimator.estimate(_to_llm_request(messages))["total"]
        after_tokens = self._estimator.estimate(_to_llm_request(result_msgs))["total"]
        return result_msgs, after_tokens - before_tokens

    # ------------------------------------------------------------------
    # Strategy E: binary-search minimum drop of candidate units
    # ------------------------------------------------------------------

    def _strategy_e_binary_drop(
        self,
        messages: list[ContextMessage],
    ) -> list[ContextMessage] | None:
        units = parse_message_units(messages)
        _, candidates, _ = self._get_candidate_units(units)
        if not candidates:
            return None

        available = self._current_budget.available_tokens if self._current_budget else 0

        def fits_msgs(msgs: list[ContextMessage]) -> bool:
            return self._estimator.estimate(_to_llm_request(msgs))["total"] <= available

        lo, hi = 1, len(candidates)
        best: list[ContextMessage] | None = None

        while lo <= hi:
            k = (lo + hi) // 2
            drop_ids = {id(u) for u in candidates[:k]}
            candidate = units_to_messages([u for u in units if id(u) not in drop_ids])
            if fits_msgs(candidate):
                best = candidate
                hi = k - 1
            else:
                lo = k + 1

        return best

    # ------------------------------------------------------------------
    # Strategy F: LLM summary of all candidate units
    # ------------------------------------------------------------------

    def _strategy_f_summarize(
        self,
        messages: list[ContextMessage],
    ) -> list[ContextMessage] | None:
        units = parse_message_units(messages)
        _, candidates, _ = self._get_candidate_units(units)
        if not candidates:
            return None

        candidate_ids = {id(u) for u in candidates}
        summary_msgs = [m for u in units if id(u) in candidate_ids for m in u.to_messages()]
        summary_msg = self._call_summary_llm(summary_msgs)
        if summary_msg is None:
            return None

        result_units: list[Unit] = []
        inserted = False
        for u in units:
            if id(u) in candidate_ids:
                if not inserted:
                    result_units.append(U_USER(messages=[summary_msg]))
                    inserted = True
            else:
                result_units.append(u)
        return units_to_messages(result_units)

    @staticmethod
    def _serialize_message_for_summary(m: ContextMessage) -> str:
        if m.tool_use is not None:
            calls = m.tool_use.all_calls()
            call_strs = [
                f"{e.tool_name}({json.dumps(e.tool_arguments, ensure_ascii=False)})"
                for e in calls
            ]
            calls_text = ", ".join(call_strs)
            content_part = f" | {m.content}" if m.content else ""
            return f"[assistant:tool_call] {calls_text}{content_part}"
        if m.tool_result is not None:
            status = "success" if m.tool_result.success else "failed"
            return f"[tool:{m.tool_result.tool_name}:{status}] {m.content}"
        return f"[{m.role}] {m.content}"

    def _call_summary_llm(
        self,
        msgs_to_summarize: list[ContextMessage],
    ) -> ContextMessage | None:
        try:
            if self._config is None:
                self._logger.error("Strategy F: no config available, cannot build summary LLM client")
                return None
            summary_provider = self._config.get("llm.summary_provider", self._trunc_cfg.summary_provider)
            history_text = "\n".join(
                self._serialize_message_for_summary(m) for m in msgs_to_summarize
            )
            summary_request = UnifiedLLMRequest(
                system_prompt=_SUMMARY_SYSTEM_PROMPT,
                messages=[LLMMessage(role="user", content=history_text)],
                tool_schemas=[],
            )
            response = self._llm_gateway.generate(summary_request, summary_provider)
            self._logger.info("Strategy F: summary LLM response", content=response.assistant_message.content)
            return ContextMessage(
                id=str(uuid4()),
                role="user",
                content=response.assistant_message.content,
                summary=SummaryMetadata(
                    stage_index=-1,
                    original_message_count=len(msgs_to_summarize),
                ),
            )
        except Exception as exc:
            self._logger.error("Strategy F: summary LLM call failed", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_candidate_units(
        self, units: list[Unit]
    ) -> tuple[list[Unit], list[Unit], list[Unit]]:
        """
        Split units into (head, candidates, tail).

        Head: all leading U_SYS units + first keep_first_user_units U_USER units.
        Tail: units from the back consuming up to 70% of available_tokens.
        Candidates: everything between head and tail — eligible for truncation.

        Falls back to config-based keep_last_units when budget is unavailable.
        """
        available = self._current_budget.available_tokens if self._current_budget else None

        # Head: leading U_SYS + first keep_first_user_units U_USER units
        head_end = 0
        user_count = 0
        kfu = self._trunc_cfg.keep_first_user_units
        for i, u in enumerate(units):
            if isinstance(u, U_SYS):
                head_end = i + 1
            elif user_count < kfu:
                if isinstance(u, U_USER):
                    user_count += 1
                head_end = i + 1
            else:
                break

        if available is None:
            kl = self._trunc_cfg.keep_last_units
            tail_start = max(len(units) - kl, head_end)
        else:
            tail_budget = int(available * 0.70)
            accumulated = 0
            tail_start = len(units)
            for i in range(len(units) - 1, head_end - 1, -1):
                unit_tokens = self._estimator.estimate(
                    _to_llm_request(units[i].to_messages())
                )["total"]
                if accumulated + unit_tokens > tail_budget:
                    break
                accumulated += unit_tokens
                tail_start = i

        return units[:head_end], units[head_end:tail_start], units[tail_start:]

class TruncatorFactory:
    @classmethod
    def create(
        cls,
        strategy: str,
        logger: Logger,
        config: ConfigReader,
        tracer: Tracer,
        llm_gateway: LLMGateway,
        estimator: BaseTokenEstimator,
    ) -> ContextTruncator:
        if strategy == "default":
            return DefaultContextTruncator(logger=logger, config=config, tracer=tracer, llm_gateway=llm_gateway, estimator=estimator)
        raise ValueError(f"Unknown truncation strategy: {strategy!r}")

