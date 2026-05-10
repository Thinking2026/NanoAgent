from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
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
from schemas.errors import TRUNCATION_FAILED, build_logic_error
from schemas.types import BudgetResult, LLMMessage, UnifiedLLMRequest
from utils.log.log import Logger

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
class U_ASSISTANT:
    """A standalone plain assistant message (no tool_use)."""
    msg: ContextMessage

    def to_messages(self) -> list[ContextMessage]:
        return [self.msg]


@dataclass
class U_TOOL_BLOCK:
    """One assistant tool-call + all parallel tool results."""
    assistant_msg: ContextMessage
    tool_msgs: list[ContextMessage] = field(default_factory=list)

    def to_messages(self) -> list[ContextMessage]:
        return [self.assistant_msg, *self.tool_msgs]


Unit = U_SYS | U_USER | U_ASSISTANT | U_TOOL_BLOCK


@dataclass
class UnitGroup:
    """
    从一个 U_USER 消息开始到下一个 U_USER 消息之前（左闭右开）的连续 unit 序列。
    最新的 group 可能未完全闭合（尚无下一个 U_USER）。
    """
    units: list[Unit] = field(default_factory=list)

    def to_messages(self) -> list[ContextMessage]:
        return [m for u in self.units for m in u.to_messages()]


def parse_message_units(messages: list[ContextMessage]) -> list[Unit]:
    """
    Partition a flat message list into structured units:
      - U_SYS       : the leading system message (at most one, at index 0)
      - U_USER      : a single user message (standalone, no trailing merging)
      - U_ASSISTANT : a standalone plain assistant message (no tool_use)
      - U_TOOL_BLOCK: one assistant tool-call + its matching tool results

    Each message maps to exactly one unit type; no trailing merging occurs.
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
            units.append(U_ASSISTANT(msg=msg))
            i += 1

        else:
            i += 1

    return units


def units_to_messages(units: list[Unit]) -> list[ContextMessage]:
    return [m for u in units for m in u.to_messages()]


def parse_unit_groups(units: list[Unit]) -> list[UnitGroup]:
    """
    Partition units into UnitGroups.

    Each U_USER starts a new group. Units before the first U_USER (U_SYS,
    leading U_ASSISTANT) are collected into a preamble group. The last group
    may be unclosed (no following U_USER to terminate it).

    Raises ValueError if any group contains messages from multiple stages.
    """
    groups: list[UnitGroup] = []
    preamble = UnitGroup()
    first_user_seen = False

    for unit in units:
        if isinstance(unit, U_USER):
            first_user_seen = True
            if preamble.units:
                groups.append(preamble)
                preamble = UnitGroup()
            groups.append(UnitGroup(units=[unit]))
        elif not first_user_seen:
            preamble.units.append(unit)
        else:
            groups[-1].units.append(unit)

    if preamble.units:
        groups.append(preamble)

    return groups

# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TruncationConfig:
    tool_arg_max_chars: int = 300
    tool_result_max_chars: int = 500
    summary_provider: str = "deepseek"
    keep_last_units: int = 3
    keep_first_user_units: int = 1


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

        # Strategy A: dedup consecutive identical tool blocks
        msgs, delta = self._strategy_a_dedup(messages)
        current_tokens += delta
        if fits(current_tokens):
            log_result("Strategy A (dedup)", msgs, current_tokens)
            return msgs
        self._logger.info("Strategy A insufficient, trying B")

        # Strategy B: simplify failed tool result content
        msgs, delta = self._strategy_b_compress_failed(msgs)
        current_tokens += delta
        if fits(current_tokens):
            log_result("Strategy B (compress failed)", msgs, current_tokens)
            return msgs
        self._logger.info("Strategy B insufficient, trying C")

        # Strategy C: trim oversized tool call arguments
        msgs, delta = self._strategy_c_trim_args(msgs)
        current_tokens += delta
        if fits(current_tokens):
            log_result("Strategy C (trim args)", msgs, current_tokens)
            return msgs
        self._logger.info("Strategy C insufficient, trying D")

        # Strategy D: trim oversized tool results
        msgs, delta = self._strategy_d_trim_results(msgs)
        current_tokens += delta
        if fits(current_tokens):
            log_result("Strategy D (trim results)", msgs, current_tokens)
            return msgs
        self._logger.info("Strategy D insufficient, trying F")

        # Strategy F: summarize all candidate units into one assistant message
        summarized = self._strategy_f_summarize(msgs)
        if summarized is not None:
            msgs = summarized
            current_tokens = self._estimator.estimate(_to_llm_request(msgs))["total"]
            if fits(current_tokens):
                log_result("Strategy F (summarize)", msgs, current_tokens)
                return msgs
        self._logger.info("Strategy F insufficient, trying fallback")

        # Fallback: keep sys + first group + last group, summarize middle
        fallback = self._fallback_summarize(msgs)
        if fallback is not None:
            msgs = fallback
            current_tokens = self._estimator.estimate(_to_llm_request(msgs))["total"]
            if fits(current_tokens):
                log_result("Fallback (summarize middle groups)", msgs, current_tokens)
                return msgs

        raise build_logic_error(TRUNCATION_FAILED, "无法裁剪: 上下文超出预算且无法进一步压缩")

    # ------------------------------------------------------------------
    # Strategy A: dedup consecutive identical tool blocks (all unit groups)
    # ------------------------------------------------------------------

    def _strategy_a_dedup(self, messages: list[ContextMessage]) -> tuple[list[ContextMessage], int]:
        units = parse_message_units(messages)

        # Find runs of consecutive U_TOOL_BLOCK units with identical signatures.
        # "Consecutive" means adjacent in the units list with no intervening units.
        # For each run of length >= 2, drop the first N-1 and annotate the last.
        to_drop: set[int] = set()       # indices in units[] to drop entirely
        annotation: dict[int, int] = {} # units[] index -> N (number of dropped predecessors)

        i = 0
        while i < len(units):
            if not isinstance(units[i], U_TOOL_BLOCK):
                i += 1
                continue
            sig = _tool_block_signature(units[i])
            j = i + 1
            while j < len(units) and isinstance(units[j], U_TOOL_BLOCK) and _tool_block_signature(units[j]) == sig:
                j += 1
            run_length = j - i
            if run_length >= 2:
                for k in range(i, j - 1):
                    to_drop.add(k)
                annotation[j - 1] = run_length - 1
            i = j

        if not to_drop:
            return messages, 0

        delta = 0
        result_units: list[Unit] = []
        for idx, u in enumerate(units):
            if idx in to_drop:
                orig_tokens = self._estimator.estimate(_to_llm_request(u.to_messages()))["total"]
                delta -= orig_tokens
                continue
            if idx in annotation and isinstance(u, U_TOOL_BLOCK):
                n = annotation[idx]
                note = f"（前面经过{n}轮相同的tool调用，已简化成一个）"
                new_content = (u.assistant_msg.content or "") + note
                new_assistant = ContextMessage(
                    id=str(uuid4()),
                    role=u.assistant_msg.role,
                    content=new_content,
                    token_count=u.assistant_msg.token_count,
                    tool_use=u.assistant_msg.tool_use,
                    summary=u.assistant_msg.summary,
                )
                orig_tokens = self._estimator.estimate(_to_llm_request([u.assistant_msg]))["total"]
                note_tokens = self._estimator.estimate(_to_llm_request([new_assistant]))["total"]
                delta += note_tokens - orig_tokens
                result_units.append(U_TOOL_BLOCK(
                    assistant_msg=new_assistant,
                    tool_msgs=u.tool_msgs,
                ))
            else:
                result_units.append(u)

        return units_to_messages(result_units), delta

    # ------------------------------------------------------------------
    # Strategy B: simplify failed tool result content (all unit groups)
    # ------------------------------------------------------------------

    def _strategy_b_compress_failed(self, messages: list[ContextMessage]) -> tuple[list[ContextMessage], int]:
        units = parse_message_units(messages)

        delta = 0
        result_units: list[Unit] = []
        any_changed = False

        for u in units:
            if not isinstance(u, U_TOOL_BLOCK) or not _has_failed_tool(u):
                result_units.append(u)
                continue

            new_tool_msgs = []
            changed = False
            for msg in u.tool_msgs:
                if msg.tool_result is not None and not msg.tool_result.success:
                    tool_name = msg.tool_result.tool_name or "unknown"
                    new_content = f"[工具调用失败: {tool_name}]"
                    orig_tokens = self._estimator.estimate(_to_llm_request([msg]))["total"]
                    new_msg = ContextMessage(
                        id=str(uuid4()),
                        role=msg.role,
                        content=new_content,
                        token_count=msg.token_count,
                        tool_result=msg.tool_result,
                    )
                    new_tokens = self._estimator.estimate(_to_llm_request([new_msg]))["total"]
                    delta += new_tokens - orig_tokens
                    new_tool_msgs.append(new_msg)
                    changed = True
                else:
                    new_tool_msgs.append(msg)

            if changed:
                any_changed = True
                result_units.append(U_TOOL_BLOCK(
                    assistant_msg=u.assistant_msg,
                    tool_msgs=new_tool_msgs,
                ))
            else:
                result_units.append(u)

        if not any_changed:
            return messages, 0

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
            ))

        if not any_changed:
            return messages, 0

        result_msgs = units_to_messages(result_units)
        before_tokens = self._estimator.estimate(_to_llm_request(messages))["total"]
        after_tokens = self._estimator.estimate(_to_llm_request(result_msgs))["total"]
        return result_msgs, after_tokens - before_tokens

    # ------------------------------------------------------------------
    # Strategy F: LLM summary of all candidate units → one assistant message
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
        if not summary_msgs:
            return None

        summary_msg = self._call_summary_llm(summary_msgs)
        if summary_msg is None:
            return None

        # Replace all candidate units with a single U_ASSISTANT summary message.
        # summary_msg.summary is not None — marks it as a summarized message.
        result_units: list[Unit] = []
        inserted = False
        for u in units:
            if id(u) in candidate_ids:
                if not inserted:
                    result_units.append(U_ASSISTANT(msg=summary_msg))
                    inserted = True
            else:
                result_units.append(u)
        return units_to_messages(result_units)

    # ------------------------------------------------------------------
    # Fallback: keep sys + first group + last group, summarize middle groups
    # ------------------------------------------------------------------

    def _fallback_summarize(
        self,
        messages: list[ContextMessage],
    ) -> list[ContextMessage] | None:
        """
        Keep system units + first UnitGroup + most recent UnitGroup (may be
        unclosed). Summarize all middle groups into one assistant message.
        Requires at least 3 groups; returns None if there is nothing to summarize.
        """
        units = parse_message_units(messages)
        sys_units: list[Unit] = [u for u in units if isinstance(u, U_SYS)]
        non_sys_units: list[Unit] = [u for u in units if not isinstance(u, U_SYS)]

        groups = parse_unit_groups(non_sys_units)

        if len(groups) <= 2:
            return None

        first_group = groups[0]
        last_group = groups[-1]   # may be unclosed — always preserved in full
        middle_groups = groups[1:-1]

        middle_msgs = [m for g in middle_groups for m in g.to_messages()]
        if not middle_msgs:
            return None

        summary_msg = self._call_summary_llm(middle_msgs)
        if summary_msg is None:
            return None

        result_units: list[Unit] = (
            list(sys_units)
            + list(first_group.units)
            + [U_ASSISTANT(msg=summary_msg)]
            + list(last_group.units)
        )
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
                role="assistant",
                content=response.assistant_message.content,
                summary=SummaryMetadata(
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

        Head  : all leading U_SYS units + the first UnitGroup (starting from
                the first U_USER).
        Tail  : the most recent UnitGroup (groups[-1]), always preserved in full
                because it may be unclosed (no following U_USER yet).
        Candidates: UnitGroups between head and tail whose combined token cost
                fits within usable_budget = (available - head_tokens - tail_tokens) * 0.70.
                Groups that don't fit are pushed into the tail.

        Returns flat unit lists for each partition.
        """
        available = self._current_budget.available_tokens if self._current_budget else 0

        sys_units: list[Unit] = [u for u in units if isinstance(u, U_SYS)]
        non_sys_units: list[Unit] = [u for u in units if not isinstance(u, U_SYS)]

        groups = parse_unit_groups(non_sys_units)

        if not groups:
            return list(sys_units), [], []

        if len(groups) == 1:
            # Only one group — it is both head and tail; nothing to truncate
            return list(sys_units) + list(groups[0].units), [], []

        # head = sys + first group; tail = last group (may be unclosed)
        head_units: list[Unit] = list(sys_units) + list(groups[0].units)
        tail_units: list[Unit] = list(groups[-1].units)
        middle_groups: list[UnitGroup] = groups[1:-1]

        if not middle_groups:
            # Only two groups — head and tail, nothing in between
            return head_units, [], tail_units

        # Compute fixed costs
        head_tokens = self._estimator.estimate(_to_llm_request(
            [m for u in head_units for m in u.to_messages()]
        ))["total"]
        tail_tokens = self._estimator.estimate(_to_llm_request(
            [m for u in tail_units for m in u.to_messages()]
        ))["total"]

        remaining = max(0, available - head_tokens - tail_tokens)
        usable_budget = int(remaining * 0.70)

        # Walk backwards through middle_groups to find which ones become tail
        accumulated = 0
        candidate_end = len(middle_groups)  # exclusive upper bound for candidates

        for i in range(len(middle_groups) - 1, -1, -1):
            group_tokens = self._estimator.estimate(
                _to_llm_request(middle_groups[i].to_messages())
            )["total"]
            if accumulated + group_tokens <= usable_budget:
                accumulated += group_tokens
                candidate_end = i  # this group fits in tail budget, push boundary left
            else:
                break

        # middle_groups[:candidate_end] → candidates
        # middle_groups[candidate_end:] → extra tail (prepended before groups[-1])
        candidate_units: list[Unit] = [u for g in middle_groups[:candidate_end] for u in g.units]
        extra_tail_units: list[Unit] = [u for g in middle_groups[candidate_end:] for u in g.units]

        return head_units, candidate_units, extra_tail_units + tail_units

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

