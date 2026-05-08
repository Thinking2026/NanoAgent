from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable
from uuid import uuid4

from agent.models.context.budget.token_budget_manager import BaseTokenBudgetManager
from agent.models.context.estimator.token_estimator import BaseTokenEstimator
from agent.models.context.context_manager import ContextMessage
from llm.llm_gateway import LLMGateway
from schemas.types import LLMMessage, UnifiedLLMRequest
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
        total_budget: int,
        estimator: BaseTokenEstimator,
    ) -> list[ContextMessage]:
        ...

# ===========================================================================
# ReAct-aware truncation
# ===========================================================================

@dataclass
class ReasoningUnit:
    assistant_msg: ContextMessage
    tool_msgs: list[ContextMessage] = field(default_factory=list)


@dataclass(frozen=True)
class ReActTruncationConfig:
    tool_arg_max_chars: int = 300
    tool_result_max_chars: int = 500
    summary_provider: str = "deepseek"
    keep_first_units: int = 1
    keep_last_units: int = 3
    summary_ratio: float = 0.20


def _parse_reasoning_units(messages: list[ContextMessage]) -> list[ReasoningUnit]:
    """Group assistant+tool message sequences into ReasoningUnits."""
    units: list[ReasoningUnit] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.role == "assistant" and msg.metadata.get("tool_calls"):
            tool_call_ids: set[str] = {
                tc["llm_raw_tool_call_id"]
                for tc in msg.metadata["tool_calls"]
                if tc.get("llm_raw_tool_call_id")
            }
            unit = ReasoningUnit(assistant_msg=msg)
            i += 1
            while i < len(messages) and messages[i].role == "tool":
                tid = messages[i].metadata.get("llm_raw_tool_call_id")
                if tid in tool_call_ids:
                    unit.tool_msgs.append(messages[i])
                    i += 1
                else:
                    break
            units.append(unit)
        else:
            i += 1
    return units


def _unit_to_messages(unit: ReasoningUnit) -> list[ContextMessage]:
    return [unit.assistant_msg, *unit.tool_msgs]


def _unit_tool_signature(unit: ReasoningUnit) -> str:
    """Stable string key for dedup: tool names + sorted arguments."""
    calls = unit.assistant_msg.metadata.get("tool_calls", [])
    parts = []
    for tc in calls:
        parts.append(f"{tc.get('name')}:{json.dumps(tc.get('arguments', {}), sort_keys=True)}")
    return "|".join(parts)


def _has_failed_tool(unit: ReasoningUnit) -> bool:
    for msg in unit.tool_msgs:
        if msg.metadata.get("success") is False:
            return True
    return False


def _to_llm_request(messages: list[ContextMessage]) -> UnifiedLLMRequest:
    """Convert ContextMessages to a bare UnifiedLLMRequest for token estimation."""
    return UnifiedLLMRequest(
        messages=[
            LLMMessage(role=m.role, content=m.content, metadata=m.metadata)
            for m in messages
        ]
    )


class ReActContextTruncator(ContextTruncator):
    def __init__(
        self,
        budget_manager: BaseTokenBudgetManager,
        logger: Logger,
        config: ConfigReader,
        llm_gateway: LLMGateway,
    ) -> None:
        self._budget_manager = budget_manager
        self._logger = logger
        self._llm_gateway = llm_gateway

        trunc_cfg = ReActTruncationConfig(
            tool_arg_max_chars=int(config.get("context.truncation.react.tool_arg_max_chars", 300)) if config is not None else 300,
            tool_result_max_chars=int(config.get("context.truncation.react.tool_result_max_chars", 500)) if config is not None else 500,
            summary_provider=(
                config.get("llm.summary_provider", "deepseek")
                if config is not None else "deepseek"
            ),
            keep_first_units=int(config.get("context.truncation.react.keep_first_units", 1)) if config is not None else 1,
            keep_last_units=int(config.get("context.truncation.react.keep_last_units", 3)) if config is not None else 3,
            summary_ratio=float(config.get("context.truncation.react.summary_ratio", 0.20)) if config is not None else 0.20,
        )
        self._trunc_cfg = trunc_cfg or ReActTruncationConfig()
        self._config = config

    #TODO 实现不同严重等级的裁剪策略，LLM API返回需要裁剪的降级时需要触发，现在只有一个模型降级
    def truncate(
        self,
        messages: list[ContextMessage],
        total_budget: int,
        effective_estimator: BaseTokenEstimator,
    ) -> list[ContextMessage]:
        if (effective_estimator is None) or (messages is None) or (0 == total_budget):
            raise ValueError("Effective estimator, messages, and total budget must be provided and non-zero")

        budget = self._budget_manager.allocate(total_budget)
        estimation = effective_estimator.estimate(_to_llm_request(messages), ["assistant", "tool"])

        assistant_budget = budget.role_budgets["assistant"].token_budget
        tool_budget = budget.role_budgets["tool"].token_budget

        self._logger.info(
            "Truncation check",
            assistant_tokens=estimation["assistant"],
            assistant_budget=assistant_budget,
            tool_tokens=estimation["tool"],
            tool_budget=tool_budget,
        )

        tokens_before = estimation["assistant"] + estimation["tool"]
        msgs_before = len(messages)

        if estimation["assistant"] <= assistant_budget and estimation["tool"] <= tool_budget:
            self._logger.info("No truncation needed, context within budget.")
            return messages

        fits = self._make_fits_fn(budget, effective_estimator)

        def _log_truncation_result(strategy: str, msgs_after: list[ContextMessage]) -> None:
            est_after = effective_estimator.estimate(_to_llm_request(msgs_after), ["assistant", "tool"])
            tokens_after = est_after["assistant"] + est_after["tool"]
            ratio = ((tokens_before - tokens_after) / tokens_before) if tokens_before > 0 else 0.0
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

        msgs = self._strategy_a_dedup(messages)
        if fits(msgs):
            _log_truncation_result("Strategy A (dedup)", msgs)
            return msgs
        self._logger.info("Strategy A insufficient, trying B")

        msgs = self._strategy_b_remove_failed(msgs)
        if fits(msgs):
            _log_truncation_result("Strategy B (remove failed)", msgs)
            return msgs
        self._logger.info("Strategy B insufficient, trying C/D")

        est = effective_estimator.estimate(_to_llm_request(msgs), ["assistant", "tool"])
        if est["assistant"] > assistant_budget:
            msgs = self._strategy_c_trim_args(msgs)
        if est["tool"] > tool_budget:
            msgs = self._strategy_d_trim_results(msgs)
        if fits(msgs):
            _log_truncation_result("Strategy C/D (trim args/results)", msgs)
            return msgs
        self._logger.info("Strategy C/D insufficient, trying E")

        if dropped := self._strategy_e_binary_drop(msgs, fits):
            if fits(dropped):
                _log_truncation_result("Strategy E (binary drop)", dropped)
                return dropped
        self._logger.info("Strategy E insufficient, trying F")

        if summarized := self._strategy_f_summarize(msgs):
            msgs = summarized
            if fits(msgs):
                _log_truncation_result("Strategy F (summarize)", msgs)
                return msgs
        self._logger.info("Strategy F insufficient, retrying E after summarize")

        if dropped := self._strategy_e_binary_drop(msgs, fits):
            msgs = dropped
            if not fits(msgs):
                est_after = effective_estimator.estimate(_to_llm_request(msgs), ["assistant", "tool"])
                tokens_after = est_after["assistant"] + est_after["tool"]
                ratio = (tokens_before - tokens_after) / tokens_before if tokens_before > 0 else 0.0
                self._logger.warning(
                    "All truncation strategies exhausted but context is still over budget",
                    msgs_before=msgs_before,
                    msgs_after=len(msgs),
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                    truncation_ratio=f"{ratio:.2%}",
                )

        return msgs

    # ------------------------------------------------------------------
    # Strategy A: remove consecutive duplicate reasoning units
    # ------------------------------------------------------------------

    def _strategy_a_dedup(self, messages: list[ContextMessage]) -> list[ContextMessage]:
        units = _parse_reasoning_units(messages)
        if len(units) < 2:
            return messages

        drop_units: set[int] = set()
        for i in range(len(units) - 1):
            if _unit_tool_signature(units[i]) == _unit_tool_signature(units[i + 1]):
                drop_units.add(i)

        if not drop_units:
            return messages

        keep_msgs = {id(m) for i, u in enumerate(units) if i not in drop_units for m in _unit_to_messages(u)}
        unit_msg_ids = {id(m) for u in units for m in _unit_to_messages(u)}
        return [m for m in messages if id(m) not in unit_msg_ids or id(m) in keep_msgs]

    # ------------------------------------------------------------------
    # Strategy B: remove failed reasoning units (middle only)
    # ------------------------------------------------------------------

    def _strategy_b_remove_failed(self, messages: list[ContextMessage]) -> list[ContextMessage]:
        units = _parse_reasoning_units(messages)
        _, middle_units, _ = self._get_middle_units(units)
        if not middle_units:
            return messages
        failed_ids = {id(m) for u in middle_units if _has_failed_tool(u) for m in _unit_to_messages(u)}
        if not failed_ids:
            return messages
        return [m for m in messages if id(m) not in failed_ids]

    # ------------------------------------------------------------------
    # Strategy C: trim oversized tool call arguments (middle only)
    # ------------------------------------------------------------------

    def _strategy_c_trim_args(self, messages: list[ContextMessage]) -> list[ContextMessage]:
        units = _parse_reasoning_units(messages)
        _, middle_units, _ = self._get_middle_units(units)
        if not middle_units:
            return messages
        middle_ids = {id(m) for u in middle_units for m in _unit_to_messages(u)}

        result: list[ContextMessage] = []
        limit = self._trunc_cfg.tool_arg_max_chars
        for msg in messages:
            if id(msg) not in middle_ids or msg.role != "assistant" or not msg.metadata.get("tool_calls"):
                result.append(msg)
                continue
            new_calls = []
            changed = False
            for tc in msg.metadata["tool_calls"]:
                args = tc.get("arguments", {})
                new_args = {}
                for k, v in args.items():
                    if isinstance(v, str) and len(v) > limit:
                        new_args[k] = v[:limit] + "(trimmed because too long)"
                        changed = True
                    else:
                        new_args[k] = v
                new_calls.append({**tc, "arguments": new_args})
            if changed:
                result.append(ContextMessage(
                    id=str(uuid4()),
                    role=msg.role,
                    content=msg.content,
                    metadata={**msg.metadata, "tool_calls": new_calls},
                ))
            else:
                result.append(msg)
        return result

    # ------------------------------------------------------------------
    # Strategy D: trim oversized tool results (middle only)
    # ------------------------------------------------------------------

    def _strategy_d_trim_results(self, messages: list[ContextMessage]) -> list[ContextMessage]:
        units = _parse_reasoning_units(messages)
        _, middle_units, _ = self._get_middle_units(units)
        if not middle_units:
            return messages
        middle_ids = {id(m) for u in middle_units for m in _unit_to_messages(u)}

        result: list[ContextMessage] = []
        limit = self._trunc_cfg.tool_result_max_chars
        for msg in messages:
            if id(msg) in middle_ids and msg.role == "tool" and len(msg.content) > limit:
                result.append(ContextMessage(
                    id=str(uuid4()),
                    role=msg.role,
                    content=msg.content[:limit] + "(trimmed because too long)",
                    metadata=msg.metadata,
                ))
            else:
                result.append(msg)
        return result

    # ------------------------------------------------------------------
    # Strategy E: binary-search minimum drop of middle units
    # ------------------------------------------------------------------

    def _strategy_e_binary_drop(
        self,
        messages: list[ContextMessage],
        fits: Callable[[list[ContextMessage]], bool],
    ) -> list[ContextMessage] | None:
        units = _parse_reasoning_units(messages)
        _, middle_units, _ = self._get_middle_units(units)
        if not middle_units:
            return None

        lo, hi = 1, len(middle_units)
        best: list[ContextMessage] | None = None

        while lo <= hi:
            k = (lo + hi) // 2
            candidate = self._drop_oldest_k(messages, units, middle_units, k)
            if fits(candidate):
                best = candidate
                hi = k - 1
            else:
                lo = k + 1

        return best

    def _drop_oldest_k(
        self,
        messages: list[ContextMessage],
        all_units: list[ReasoningUnit],
        middle_units: list[ReasoningUnit],
        k: int,
    ) -> list[ContextMessage]:
        drop_ids = {id(m) for u in middle_units[:k] for m in _unit_to_messages(u)}
        unit_ids = {id(m) for u in all_units for m in _unit_to_messages(u)}
        return [m for m in messages if id(m) not in unit_ids or id(m) not in drop_ids]

    # ------------------------------------------------------------------
    # Strategy F: LLM summary of oldest summary_ratio fraction of middle units
    # ------------------------------------------------------------------

    def _strategy_f_summarize(
        self,
        messages: list[ContextMessage],
    ) -> list[ContextMessage] | None:
        units = _parse_reasoning_units(messages)
        _, middle_units, _ = self._get_middle_units(units)
        if not middle_units:
            return None

        n_to_summarize = max(1, int(len(middle_units) * self._trunc_cfg.summary_ratio))
        summary_units = middle_units[:n_to_summarize]

        summary_msgs = [m for u in summary_units for m in _unit_to_messages(u)]
        summary_msg = self._call_summary_llm(summary_msgs)
        if summary_msg is None:
            return None

        summary_ids = {id(m) for m in summary_msgs}
        result: list[ContextMessage] = []
        inserted = False
        for m in messages:
            if id(m) in summary_ids:
                if not inserted:
                    result.append(summary_msg)
                    inserted = True
            else:
                result.append(m)
        return result

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
                f"[{m.role}] {m.content}" for m in msgs_to_summarize
            )
            summary_request = UnifiedLLMRequest(
                system_prompt=(
                    "You are a context compressor. Summarize the following reasoning steps "
                    "into a single concise assistant message. Preserve key facts, tool results, "
                    "and conclusions. Output only the summary text, no preamble."
                ),
                messages=[LLMMessage(role="user", content=history_text)],
                tool_schemas=[],
            )
            response = self._llm_gateway.generate(summary_request, summary_provider)
            self._logger.info("Strategy F: summary LLM response", content=response.assistant_message.content)
            return ContextMessage(
                id=str(uuid4()),
                role="assistant",
                content=response.assistant_message.content,
                metadata={"summarized": True},
            )
        except Exception as exc:
            self._logger.error("Strategy F: summary LLM call failed", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_middle_units(
        self, units: list[ReasoningUnit]
    ) -> tuple[list[ReasoningUnit], list[ReasoningUnit], list[ReasoningUnit]]:
        kf = self._trunc_cfg.keep_first_units
        kl = self._trunc_cfg.keep_last_units
        if len(units) < kf + kl:
            return [], [], []
        head = units[:kf]
        end_idx = len(units) - kl if kl > 0 else len(units)
        tail = units[end_idx:] if kl > 0 else []
        middle = units[kf:end_idx]
        return head, middle, tail

    def _make_fits_fn(self, budget, estimator: BaseTokenEstimator) -> Callable[[list[ContextMessage]], bool]:
        def fits(msgs: list[ContextMessage]) -> bool:
            est = estimator.estimate(_to_llm_request(msgs), ["assistant", "tool"])
            return (
                est["assistant"] <= budget.role_budgets["assistant"].token_budget
                and est["tool"] <= budget.role_budgets["tool"].token_budget
            )
        return fits


class TruncatorFactory:
    @classmethod
    def create(
        cls,
        strategy: str,
        budget_manager: BaseTokenBudgetManager,
        logger: Logger,
        config: ConfigReader,
        llm_gateway: LLMGateway
    ) -> ContextTruncator:
        if strategy == "react":
            return ReActContextTruncator(budget_manager, logger, config, llm_gateway)
        raise ValueError(f"Unknown truncation strategy: {strategy!r}")
