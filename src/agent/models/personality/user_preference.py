from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from infra.observability.tracing.tracer import Tracer
from schemas.task import Task, UserPreferenceEntry
from utils.time.time import now as _time_now
from schemas.types import LLMMessage, UnifiedLLMRequest
from utils.env_util.runtime_env import get_project_root
import utils.file.file as file_handler
from utils.log.log import Logger, zap

if TYPE_CHECKING:
    from config.config import ConfigReader
    from llm.llm_gateway import LLMGateway

_COMPACT_THRESHOLD_BYTES = 64 * 1024  # 64 KB
_COMPACT_DROP_LINES = 20

_PREFERENCE_FILE_SUBPATH = Path("var") / "personality" / "user_preference.json"

_EXTRACT_SYSTEM_PROMPT = """\
You are a user-preference analyst. Given a user message, extract any personal preferences, \
habits, or style requirements that should be remembered for future interactions.
Return a JSON array of objects. Each object must have:
  - "user_id": string — identifier for the user (use "unknown" if not available)
  - "keywords": array of strings — keywords that describe when this preference applies
  - "content": string — the preference description

If no preferences can be extracted, return an empty JSON array: []
Respond with only valid JSON. No markdown fences."""

_QUERY_SYSTEM_PROMPT = """\
You are a user-preference retrieval assistant. Given a task context and a list of stored user \
preferences (each prefixed with its 0-based index), return the indices of preferences that are \
relevant to the task.
Return a JSON array of integers. If none are relevant, return [].
Respond with only valid JSON. No markdown fences."""


class PersonalityManager:
    def __init__(self, config: ConfigReader, logger: Logger, tracer: Tracer):
        self._config = config
        self._logger = logger
        self._tracer = tracer
        self._file_handler = file_handler

    def _preference_path(self) -> Path:
        return get_project_root() / _PREFERENCE_FILE_SUBPATH

    def extract_and_save_user_preference(
        self, input: str, llm_gateway: LLMGateway) -> list[UserPreferenceEntry] | None:
        provider = self._config.get("llm.summary_providers", ["deepseek"])[0] if self._config else "deepseek"
        self._logger.info(
            "User preference extraction started",
            zap.any("input_length", len(input)),
            zap.any("provider", provider),
        )
        with self._tracer.start_span(
            "personality.extract_preferences",
            "personality",
            {"input_length": len(input), "provider": provider},
        ) as span:
            response = llm_gateway.generate(
                UnifiedLLMRequest(
                    messages=[LLMMessage(role="user", content=input)],
                    system_prompt=_EXTRACT_SYSTEM_PROMPT,
                    max_tokens=512,
                    temperature=0.0,
                ),
                provider,
            )
            entries = _parse_preference_list(response.assistant_message.content)
            span.add_attributes({"entry_count": len(entries)})
        if not entries:
            self._logger.info("User preference extraction produced no entries")
            return None

        path = self._preference_path()
        lines = "\n".join(json.dumps(_entry_to_dict(e), ensure_ascii=False) for e in entries) + "\n"
        self._file_handler.append_text(path, lines)
        self.compact()
        self._logger.info(
            "User preferences saved",
            zap.any("path", path),
            zap.any("entry_count", len(entries)),
        )
        return entries

    def query_related_user_preference(
        self, task: Task, llm_gateway: LLMGateway) -> list[UserPreferenceEntry] | None:
        path = self._preference_path()
        if not self._file_handler.exists(path):
            self._logger.info("Preference query skipped, file not found", zap.any("task_id", task.id), zap.any("path", path))
            return None

        raw_lines = self._file_handler.read_lines(path, skip_empty=True)
        if not raw_lines:
            self._logger.info("Preference query skipped, file empty", zap.any("task_id", task.id), zap.any("path", path))
            return None

        all_entries: list[UserPreferenceEntry] = []
        for line in raw_lines:
            try:
                all_entries.append(_entry_from_dict(json.loads(line)))
            except Exception:
                continue

        if not all_entries:
            self._logger.info("Preference query skipped, no parseable entries", zap.any("task_id", task.id))
            return None

        all_entries.sort(key=lambda e: e.created_at, reverse=True)
        task_context = _build_task_context(task)
        preferences_block = "\n".join(
            f"{i}: {json.dumps(_entry_to_dict(e), ensure_ascii=False)}"
            for i, e in enumerate(all_entries)
        )
        prompt = (
            f"Task context:\n{task_context}\n\n"
            f"Stored preferences (index: JSON):\n{preferences_block}"
        )
        try:
            provider = self._config.get("llm.summary_providers", ["deepseek"])[0] if self._config else "deepseek"
            self._logger.info(
                "Querying related user preferences",
                zap.any("task_id", task.id),
                zap.any("entry_count", len(all_entries)),
                zap.any("provider", provider),
            )
            with self._tracer.start_span(
                "personality.query_preferences",
                "personality",
                {"task_id": task.id, "entry_count": len(all_entries), "provider": provider},
            ) as span:
                response = llm_gateway.generate(
                    UnifiedLLMRequest(
                        messages=[LLMMessage(role="user", content=prompt)],
                        system_prompt=_QUERY_SYSTEM_PROMPT,
                        max_tokens=256,
                        temperature=0.0,
                    ),
                    provider,
                )
                indices = _parse_index_list(response.assistant_message.content)
                matched = [all_entries[i] for i in indices if 0 <= i < len(all_entries)]
                span.add_attributes({"matched_count": len(matched), "indices": indices})
            self._logger.info(
                "Related preference query complete",
                zap.any("task_id", task.id),
                zap.any("matched_count", len(matched)),
                zap.any("indices", indices),
            )
            return matched if matched else None
        except Exception as exc:
            self._logger.error(
                "Related preference query failed",
                zap.any("task_id", task.id),
                zap.any("error", exc),
            )
            return None

    def load_all_preferences(self) -> list[UserPreferenceEntry]:
        path = self._preference_path()
        if not self._file_handler.exists(path):
            self._logger.info("Preference file not found", zap.any("path", path))
            return []
        raw_lines = self._file_handler.read_lines(path, skip_empty=True)
        result = []
        for line in raw_lines:
            try:
                result.append(_entry_from_dict(json.loads(line)))
            except Exception:
                continue
        result.sort(key=lambda e: e.created_at, reverse=True)
        self._logger.info(
            "User preferences loaded",
            zap.any("path", path),
            zap.any("entry_count", len(result)),
            zap.any("raw_line_count", len(raw_lines)),
        )
        return result

    def compact(self) -> None:
        path = self._preference_path()
        if not self._file_handler.exists(path):
            self._logger.info("Preference compact skipped, file not found", zap.any("path", path))
            return

        if self._file_handler.file_size(path) <= _COMPACT_THRESHOLD_BYTES:
            self._logger.info(
                "Preference compact skipped, below threshold",
                zap.any("path", path),
                zap.any("size", self._file_handler.file_size(path)),
                zap.any("threshold", _COMPACT_THRESHOLD_BYTES),
            )
            return

        raw_lines = self._file_handler.read_lines(path, skip_empty=True)
        trimmed = raw_lines[_COMPACT_DROP_LINES:]
        self._file_handler.write_text(path, "\n".join(trimmed) + "\n" if trimmed else "")
        self._logger.info(
            "Preference file compacted",
            zap.any("path", path),
            zap.any("dropped_lines", min(_COMPACT_DROP_LINES, len(raw_lines))),
            zap.any("remaining_lines", len(trimmed)),
        )


def _build_task_context(task: Task) -> str:
    parts = [f"description: {task.description}"]
    if task.intent:
        parts.append(f"intent: {task.intent}")
    if task.task_type:
        parts.append(f"task_type: {task.task_type}")
    if task.output_constraints:
        parts.append(f"output_constraints: {task.output_constraints}")
    if task.notes:
        parts.append(f"notes: {task.notes}")
    if task.required_tools:
        parts.append(f"required_tools: {', '.join(task.required_tools)}")
    return "\n".join(parts)


def _entry_to_dict(e: UserPreferenceEntry) -> dict:
    return {
        "user_id": e.user_id,
        "keywords": e.keywords,
        "content": e.content,
        "created_at": e.created_at.isoformat(timespec="seconds"),
    }


def _entry_from_dict(data: dict) -> UserPreferenceEntry:
    raw_ts = data.get("created_at")
    try:
        created_at = datetime.fromisoformat(raw_ts) if raw_ts else _time_now()
    except (ValueError, TypeError):
        created_at = _time_now()
    return UserPreferenceEntry(
        user_id=str(data.get("user_id", "unknown")),
        keywords=list(data.get("keywords", [])),
        content=str(data.get("content", "")),
        created_at=created_at,
    )


def _parse_index_list(text: str) -> list[int]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].startswith("```") else lines[1:]
        text = "\n".join(inner)
    try:
        data = json.loads(text)
        if not isinstance(data, list):
            return []
        return [int(i) for i in data if isinstance(i, (int, float))]
    except Exception:
        return []


def _parse_preference_list(text: str) -> list[UserPreferenceEntry]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].startswith("```") else lines[1:]
        text = "\n".join(inner)
    try:
        data = json.loads(text)
        if not isinstance(data, list):
            return []
        return [_entry_from_dict(item) for item in data if isinstance(item, dict)]
    except Exception:
        return []
