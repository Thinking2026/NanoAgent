from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from infra.observability.tracing.tracer import Tracer
from infra.rendering_engine import Jinja2PromptRenderer, PromptRenderer
from schemas.event_bus import EventBus
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


class PersonalityManager:
    def __init__(self, config: ConfigReader, logger: Logger, tracer: Tracer, event_bus: EventBus, renderer: PromptRenderer | None = None) -> None:
        self._config = config
        self._logger = logger
        self._tracer = tracer
        self._event_bus = event_bus
        self._file_handler = file_handler
        self._renderer: PromptRenderer = renderer or Jinja2PromptRenderer()

    def _preference_path(self) -> Path:
        return get_project_root() / _PREFERENCE_FILE_SUBPATH

    def extract_and_save_user_preference(
        self,
        task_description: str,
        llm_gateway: LLMGateway,
        conversation_snippet: str | None = None,
    ) -> list[UserPreferenceEntry] | None:
        provider = self._config.get("llm.summary_providers", ["deepseek"])[0] if self._config else "deepseek"
        user_prompt = self._renderer.render("personality_manager/extract_prompt.j2", {
            "task_description": task_description,
            "conversation_snippet": conversation_snippet,
        })
        self._logger.info(
            "User preference extraction started",
            zap.any("task_length", len(task_description)),
            zap.any("has_snippet", conversation_snippet is not None),
            zap.any("provider", provider),
        )
        with self._tracer.start_span(
            "personality.extract_preferences",
            "personality",
            {"task_length": len(task_description), "provider": provider},
        ) as span:
            response = llm_gateway.generate(
                UnifiedLLMRequest(
                    messages=[LLMMessage(role="user", content=user_prompt)],
                    system_prompt=self._renderer.render("personality_manager/system_extract.j2", {}),
                    temperature=0.0,
                    json_mode=True,
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

    def query_user_preference(self) -> str | None:
        summary_subpath = (
            self._config.get(
                "personality_manager.summary_file",
                "var/personality/user_preference_summary.md",
            )
            if self._config
            else "var/personality/user_preference_summary.md"
        )
        path = get_project_root() / summary_subpath
        if not self._file_handler.exists(path):
            self._logger.info("User preference summary not found", zap.any("path", path))
            return None
        content = self._file_handler.read_text(path).strip()
        self._logger.info("User preference summary", zap.any("content", content))

        return content or None

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
