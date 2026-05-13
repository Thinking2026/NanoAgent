from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from infra.observability.tracing.tracer import Tracer
from infra.rendering_engine import Jinja2PromptRenderer, PromptRenderer
from schemas.task import KnowledgeEntry, KnowledgeEntryType, Task
from utils.time.time import now as _time_now
from schemas.types import LLMMessage, UnifiedLLMRequest
from utils.env_util.runtime_env import get_project_root
import utils.file.file as file_handler
from utils.log.log import Logger, zap

if TYPE_CHECKING:
    from config.config import ConfigReader
    from llm.llm_gateway import LLMGateway

_KNOWLEDGE_FILE_SUBPATH = Path("var") / "knowledge" / "knowledge.json"


class KnowledgeLoader:
    def __init__(self, config: ConfigReader, logger: Logger, tracer: Tracer, renderer: PromptRenderer | None = None) -> None:
        self._config = config
        self._logger = logger
        self._tracer = tracer
        self._file_handler = file_handler
        self._renderer: PromptRenderer = renderer or Jinja2PromptRenderer()

    def _knowledge_path(self) -> Path:
        return get_project_root() / _KNOWLEDGE_FILE_SUBPATH

    def load_all_entries(self) -> list[KnowledgeEntry]:
        path = self._knowledge_path()
        if not self._file_handler.exists(path):
            self._logger.info("Knowledge file not found", zap.any("path", path))
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
            "Knowledge entries loaded",
            zap.any("path", path),
            zap.any("entry_count", len(result)),
            zap.any("raw_line_count", len(raw_lines)),
        )
        return result

    def query_related_knowledge(
        self, task: Task, llm_gateway: LLMGateway) -> list[KnowledgeEntry] | None:
        path = self._knowledge_path()
        if not self._file_handler.exists(path):
            self._logger.info("Knowledge query skipped, file not found", zap.any("task_id", task.id), zap.any("path", path))
            return None

        raw_lines = self._file_handler.read_lines(path, skip_empty=True)
        if not raw_lines:
            self._logger.info("Knowledge query skipped, file empty", zap.any("task_id", task.id), zap.any("path", path))
            return None

        all_entries: list[KnowledgeEntry] = []
        for line in raw_lines:
            try:
                all_entries.append(_entry_from_dict(json.loads(line)))
            except Exception:
                continue

        if not all_entries:
            self._logger.info("Knowledge query skipped, no parseable entries", zap.any("task_id", task.id))
            return None

        all_entries.sort(key=lambda e: e.created_at, reverse=True)
        max_entries_enable_to_load = self._config.get("knowledge.loader.max_entries_enable_to_load", 20)
        if len(all_entries) > max_entries_enable_to_load:
            all_entries = all_entries[-max_entries_enable_to_load:]
        entries_dicts = [_entry_to_dict(e) for e in all_entries]
        prompt = self._renderer.render("knowledge_loader/query_prompt.j2", {
            "task": task,
            "entries": entries_dicts,
        })
        system_prompt = self._renderer.render("knowledge_loader/system.j2", {})
        try:
            provider = self._config.get("llm.summary_providers", ["deepseek"])[0] if self._config else "deepseek"
            self._logger.info(
                "Querying related knowledge",
                zap.any("task_id", task.id),
                zap.any("entry_count", len(all_entries)),
                zap.any("provider", provider),
            )
            with self._tracer.start_span(
                "knowledge.query_related",
                "knowledge",
                {"task_id": task.id, "entry_count": len(all_entries), "provider": provider},
            ) as span:
                response = llm_gateway.generate(
                    UnifiedLLMRequest(
                        messages=[LLMMessage(role="user", content=prompt)],
                        system_prompt=system_prompt,
                        max_tokens=256,
                        temperature=0.0,
                    ),
                    provider,
                )
                indices = _parse_index_list(response.assistant_message.content)
                matched = [all_entries[i] for i in indices if 0 <= i < len(all_entries)]
                span.add_attributes({"matched_count": len(matched), "indices": indices})
            self._logger.info(
                "Related knowledge query complete",
                zap.any("task_id", task.id),
                zap.any("matched_count", len(matched)),
                zap.any("indices", indices),
            )
            return matched if matched else None
        except Exception as exc:
            self._logger.error(
                "Related knowledge query failed",
                zap.any("task_id", task.id),
                zap.any("error", exc),
            )
            return None


def _entry_to_dict(e: KnowledgeEntry) -> dict:
    return {
        "entry_id": e.entry_id,
        "title": e.title,
        "tags": e.tags,
        "content": e.content,
        "entry_type": e.entry_type.value,
        "created_at": e.created_at.isoformat(timespec="seconds"),
    }


def _entry_from_dict(data: dict) -> KnowledgeEntry:
    from uuid import uuid4
    raw_ts = data.get("created_at")
    try:
        created_at = datetime.fromisoformat(raw_ts) if raw_ts else _time_now()
    except (ValueError, TypeError):
        created_at = _time_now()
    raw_type = data.get("entry_type", "")
    entry_type = next(
        (m for m in KnowledgeEntryType if m.value == raw_type),
        KnowledgeEntryType.BUSINESS_BACKGROUND,
    )
    return KnowledgeEntry(
        entry_id=str(data.get("entry_id", str(uuid4()))),
        title=str(data.get("title", "")),
        tags=list(data.get("tags", [])),
        content=str(data.get("content", "")),
        entry_type=entry_type,
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
