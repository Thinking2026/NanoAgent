from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from infra.observability.tracing.tracer import Tracer
from infra.rendering_engine import Jinja2PromptRenderer, PromptRenderer
from schemas.event_bus import EventBus
from schemas.task import KnowledgeEntry, Task
from utils.time.time import now as _time_now
from schemas.types import LLMMessage, UnifiedLLMRequest
from utils.env_util.runtime_env import get_project_root
import utils.file.file as file_handler
from utils.log.log import Logger, zap

if TYPE_CHECKING:
    from config.config import ConfigReader
    from llm.llm_gateway import LLMGateway

_COMPACT_THRESHOLD_BYTES = 128 * 1024  # 128 KB — default; overridden by config in __init__
_COMPACT_DROP_LINES = 20

_KNOWLEDGE_FILE_SUBPATH = Path("var") / "knowledge" / "knowledge.json"


class KnowledgeManager:
    def __init__(self, config: ConfigReader, logger: Logger, tracer: Tracer, event_bus: EventBus, renderer: PromptRenderer | None = None):
        self._config = config
        self._logger = logger
        self._tracer = tracer
        self._event_bus = event_bus
        self._file_handler = file_handler
        self._renderer: PromptRenderer = renderer or Jinja2PromptRenderer()
        global _COMPACT_THRESHOLD_BYTES
        _COMPACT_THRESHOLD_BYTES = (
            config.positive_int("knowledge.manager.compact_threshold_bytes", 128 * 1024)
            if config else 128 * 1024
        )

    def _knowledge_path(self) -> Path:
        return get_project_root() / _KNOWLEDGE_FILE_SUBPATH

    def extract_and_save(
        self,
        task: Task,
        result: str,
        llm_gateway: LLMGateway,
        conversation_snippet: str | None = None,
    ) -> list[KnowledgeEntry] | None:
        provider = self._config.get("llm.summary_providers", ["deepseek"])[0] if self._config else "deepseek"
        user_prompt = self._renderer.render("knowledge_manager/extract_prompt.j2", {
            "task": task,
            "result": result,
            "conversation_snippet": conversation_snippet,
        })
        self._logger.info(
            "Knowledge extraction started",
            zap.any("task_id", task.id),
            zap.any("result_length", len(result)),
            zap.any("has_snippet", conversation_snippet is not None),
            zap.any("provider", provider),
        )
        # Note: this span is a no-op when called from a background thread —
        # threading.local() has no active trace in the async context.
        with self._tracer.start_span(
            "knowledge.extract_and_save",
            "knowledge",
            {"task_id": task.id, "result_length": len(result), "provider": provider},
        ) as span:
            response = llm_gateway.generate(
                UnifiedLLMRequest(
                    messages=[LLMMessage(role="user", content=user_prompt)],
                    system_prompt=self._renderer.render("knowledge_manager/system.j2", {}),
                    json_mode=True,
                ),
                provider,
            )
            entries = _parse_knowledge_list(response.assistant_message.content)
            span.add_attributes({"entry_count": len(entries)})
        if not entries:
            self._logger.info("Knowledge extraction produced no reusable entries")
            return None

        path = self._knowledge_path()
        lines = "\n".join(json.dumps(_entry_to_dict(e), ensure_ascii=False) for e in entries) + "\n"
        self._file_handler.append_text(path, lines)
        self.compact()
        self._logger.info(
            "Knowledge entries saved",
            zap.any("path", path),
            zap.any("entry_count", len(entries)),
        )
        return entries

    def compact(self) -> None:
        path = self._knowledge_path()
        if not self._file_handler.exists(path):
            self._logger.info("Knowledge compact skipped, file not found", zap.any("path", path))
            return

        if self._file_handler.file_size(path) <= _COMPACT_THRESHOLD_BYTES:
            self._logger.info(
                "Knowledge compact skipped, below threshold",
                zap.any("path", path),
                zap.any("size", self._file_handler.file_size(path)),
                zap.any("threshold", _COMPACT_THRESHOLD_BYTES),
            )
            return

        raw_lines = self._file_handler.read_lines(path, skip_empty=True)
        trimmed = raw_lines[_COMPACT_DROP_LINES:]
        self._file_handler.write_text(path, "\n".join(trimmed) + "\n" if trimmed else "")
        self._logger.info(
            "Knowledge file compacted",
            zap.any("path", path),
            zap.any("dropped_lines", min(_COMPACT_DROP_LINES, len(raw_lines))),
            zap.any("remaining_lines", len(trimmed)),
        )


def _entry_to_dict(e: KnowledgeEntry) -> dict:
    return {
        "doc_id": e.doc_id,
        "file_name": e.file_name,
        "file_path": e.file_path,
        "doc_title": e.doc_title,
        "doc_type": e.doc_type,
        "chunk_index": e.chunk_index,
        "content": e.content,
        "created_at": e.created_at.isoformat(timespec="seconds"),
    }


def _parse_knowledge_list(text: str) -> list[KnowledgeEntry]:
    from utils.time.time import now as _time_now
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].startswith("```") else lines[1:]
        text = "\n".join(inner)
    try:
        data = json.loads(text)
        if not isinstance(data, list):
            return []
        return [
            KnowledgeEntry(
                doc_id=str(item.get("doc_id", str(uuid4()))),
                file_name=str(item.get("file_name", "")),
                file_path=str(item.get("file_path", "")),
                doc_title=str(item.get("doc_title", item.get("title", ""))),
                doc_type=str(item.get("doc_type", "")),
                chunk_index=int(item.get("chunk_index", 0)),
                content=str(item.get("content", "")),
            )
            for item in data
            if isinstance(item, dict) and item.get("content")
        ]
    except Exception:
        return []
