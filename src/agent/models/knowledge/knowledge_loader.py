from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from infra.observability.tracing.tracer import Tracer
from infra.rendering_engine import Jinja2PromptRenderer, PromptRenderer
from schemas.event_bus import EventBus
from schemas.task import KnowledgeEntry, Task
from schemas.types import LLMMessage, UnifiedLLMRequest
from utils.env_util.runtime_env import get_project_root
from utils.log.log import Logger, zap

if TYPE_CHECKING:
    from config.config import ConfigReader
    from llm.llm_gateway import LLMGateway
    from ragplus.embedder import Embedder
    from ragplus.retrieval.reranker import Reranker


class KnowledgeLoader:
    def __init__(
        self,
        config: ConfigReader,
        logger: Logger,
        tracer: Tracer,
        event_bus: EventBus,
        renderer: PromptRenderer | None = None,
    ) -> None:
        self._config = config
        self._logger = logger
        self._tracer = tracer
        self._event_bus = event_bus
        self._renderer: PromptRenderer = renderer or Jinja2PromptRenderer()
        self._embedder: Embedder | None = None
        self._reranker: Reranker | None = None
        # Use local HuggingFace cache; skip network requests on every startup.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    def _get_embedder(self) -> "Embedder":
        if self._embedder is None:
            from ragplus.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder

    def _get_reranker(self) -> "Reranker":
        if self._reranker is None:
            from ragplus.retrieval.reranker import Reranker
            self._reranker = Reranker()
        return self._reranker

    def _index_dir(self) -> Path:
        raw = self._config.get("knowledge.loader.index_dir", "var/knowledge/test_knowledge_base")
        p = Path(raw)
        return p if p.is_absolute() else get_project_root() / p

    def query_related_knowledge(
        self, query: str, task: Task, llm_gateway: LLMGateway
    ) -> str | None:
        index_dir = self._index_dir()
        if not index_dir.exists():
            self._logger.error(
                "Knowledge query skipped, index dir not found",
                zap.any("task_id", task.id),
                zap.any("index_dir", index_dir),
            )
            return None

        if not query or not query.strip():
            self._logger.error(
                "Knowledge query skipped, empty query",
                zap.any("task_id", task.id),
            )
            return None

        try:
            from ragplus.vectorstore import VectorStore
            from ragplus.retrieval.hybrid import HybridRetriever
        except ImportError as exc:
            self._logger.error(
                "ragplus import failed, knowledge query skipped",
                zap.any("task_id", task.id),
                zap.any("error", exc),
            )
            return None

        top_k_search: int = self._config.get("knowledge.loader.top_k_search", 12)
        top_k_rerank: int = self._config.get("knowledge.loader.top_k_search", 8)
        provider = self._config.get("llm.summary_providers", ["deepseek"])[0] if self._config else "deepseek"

        try:
            vectorstore = VectorStore(persist_dir=str(index_dir))
            if not vectorstore.texts:
                self._logger.info(
                    "Knowledge query skipped, index is empty",
                    zap.any("task_id", task.id),
                )
                return None

            embedder = self._get_embedder()
            retriever = HybridRetriever(vectorstore, embedder, bm25_weight=0.3, embedding_weight=0.7)

            self._logger.info(
                "Retrieving knowledge chunks",
                zap.any("task_id", task.id),
                zap.any("query", query[:80]),
                zap.any("top_k", top_k_search),
                zap.any("provider", provider),
            )

            with self._tracer.start_span(
                "knowledge.hybrid_retrieve",
                "knowledge",
                {"task_id": task.id, "top_k": top_k_search},
            ) as span:
                results = retriever.search(query, k=top_k_search)
                span.add_attributes({"retrieved_count": len(results)})

            if not results:
                self._logger.info(
                    "Knowledge query returned no results in hybrid_retrieve phase",
                    zap.any("task_id", task.id),
                )
                return None

            # cross-encoder reranking
            texts = [text for text, _, _ in results]
            meta_by_text = {text: meta for text, meta, _ in results}

            reranker = self._get_reranker()
            reranked = reranker.rerank(query, texts, top_k=top_k_rerank)

            entries: list[KnowledgeEntry] = []
            for text, _ in reranked:
                meta = meta_by_text.get(text)
                if meta is not None:
                    entries.append(_entry_from_meta(meta, text))

            if not entries:
                self._logger.info(
                    "Knowledge query returned no results in rerank phase",
                    zap.any("task_id", task.id),
                )
                return None

        except Exception as exc:
            self._logger.error(
                "Knowledge retrieval failed",
                zap.any("task_id", task.id),
                zap.any("error", exc),
            )
            return None

        # LLM synthesis: relevance filtering + knowledge integration
        entries_dicts = [_entry_to_dict(e) for e in entries]
        self._logger.info(
                "start to use LLM for correlation analysis",
                zap.any("task_id", task.id),
                zap.any("knowledge_entries_recall", entries_dicts),
            )
        prompt = self._renderer.render("knowledge_loader/query_prompt.j2", {
            "task": task,
            "entries": entries_dicts,
        })
        system_prompt = self._renderer.render("knowledge_loader/system.j2", {})
        try:
            with self._tracer.start_span(
                "knowledge.llm_synthesis",
                "knowledge",
                {"task_id": task.id, "chunk_count": len(entries), "provider": provider},
            ) as span:
                response = llm_gateway.generate(
                    UnifiedLLMRequest(
                        messages=[LLMMessage(role="user", content=prompt)],
                        system_prompt=system_prompt,
                        temperature=0.1,
                    ),
                    provider,
                )
                result = response.assistant_message.content.strip()
                span.add_attributes({"result_chars": len(result)})

            self._logger.info(
                "Knowledge synthesis complete",
                zap.any("task_id", task.id),
                zap.any("chunk_count", len(entries)),
                zap.any("result", result),
            )
            return result or None

        except Exception as exc:
            self._logger.error(
                "Knowledge synthesis failed",
                zap.any("task_id", task.id),
                zap.any("error", exc),
            )
            return None


def _entry_from_meta(meta: dict, content: str) -> KnowledgeEntry:
    from datetime import datetime
    from utils.time.time import now as _time_now
    raw_ts = meta.get("doc_create_time")
    try:
        created_at = datetime.fromisoformat(raw_ts) if raw_ts else _time_now()
    except (ValueError, TypeError):
        created_at = _time_now()
    return KnowledgeEntry(
        doc_id=meta.get("doc_id", ""),
        file_name=meta.get("doc_name", ""),
        file_path=meta.get("doc_path", ""),
        doc_title=meta.get("doc_title", ""),
        doc_type=meta.get("doc_type", ""),
        chunk_index=int(meta.get("chunk_index", 0)),
        content=content,
        created_at=created_at,
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
    }
