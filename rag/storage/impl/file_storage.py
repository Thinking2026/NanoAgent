from __future__ import annotations

import json
from pathlib import Path

from rag.storage.storage import BaseStorage


class FileStorage(BaseStorage):
    backend_name = "file"

    def __init__(self, file_path: str, documents: list[dict] | None = None) -> None:
        self._file_path = Path(file_path)
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._file_path.exists():
            self._write_documents(documents or self._build_default_documents())

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        documents = self.get_documents()
        query_terms = [term.strip().lower() for term in query.split() if term.strip()]
        ranked = sorted(
            documents,
            key=lambda doc: self._score(doc, query_terms),
            reverse=True,
        )
        return [doc for doc in ranked[:top_k] if self._score(doc, query_terms) > 0]

    def get_documents(self) -> list[dict]:
        return json.loads(self._file_path.read_text(encoding="utf-8"))

    def write_documents(self, documents: list[dict]) -> None:
        self._write_documents(documents)

    def _write_documents(self, documents: list[dict]) -> None:
        self._file_path.write_text(
            json.dumps(documents, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _score(document: dict, query_terms: list[str]) -> int:
        haystack = f"{document.get('title', '')} {document.get('content', '')}".lower()
        return sum(1 for term in query_terms if term in haystack)

    @staticmethod
    def _build_default_documents() -> list[dict]:
        return [
            {
                "id": "doc-agent-loop",
                "title": "Agent Event Loop",
                "content": (
                    "Agent prototype should read user messages, retrieve external context, "
                    "call the LLM, optionally execute tools, and write final answers back."
                ),
            },
            {
                "id": "doc-react",
                "title": "ReAct Prompting",
                "content": (
                    "ReAct combines reasoning traces with actions. The agent thinks, selects "
                    "a tool, observes tool output, and then writes a final answer."
                ),
            },
            {
                "id": "doc-threading",
                "title": "Threaded Agent Design",
                "content": (
                    "A simple prototype can use one user thread for CLI IO and one agent "
                    "thread for the autonomous event loop, communicating via queues."
                ),
            },
        ]
