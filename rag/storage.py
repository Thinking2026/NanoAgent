from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable


class BaseStorage(ABC):
    backend_name: str = "base"

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def get_documents(self) -> list[dict]:
        raise NotImplementedError


class InMemoryStorage(BaseStorage):
    backend_name = "memory"

    def __init__(self, documents: list[dict] | None = None) -> None:
        self._documents = documents or self._build_default_documents()

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_terms = [term.strip().lower() for term in query.split() if term.strip()]
        ranked = sorted(
            self._documents,
            key=lambda doc: self._score(doc, query_terms),
            reverse=True,
        )
        return [doc for doc in ranked[:top_k] if self._score(doc, query_terms) > 0]

    def get_documents(self) -> list[dict]:
        return list(self._documents)

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


class SQLiteStorage(BaseStorage):
    backend_name = "sqlite"

    def __init__(self, database_path: str) -> None:
        self._database_path = Path(database_path)
        self._initialize()

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        like_query = f"%{query.strip()}%"
        with sqlite3.connect(self._database_path) as connection:
            rows = connection.execute(
                """
                SELECT id, title, content
                FROM documents
                WHERE title LIKE ? OR content LIKE ?
                LIMIT ?
                """,
                (like_query, like_query, top_k),
            ).fetchall()
        return [
            {"id": row[0], "title": row[1], "content": row[2]}
            for row in rows
        ]

    def get_documents(self) -> list[dict]:
        with sqlite3.connect(self._database_path) as connection:
            rows = connection.execute(
                """
                SELECT id, title, content
                FROM documents
                ORDER BY id
                """
            ).fetchall()
        return [
            {"id": row[0], "title": row[1], "content": row[2]}
            for row in rows
        ]

    def seed(self, documents: list[dict]) -> None:
        with sqlite3.connect(self._database_path) as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO documents(id, title, content)
                VALUES (?, ?, ?)
                """,
                [(doc["id"], doc["title"], doc["content"]) for doc in documents],
            )
            connection.commit()

    def _initialize(self) -> None:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL
                )
                """
            )
            connection.commit()


class StorageRegistry:
    def __init__(self, storages: Iterable[BaseStorage] | None = None) -> None:
        self._storages: dict[str, BaseStorage] = {}
        for storage in storages or []:
            self.register(storage)

    def register(self, storage: BaseStorage) -> None:
        self._storages[storage.backend_name] = storage

    def get(self, backend_name: str) -> BaseStorage:
        try:
            return self._storages[backend_name]
        except KeyError as exc:
            available = ", ".join(sorted(self._storages)) or "<none>"
            raise ValueError(
                f"Unknown storage backend: {backend_name}. Available backends: {available}"
            ) from exc

    def list_backends(self) -> list[str]:
        return sorted(self._storages)
