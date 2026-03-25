from __future__ import annotations

import sqlite3
from pathlib import Path

from rag.storage.storage import BaseStorage


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
