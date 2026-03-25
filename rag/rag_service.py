from __future__ import annotations

from rag.storage import BaseStorage


class RAGService:
    def __init__(self, storage: BaseStorage) -> None:
        self._storage = storage

    def use_storage(self, storage: BaseStorage) -> None:
        self._storage = storage

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        matches = self._storage.search(query, top_k=top_k)
        return [
            {
                "source_id": item["id"],
                "title": item["title"],
                "content": item["content"],
            }
            for item in matches
        ]
