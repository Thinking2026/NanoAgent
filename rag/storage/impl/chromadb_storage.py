from __future__ import annotations

from rag.storage.storage import BaseStorage
from schemas import build_error


class ChromaDBStorage(BaseStorage):
    backend_name = "chromadb"

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "agent_documents",
    ) -> None:
        try:
            import chromadb
        except ModuleNotFoundError as exc:
            raise build_error(
                "STORAGE_DEPENDENCY_ERROR",
                "ChromaDB storage requires the `chromadb` package to be installed.",
            ) from exc

        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        result = self._collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        return [
            {
                "id": ids[index],
                "title": (metadatas[index] or {}).get("title", ids[index]),
                "content": documents[index],
            }
            for index in range(len(ids))
        ]

    def get_documents(self) -> list[dict]:
        result = self._collection.get(include=["documents", "metadatas"])
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        return [
            {
                "id": ids[index],
                "title": (metadatas[index] or {}).get("title", ids[index]),
                "content": documents[index],
            }
            for index in range(len(ids))
        ]

    def upsert_documents(self, documents: list[dict]) -> None:
        self._collection.upsert(
            ids=[doc["id"] for doc in documents],
            documents=[doc["content"] for doc in documents],
            metadatas=[{"title": doc.get("title", doc["id"])} for doc in documents],
        )
