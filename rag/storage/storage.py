from __future__ import annotations

from abc import ABC, abstractmethod


class BaseStorage(ABC):
    backend_name: str = "base"

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def get_documents(self) -> list[dict]:
        raise NotImplementedError

    def close(self) -> None:
        return None
