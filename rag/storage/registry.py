from __future__ import annotations

from typing import Iterable

from rag.storage.storage import BaseStorage
from schemas import build_error


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
            raise build_error(
                "STORAGE_BACKEND_NOT_FOUND",
                f"Unknown storage backend: {backend_name}. Available backends: {available}",
            ) from exc

    def list_backends(self) -> list[str]:
        return sorted(self._storages)

    def close_all(self) -> None:
        for storage in self._storages.values():
            storage.close()
