from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AgentReachHealthStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def write_success(
        self,
        *,
        health: dict[str, Any],
        checked_at: float,
        duration_ms: int,
    ) -> dict[str, Any]:
        snapshot = {
            "success": True,
            "checked_at": checked_at,
            "duration_ms": duration_ms,
            "health": health,
            "error": None,
        }
        self._write(snapshot)
        return snapshot

    def write_failure(self, *, message: str, checked_at: float) -> dict[str, Any]:
        snapshot = {
            "success": False,
            "checked_at": checked_at,
            "duration_ms": None,
            "health": {},
            "error": message,
        }
        self._write(snapshot)
        return snapshot

    def _write(self, snapshot: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)
