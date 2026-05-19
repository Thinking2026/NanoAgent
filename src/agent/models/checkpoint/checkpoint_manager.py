from __future__ import annotations

import pickle
import threading
from datetime import datetime
from pathlib import Path

from schemas.types import CheckpointData


class CheckpointManager:
    BASE_DIR = Path("var/checkpoint")

    @staticmethod
    def save_async(data: CheckpointData, task_name: str) -> None:
        """Serialize *data* to disk in a background thread — does not block the caller."""
        threading.Thread(
            target=CheckpointManager._save,
            args=(data, task_name),
            daemon=True,
        ).start()

    @staticmethod
    def _save(data: CheckpointData, task_name: str) -> None:
        dir_ = CheckpointManager.BASE_DIR / task_name
        dir_.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # step order is 1-based
        filename = f"{data.task_id}_stage_{data.completed_step_index + 1}_{ts}.cpt"
        path = dir_ / filename
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path: str | Path) -> CheckpointData:
        with open(path, "rb") as f:
            return pickle.load(f)
