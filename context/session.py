from __future__ import annotations

import threading

from schemas import SessionStatus


class Session:
    def __init__(self, status: SessionStatus = SessionStatus.NEW_TASK) -> None:
        self._status = status
        self._lock = threading.RLock()

    def get_status(self) -> SessionStatus:
        with self._lock:
            return self._status

    def set_status(self, status: SessionStatus) -> None:
        with self._lock:
            self._status = status

    def begin(self) -> None:
        self.set_status(SessionStatus.IN_PROGRESS)

    def reset(self) -> None:
        self.set_status(SessionStatus.NEW_TASK)
