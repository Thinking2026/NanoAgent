from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class JobResult:
    success: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)


class BaseJob(Protocol):
    name: str

    def run_once(self) -> JobResult:
        ...
