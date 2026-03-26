from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AgentError(Exception):
    code: str
    message: str

    def __post_init__(self) -> None:
        super().__init__(str(self))

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


def build_error(code: str, message: str) -> AgentError:
    return AgentError(code=code, message=message)


class ConfigError(AgentError):
    def __init__(self, message: str) -> None:
        super().__init__(code="CONFIG_ERROR", message=message)
