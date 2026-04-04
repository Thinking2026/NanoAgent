from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from utils.timezone import strftime


@dataclass(frozen=True)
class LogField:
    key: str
    value: Any


class zap:
    @staticmethod
    def any(key: str, value: Any) -> LogField:
        return LogField(key=key, value=value)


class Logger:
    def __init__(self, log_dir: str | Path = "logs") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def info(self, description: str, *fields: LogField, **named_fields: Any) -> None:
        self._write("INFO", description, *fields, **named_fields)

    def error(self, description: str, *fields: LogField, **named_fields: Any) -> None:
        self._write("ERR", description, *fields, **named_fields)

    def _write(
        self,
        level: str,
        description: str,
        *fields: LogField,
        **named_fields: Any,
    ) -> None:
        entries = self._build_entries(fields, named_fields)
        caller_file, caller_line = self._get_caller_location()
        entries.insert(0, ("line", caller_line))
        entries.insert(0, ("file", caller_file))
        log_line = self._format_line(level, description, entries)
        log_path = self._build_log_path(level)
        with self._lock:
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(log_line + "\n")

    def _build_entries(
        self,
        fields: tuple[LogField, ...],
        named_fields: dict[str, Any],
    ) -> list[tuple[str, Any]]:
        entries: list[tuple[str, Any]] = []
        for field in fields:
            if not isinstance(field, LogField):
                raise TypeError("logger fields must be created with zap.any(key, value)")
            entries.append((field.key, field.value))
        for key, value in named_fields.items():
            entries.append((key, value))
        return entries

    def _format_line(
        self,
        level: str,
        description: str,
        entries: list[tuple[str, Any]],
    ) -> str:
        parts = [f"[{level}]: {description}"]
        for key, value in entries:
            parts.append(f"[{key}]: {value}")
        return ", ".join(parts)

    def _build_log_path(self, level: str) -> Path:
        timestamp = strftime("%Y%m%d%H")
        suffix = "info" if level == "INFO" else "err"
        return self._log_dir / f"{timestamp}_{suffix}.log"

    @staticmethod
    def _get_caller_location() -> tuple[str, int]:
        frame = inspect.currentframe()
        try:
            if frame is None:
                return ("<unknown>", 0)
            caller = frame.f_back
            while caller is not None:
                code_name = caller.f_code.co_name
                if code_name not in {"_get_caller_location", "_write", "info", "error"}:
                    return (Path(caller.f_code.co_filename).name, caller.f_lineno)
                caller = caller.f_back
            return ("<unknown>", 0)
        finally:
            del frame
