from __future__ import annotations

from .config import JsonConfig


class ConfigValueReader:
    def __init__(self, config: JsonConfig) -> None:
        self._config = config

    def positive_float(self, key_path: str, default: float) -> float:
        try:
            value = float(self._config.get(key_path, default))
        except (TypeError, ValueError):
            return default
        if value <= 0:
            return default
        return value

    def positive_int(self, key_path: str, default: int) -> int:
        try:
            value = int(self._config.get(key_path, default))
        except (TypeError, ValueError):
            return default
        if value <= 0:
            return default
        return value

    def retry_delays(
        self,
        key_path: str,
        default: tuple[float, ...] = (1.0, 2.0, 4.0),
    ) -> tuple[float, ...]:
        raw = self._config.get(key_path, list(default))
        if not isinstance(raw, list):
            return default

        parsed: list[float] = []
        for item in raw:
            try:
                value = float(item)
            except (TypeError, ValueError):
                continue
            if value > 0:
                parsed.append(value)
        if not parsed:
            return default
        return tuple(parsed)
