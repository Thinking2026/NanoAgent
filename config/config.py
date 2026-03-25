from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    pass


class JsonConfig:
    def __init__(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path)
        self._data: dict[str, Any] = {}
        self.reload()

    @property
    def config_path(self) -> Path:
        return self._config_path

    def reload(self) -> None:
        if not self._config_path.exists():
            raise ConfigError(f"Config file does not exist: {self._config_path}")

        try:
            self._data = json.loads(self._config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON config: {self._config_path}") from exc

        if not isinstance(self._data, dict):
            raise ConfigError("Top-level JSON config must be an object.")

    def get(self, key_path: str, default: Any = None) -> Any:
        try:
            return self._resolve_path(key_path)
        except ConfigError:
            return default

    def require(self, key_path: str) -> Any:
        return self._resolve_path(key_path)

    def get_object(self, key_path: str = "") -> dict[str, Any]:
        if not key_path:
            return dict(self._data)

        value = self._resolve_path(key_path)
        if not isinstance(value, dict):
            raise ConfigError(f"Config path is not an object: {key_path}")
        return dict(value)

    def has(self, key_path: str) -> bool:
        try:
            self._resolve_path(key_path)
            return True
        except ConfigError:
            return False

    def as_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def _resolve_path(self, key_path: str) -> Any:
        if not key_path:
            return self._data

        current: Any = self._data
        for key in key_path.split("."):
            if not isinstance(current, dict):
                raise ConfigError(f"Cannot traverse non-object node while reading: {key_path}")
            if key not in current:
                raise ConfigError(f"Missing config key: {key_path}")
            current = current[key]
        return current


def load_config(config_path: str | Path) -> JsonConfig:
    return JsonConfig(config_path)
