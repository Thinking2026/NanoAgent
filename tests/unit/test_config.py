from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.config import ConfigReader
from schemas.errors import ConfigError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_config(data: dict, tmp_path: Path) -> Path:
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# ConfigReader — loading
# ---------------------------------------------------------------------------

def test_load_simple_config(tmp_path):
    p = write_config({"key": "value"}, tmp_path)
    cfg = ConfigReader(p)
    assert cfg.get("key") == "value"


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        ConfigReader(tmp_path / "nonexistent.json")


def test_invalid_json_raises_config_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json}", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid JSON"):
        ConfigReader(p)


def test_non_object_json_raises_config_error(tmp_path):
    p = tmp_path / "array.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be an object"):
        ConfigReader(p)


# ---------------------------------------------------------------------------
# ConfigReader — get / require / has
# ---------------------------------------------------------------------------

def test_get_nested_key(tmp_path):
    p = write_config({"a": {"b": {"c": 42}}}, tmp_path)
    cfg = ConfigReader(p)
    assert cfg.get("a.b.c") == 42


def test_get_missing_returns_default(tmp_path):
    p = write_config({}, tmp_path)
    cfg = ConfigReader(p)
    assert cfg.get("missing.key", "default") == "default"


def test_get_missing_returns_none_by_default(tmp_path):
    p = write_config({}, tmp_path)
    cfg = ConfigReader(p)
    assert cfg.get("missing") is None


def test_require_existing_key(tmp_path):
    p = write_config({"name": "agent"}, tmp_path)
    cfg = ConfigReader(p)
    assert cfg.require("name") == "agent"


def test_require_missing_key_raises(tmp_path):
    p = write_config({}, tmp_path)
    cfg = ConfigReader(p)
    with pytest.raises(ConfigError, match="Missing config key"):
        cfg.require("missing")


def test_has_existing_key(tmp_path):
    p = write_config({"x": 1}, tmp_path)
    cfg = ConfigReader(p)
    assert cfg.has("x") is True


def test_has_missing_key(tmp_path):
    p = write_config({}, tmp_path)
    cfg = ConfigReader(p)
    assert cfg.has("missing") is False


def test_has_nested_key(tmp_path):
    p = write_config({"a": {"b": 1}}, tmp_path)
    cfg = ConfigReader(p)
    assert cfg.has("a.b") is True
    assert cfg.has("a.c") is False


# ---------------------------------------------------------------------------
# ConfigReader — get_object / as_dict
# ---------------------------------------------------------------------------

def test_get_object_root(tmp_path):
    data = {"a": 1, "b": 2}
    p = write_config(data, tmp_path)
    cfg = ConfigReader(p)
    assert cfg.get_object() == data


def test_get_object_nested(tmp_path):
    p = write_config({"section": {"x": 10}}, tmp_path)
    cfg = ConfigReader(p)
    assert cfg.get_object("section") == {"x": 10}


def test_get_object_non_dict_raises(tmp_path):
    p = write_config({"key": "string"}, tmp_path)
    cfg = ConfigReader(p)
    with pytest.raises(ConfigError, match="not an object"):
        cfg.get_object("key")


def test_as_dict(tmp_path):
    data = {"a": 1}
    p = write_config(data, tmp_path)
    cfg = ConfigReader(p)
    assert cfg.as_dict() == data


def test_as_dict_is_copy(tmp_path):
    p = write_config({"a": 1}, tmp_path)
    cfg = ConfigReader(p)
    d = cfg.as_dict()
    d["b"] = 2
    assert cfg.get("b") is None


# ---------------------------------------------------------------------------
# ConfigReader — reload
# ---------------------------------------------------------------------------

def test_reload_picks_up_changes(tmp_path):
    p = write_config({"v": 1}, tmp_path)
    cfg = ConfigReader(p)
    assert cfg.get("v") == 1
    p.write_text(json.dumps({"v": 99}), encoding="utf-8")
    cfg.reload()
    assert cfg.get("v") == 99


def test_config_path_property(tmp_path):
    p = write_config({}, tmp_path)
    cfg = ConfigReader(p)
    assert cfg.config_path == p


# ---------------------------------------------------------------------------
# ConfigReader — traverse non-object node
# ---------------------------------------------------------------------------

def test_traverse_non_object_raises(tmp_path):
    p = write_config({"a": "string"}, tmp_path)
    cfg = ConfigReader(p)
    with pytest.raises(ConfigError, match="Cannot traverse"):
        cfg.require("a.b")


# ---------------------------------------------------------------------------
# ConfigReader typed helpers
# ---------------------------------------------------------------------------

def make_reader(data: dict, tmp_path: Path) -> ConfigReader:
    p = write_config(data, tmp_path)
    return ConfigReader(p)


def test_positive_float_valid(tmp_path):
    r = make_reader({"timeout": 5.5}, tmp_path)
    assert r.positive_float("timeout", 1.0) == pytest.approx(5.5)


def test_positive_float_uses_default_when_missing(tmp_path):
    r = make_reader({}, tmp_path)
    assert r.positive_float("missing", 2.0) == pytest.approx(2.0)


def test_positive_float_uses_default_when_zero(tmp_path):
    r = make_reader({"t": 0}, tmp_path)
    assert r.positive_float("t", 3.0) == pytest.approx(3.0)


def test_positive_float_uses_default_when_negative(tmp_path):
    r = make_reader({"t": -1}, tmp_path)
    assert r.positive_float("t", 3.0) == pytest.approx(3.0)


def test_positive_float_uses_default_when_non_numeric(tmp_path):
    r = make_reader({"t": "abc"}, tmp_path)
    assert r.positive_float("t", 1.5) == pytest.approx(1.5)


def test_positive_int_valid(tmp_path):
    r = make_reader({"count": 10}, tmp_path)
    assert r.positive_int("count", 1) == 10


def test_positive_int_uses_default_when_zero(tmp_path):
    r = make_reader({"count": 0}, tmp_path)
    assert r.positive_int("count", 5) == 5


def test_positive_int_uses_default_when_negative(tmp_path):
    r = make_reader({"count": -3}, tmp_path)
    assert r.positive_int("count", 5) == 5


def test_positive_int_uses_default_when_non_numeric(tmp_path):
    r = make_reader({"count": "bad"}, tmp_path)
    assert r.positive_int("count", 5) == 5


def test_retry_delays_valid(tmp_path):
    r = make_reader({"delays": [1.0, 2.0, 4.0]}, tmp_path)
    assert r.retry_delays("delays") == (1.0, 2.0, 4.0)


def test_retry_delays_filters_non_positive(tmp_path):
    r = make_reader({"delays": [0, -1, 2.0, 3.0]}, tmp_path)
    result = r.retry_delays("delays")
    assert result == (2.0, 3.0)


def test_retry_delays_uses_default_when_missing(tmp_path):
    r = make_reader({}, tmp_path)
    default = (1.0, 2.0, 4.0)
    assert r.retry_delays("missing", default) == default


def test_retry_delays_uses_default_when_not_list(tmp_path):
    r = make_reader({"delays": "bad"}, tmp_path)
    default = (1.0,)
    assert r.retry_delays("delays", default) == default


def test_retry_delays_uses_default_when_all_invalid(tmp_path):
    r = make_reader({"delays": [0, -1]}, tmp_path)
    default = (5.0,)
    assert r.retry_delays("delays", default) == default
