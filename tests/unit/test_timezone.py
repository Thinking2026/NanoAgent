from __future__ import annotations

import os
from datetime import timezone, timedelta

import pytest

from utils.env_util.runtime_env import TIMEZONE_ENV, set_timezone_name
from utils.time.time import (
    get_timezone,
    now,
    isoformat,
    strftime,
    timestamp_full,
    timestamp_date,
    timezone_label,
    log_timestamp,
)


# ---------------------------------------------------------------------------
# get_timezone
# ---------------------------------------------------------------------------

def test_get_timezone_default_is_utc8(monkeypatch):
    monkeypatch.delenv(TIMEZONE_ENV, raising=False)
    tz = get_timezone()
    assert tz.utcoffset(None) == timedelta(hours=8)


def test_get_timezone_shanghai(monkeypatch):
    monkeypatch.setenv(TIMEZONE_ENV, "shanghai")
    tz = get_timezone()
    assert tz.utcoffset(None) == timedelta(hours=8)


def test_get_timezone_beijing(monkeypatch):
    monkeypatch.setenv(TIMEZONE_ENV, "beijing")
    tz = get_timezone()
    assert tz.utcoffset(None) == timedelta(hours=8)


def test_get_timezone_utc(monkeypatch):
    monkeypatch.setenv(TIMEZONE_ENV, "utc")
    tz = get_timezone()
    assert tz == timezone.utc


def test_get_timezone_gmt(monkeypatch):
    monkeypatch.setenv(TIMEZONE_ENV, "gmt")
    tz = get_timezone()
    assert tz == timezone.utc


def test_get_timezone_unknown_defaults_to_utc8(monkeypatch):
    monkeypatch.setenv(TIMEZONE_ENV, "unknown_tz")
    tz = get_timezone()
    assert tz.utcoffset(None) == timedelta(hours=8)


# ---------------------------------------------------------------------------
# now / isoformat / strftime
# ---------------------------------------------------------------------------

def test_now_returns_aware_datetime(monkeypatch):
    monkeypatch.delenv(TIMEZONE_ENV, raising=False)
    dt = now()
    assert dt.tzinfo is not None


def test_isoformat_returns_string(monkeypatch):
    monkeypatch.delenv(TIMEZONE_ENV, raising=False)
    result = isoformat()
    assert isinstance(result, str)
    assert "T" in result or len(result) > 10


def test_isoformat_seconds_precision(monkeypatch):
    monkeypatch.delenv(TIMEZONE_ENV, raising=False)
    result = isoformat(timespec="seconds")
    # Should not have microseconds
    assert "." not in result.split("+")[0].split("-")[-1]


def test_strftime_format(monkeypatch):
    monkeypatch.delenv(TIMEZONE_ENV, raising=False)
    result = strftime("%Y-%m-%d")
    assert len(result) == 10
    parts = result.split("-")
    assert len(parts) == 3


def test_timestamp_full_format(monkeypatch):
    monkeypatch.delenv(TIMEZONE_ENV, raising=False)
    result = timestamp_full()
    assert len(result) == 19  # YYYY-MM-DD HH:MM:SS
    assert result[4] == "-"
    assert result[10] == " "


def test_timestamp_date_format(monkeypatch):
    monkeypatch.delenv(TIMEZONE_ENV, raising=False)
    result = timestamp_date()
    assert len(result) == 10
    assert result[4] == "-"
    assert result[7] == "-"


# ---------------------------------------------------------------------------
# timezone_label
# ---------------------------------------------------------------------------

def test_timezone_label_utc8(monkeypatch):
    monkeypatch.setenv(TIMEZONE_ENV, "shanghai")
    label = timezone_label()
    assert label == "UTC+8"


def test_timezone_label_utc(monkeypatch):
    monkeypatch.setenv(TIMEZONE_ENV, "utc")
    label = timezone_label()
    assert label == "UTC"


# ---------------------------------------------------------------------------
# log_timestamp
# ---------------------------------------------------------------------------

def test_log_timestamp_format(monkeypatch):
    monkeypatch.delenv(TIMEZONE_ENV, raising=False)
    result = log_timestamp()
    assert isinstance(result, str)
    assert "UTC" in result
    # Should contain date and time
    assert "-" in result
    assert ":" in result
