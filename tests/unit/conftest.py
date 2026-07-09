from __future__ import annotations

import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark test as integration (real network/CLI calls)")
    config.addinivalue_line("markers", "slow: mark test as slow")


@pytest.fixture(autouse=True)
def _set_project_root(monkeypatch):
    monkeypatch.setenv("NANOAGENT_PROJECT_ROOT", str(PROJECT_ROOT))
