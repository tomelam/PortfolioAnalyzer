"""Shared pytest fixtures and network-blocking guardrails."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def fixtures_dir() -> Path:
    return REPO_ROOT / "tests" / "fixtures"


@pytest.fixture
def portfolios_dir() -> Path:
    return REPO_ROOT / "tests" / "fixtures" / "portfolios"


@pytest.fixture
def api_responses_dir() -> Path:
    return REPO_ROOT / "tests" / "fixtures" / "api_responses"


@pytest.fixture
def golden_dir() -> Path:
    return REPO_ROOT / "tests" / "golden"


@pytest.fixture(autouse=True)
def _block_unmocked_network(request, monkeypatch):
    if "network" in request.keywords:
        return

    import requests

    def _blocked(*args, **kwargs):
        raise RuntimeError(
            "Unmocked network call in non-`network` test. "
            "Either mark the test with @pytest.mark.network or mock the request."
        )

    monkeypatch.setattr(requests.sessions.Session, "request", _blocked)
