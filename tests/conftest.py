"""Shared pytest fixtures and network-blocking guardrails."""

from __future__ import annotations

import pytest


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
