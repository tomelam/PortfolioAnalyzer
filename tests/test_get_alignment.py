"""Behavioral test for ``data_loader.get_aligned_portfolio_civs``.

The legacy version loaded a pickled golden produced from a live
mfapi.in run and compared byte-for-byte. That made the test (a)
network-dependent and (b) brittle to pandas version drift.

This rewrite mocks ``requests.get`` to return deterministic synthetic
NAV payloads, then verifies the *contract*:

- The result is a DataFrame indexed by date.
- One column per fund, named with the fund's ``name``.
- The index is the intersection of all funds' date ranges.
- No NaN remains after alignment.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests

from data_loader import get_aligned_portfolio_civs


def _mfapi_payload(dates_and_navs: list[tuple[str, float]]) -> dict:
    """Build a mock mfapi.in JSON payload from (DD-MM-YYYY, nav) pairs."""
    return {
        "status": "SUCCESS",
        "meta": {"fund_house": "Mock AMC", "scheme_code": 0, "scheme_name": "Mock"},
        "data": [{"date": d, "nav": f"{n:.4f}"} for d, n in dates_and_navs],
    }


def _mock_response(payload: dict) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = payload
    resp.text = json.dumps(payload)
    resp.raise_for_status.return_value = None
    return resp


@pytest.fixture
def patch_requests(monkeypatch):
    """Map URL → response; raises if a URL isn't pre-registered."""
    registry: dict[str, MagicMock] = {}

    def fake_get(url, timeout=None):
        if url not in registry:
            raise AssertionError(f"Unmocked URL: {url}")
        return registry[url]

    monkeypatch.setattr(requests, "get", fake_get)
    return registry


@pytest.mark.order(7)
def test_get_aligned_portfolio_civs_aligns_to_common_dates(patch_requests) -> None:
    """Two funds with partially overlapping date ranges → aligned result
    is indexed by the intersection of the two ranges."""

    fund_a_dates = ["02-01-2024", "03-01-2024", "04-01-2024", "05-01-2024", "08-01-2024"]
    fund_b_dates = ["04-01-2024", "05-01-2024", "08-01-2024", "09-01-2024", "10-01-2024"]

    patch_requests["https://api.mfapi.in/mf/AAA"] = _mock_response(
        _mfapi_payload([(d, 100.0 + i) for i, d in enumerate(fund_a_dates)])
    )
    patch_requests["https://api.mfapi.in/mf/BBB"] = _mock_response(
        _mfapi_payload([(d, 200.0 + i) for i, d in enumerate(fund_b_dates)])
    )

    portfolio = {
        "funds": [
            {"name": "Fund A", "url": "https://api.mfapi.in/mf/AAA", "allocation": 0.5},
            {"name": "Fund B", "url": "https://api.mfapi.in/mf/BBB", "allocation": 0.5},
        ]
    }

    result = get_aligned_portfolio_civs(portfolio)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"Fund A", "Fund B"}

    expected_dates = pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08"])
    assert list(result.index) == list(expected_dates)

    assert result.notna().all().all()

    # Spot-check actual NAV values. Fund A's 04-01-2024 was the 3rd entry → 102.
    assert result.at[pd.Timestamp("2024-01-04"), "Fund A"] == pytest.approx(102.0)
    # Fund B's 04-01-2024 was the first entry → 200.
    assert result.at[pd.Timestamp("2024-01-04"), "Fund B"] == pytest.approx(200.0)


def test_get_aligned_portfolio_civs_values_are_floats(patch_requests) -> None:
    patch_requests["https://api.mfapi.in/mf/XYZ"] = _mock_response(
        _mfapi_payload([("02-01-2024", 100.0), ("03-01-2024", 101.0)])
    )
    portfolio = {"funds": [{"name": "X", "url": "https://api.mfapi.in/mf/XYZ"}]}

    result = get_aligned_portfolio_civs(portfolio)
    assert result["X"].dtype == float
