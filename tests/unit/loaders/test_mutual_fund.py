"""Tests for the mutual-fund NAV loader.

Contract: fetch_navs(url) hits the mfapi.in JSON endpoint and returns a
DataFrame with a 'nav' column (float) and a sorted DatetimeIndex named
'date'. The function retries on transient network failure and raises
RuntimeError after exhausting retries.

Previously embedded in data_loader.py:606 as fetch_navs_of_mutual_fund;
extracted here so the loader is unit-testable without a network round-trip
and so per-asset extraction can proceed incrementally.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests

from mutual_fund_loader import fetch_navs

SAMPLE_RESPONSE = {
    "status": "SUCCESS",
    "meta": {"fund_house": "Test AMC", "scheme_code": 999999, "scheme_name": "Test Fund"},
    "data": [
        {"date": "31-12-2024", "nav": "123.4567"},
        {"date": "30-12-2024", "nav": "122.9876"},
        {"date": "27-12-2024", "nav": "121.5432"},
    ],
}


def _mock_response(payload, status_code=200):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.text = json.dumps(payload)
    resp.raise_for_status.return_value = None
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status_code}")
    return resp


@pytest.fixture
def patch_requests(monkeypatch):
    calls = []

    def factory(responses):
        it = iter(responses)

        def fake_get(url, timeout=None):
            calls.append(url)
            r = next(it)
            if isinstance(r, Exception):
                raise r
            return r

        monkeypatch.setattr(requests, "get", fake_get)
        return calls

    return factory


def test_returns_dataframe(patch_requests) -> None:
    patch_requests([_mock_response(SAMPLE_RESPONSE)])
    df = fetch_navs("https://api.mfapi.in/mf/999999")
    assert isinstance(df, pd.DataFrame)


def test_has_nav_column_float(patch_requests) -> None:
    patch_requests([_mock_response(SAMPLE_RESPONSE)])
    df = fetch_navs("https://api.mfapi.in/mf/999999")
    assert "nav" in df.columns
    assert df["nav"].dtype == float


def test_datetime_index_sorted(patch_requests) -> None:
    patch_requests([_mock_response(SAMPLE_RESPONSE)])
    df = fetch_navs("https://api.mfapi.in/mf/999999")
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing


def test_parses_dayfirst_dates(patch_requests) -> None:
    patch_requests([_mock_response(SAMPLE_RESPONSE)])
    df = fetch_navs("https://api.mfapi.in/mf/999999")
    # "31-12-2024" → 2024-12-31, not 2024-31-12 or similar.
    assert df.index[-1] == pd.Timestamp("2024-12-31")


def test_retries_on_transient_failure(patch_requests) -> None:
    calls = patch_requests(
        [
            requests.ConnectionError("transient"),
            requests.Timeout("transient"),
            _mock_response(SAMPLE_RESPONSE),
        ]
    )
    df = fetch_navs("https://api.mfapi.in/mf/999999", retries=5)
    assert len(df) == 3
    assert len(calls) == 3


def test_raises_after_exhausting_retries(patch_requests) -> None:
    patch_requests([requests.ConnectionError("nope")] * 4)
    with pytest.raises(RuntimeError, match="after 4 retries"):
        fetch_navs("https://api.mfapi.in/mf/999999", retries=4)


def test_raises_on_missing_data_key(patch_requests) -> None:
    patch_requests([_mock_response({"status": "OK", "data": []})] * 3)
    with pytest.raises(RuntimeError):
        fetch_navs("https://api.mfapi.in/mf/999999", retries=3)
