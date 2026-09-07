"""Tests for the SCSS (Senior Citizens Savings Scheme) loader.

The full pipeline scrapes nsiindia.gov.in for the historical interest
rate table. To test it deterministically we decompose into:
- parse_scss_html(html) — pure parser, returns a DataFrame
- fetch_scss_html(url) — network only, mocked via patched requests.get
- load_scss_interest_rates() — composition; verified via mocking

Contract for parse_scss_html: returns a DataFrame indexed by the
financial-year start date with an 'interest' column of floats.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests
from webgrab import http

from portfolioanalyzer.loaders.scss import (
    fetch_scss_html,
    load_scss_interest_rates,
    parse_scss_html,
)

FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures" / "api_responses"
SAMPLE_HTML = (FIXTURES / "scss_nsi.html").read_text()


def test_parse_returns_dataframe() -> None:
    df = parse_scss_html(SAMPLE_HTML)
    assert isinstance(df, pd.DataFrame)
    assert "interest" in df.columns


def test_parse_datetime_index_sorted() -> None:
    df = parse_scss_html(SAMPLE_HTML)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing


def test_parse_starts_with_known_rate() -> None:
    df = parse_scss_html(SAMPLE_HTML)
    # The 2004-08-02 row has rate 9.0 in the fixture.
    assert df.loc[pd.Timestamp("2004-08-02"), "interest"] == pytest.approx(9.0)


def test_parse_skips_unparseable_to_current_row() -> None:
    """The "01-04-2023 to current" row has a parseable start date.
    Even though "current" isn't a real date, only the start is parsed."""
    df = parse_scss_html(SAMPLE_HTML)
    assert pd.Timestamp("2023-04-01") in df.index


def test_parse_rates_are_floats() -> None:
    df = parse_scss_html(SAMPLE_HTML)
    assert df["interest"].dtype == float


def test_parse_missing_table_returns_empty() -> None:
    df = parse_scss_html("<html><body><p>nothing here</p></body></html>")
    assert df.empty


def test_load_via_mocked_fetch(monkeypatch) -> None:
    """End-to-end loader call with requests.get mocked to return the fixture."""

    def fake_get(url, **kwargs):
        resp = MagicMock(spec=requests.Response)
        resp.text = SAMPLE_HTML
        # webgrab inspects the status directly so it can decline to retry a
        # definite answer; a MagicMock attribute is not an int.
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        return resp

    monkeypatch.setattr(requests, "get", fake_get)
    df = load_scss_interest_rates()
    assert not df.empty
    assert df.index.is_monotonic_increasing
    assert df["interest"].iloc[0] == pytest.approx(9.0)


def test_fetch_propagates_network_error(monkeypatch) -> None:
    """Still loud, but as webgrab's FetchError rather than the raw requests
    exception: every fetcher in this project now fails the same way, and the
    message names the URL and how many attempts were made."""
    def fake_get(url, **kwargs):
        raise requests.ConnectionError("network unreachable")

    monkeypatch.setattr(requests, "get", fake_get)
    with pytest.raises(http.FetchError, match="example.com"):
        fetch_scss_html("https://example.com", timeout=1)
