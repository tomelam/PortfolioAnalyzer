"""Tests for the risk-free-rate loader and aligner.

Contracts:
- fetch_and_standardize_risk_free_rates(path, fmt, max_delay) returns a
  Series whose values have been divided by 100 (percent → decimal).
- align_dynamic_risk_free_rates(portfolio_returns, risk_free_data)
  reindexes to the portfolio's dates, time-interpolates gaps, and
  forward/back-fills any remaining edges.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from risk_free_loader import (
    align_dynamic_risk_free_rates,
    fetch_and_standardize_risk_free_rates,
)

FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures" / "data"
FIXTURE = str(FIXTURES / "risk_free_tiny.csv")


def test_fetch_returns_series() -> None:
    s = fetch_and_standardize_risk_free_rates(FIXTURE, "%d/%m/%Y", max_allowed_delay_days=None)
    assert isinstance(s, pd.Series)


def test_fetch_converts_percent_to_decimal() -> None:
    # Fixture has 6.95, 6.98, 7.01, 7.00, 7.05 in percent.
    s = fetch_and_standardize_risk_free_rates(FIXTURE, "%d/%m/%Y", max_allowed_delay_days=None)
    assert s.iloc[0] == pytest.approx(0.0695)
    assert s.iloc[-1] == pytest.approx(0.0705)


def test_fetch_raises_on_stale_data() -> None:
    with pytest.raises(ValueError):
        # Fixture's latest date is 2024-02-07; today is years later.
        fetch_and_standardize_risk_free_rates(FIXTURE, "%d/%m/%Y", max_allowed_delay_days=3)


def test_align_to_portfolio_dates() -> None:
    rf = fetch_and_standardize_risk_free_rates(FIXTURE, "%d/%m/%Y", max_allowed_delay_days=None)
    portfolio_dates = pd.date_range("2024-02-01", "2024-02-08", freq="D")
    portfolio_returns = pd.Series(0.001, index=portfolio_dates)

    aligned = align_dynamic_risk_free_rates(portfolio_returns, rf)
    assert isinstance(aligned, (pd.Series, pd.DataFrame))
    aligned_series = aligned if isinstance(aligned, pd.Series) else aligned.iloc[:, 0]
    assert len(aligned_series) == len(portfolio_dates)
    assert not aligned_series.isnull().any(), "alignment must fill all gaps"


def test_align_interpolates_weekend_gaps() -> None:
    # Fixture skips Feb 3-4 (weekend). Aligning to a daily index should
    # interpolate values, not leave them NaN.
    rf = fetch_and_standardize_risk_free_rates(FIXTURE, "%d/%m/%Y", max_allowed_delay_days=None)
    portfolio_dates = pd.date_range("2024-02-02", "2024-02-05", freq="D")
    portfolio_returns = pd.Series(0.001, index=portfolio_dates)

    aligned = align_dynamic_risk_free_rates(portfolio_returns, rf)
    aligned_series = aligned if isinstance(aligned, pd.Series) else aligned.iloc[:, 0]
    # Feb 3 and Feb 4 are gaps; interpolation should land between
    # 6.98% (Feb 2) and 7.01% (Feb 5).
    assert 0.0698 <= aligned_series.loc["2024-02-03"] <= 0.0701
    assert 0.0698 <= aligned_series.loc["2024-02-04"] <= 0.0701
