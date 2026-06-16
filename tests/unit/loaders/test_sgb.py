"""Tests for the SGB (Sovereign Gold Bond) loader.

Contract: create_sgb_daily_returns reads a CSV of SGB tranche issue
dates and issue prices, resamples to daily, computes price returns,
adds the fixed 2.5% annual SGB interest coupon as a daily increment,
and returns a compounded value Series named 'value'.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sgb_loader import create_sgb_daily_returns

FIXTURE = str(
    Path(__file__).resolve().parent.parent.parent / "fixtures" / "data" / "sgb_tiny.csv"
)


def test_returns_series() -> None:
    s = create_sgb_daily_returns(csv_path=FIXTURE)
    assert isinstance(s, pd.Series)


def test_series_named_value() -> None:
    s = create_sgb_daily_returns(csv_path=FIXTURE)
    assert s.name == "value"


def test_datetime_index() -> None:
    s = create_sgb_daily_returns(csv_path=FIXTURE)
    assert isinstance(s.index, pd.DatetimeIndex)
    assert s.index.is_monotonic_increasing


def test_starts_near_initial_value() -> None:
    s = create_sgb_daily_returns(csv_path=FIXTURE, initial_value=100)
    # The first value should be ~100 (initial_value) plus a tiny daily-interest
    # increment. Tolerance accommodates the first-day compounding step.
    assert 99 <= s.iloc[0] <= 101


def test_grows_over_time() -> None:
    s = create_sgb_daily_returns(csv_path=FIXTURE)
    # SGB tranches in the fixture are all positive-trending, plus 2.5% interest.
    # Across the entire range the compounded value must increase.
    assert s.iloc[-1] > s.iloc[0]


def test_missing_file_raises() -> None:
    with pytest.raises((FileNotFoundError, IOError, Exception)):
        create_sgb_daily_returns(csv_path="/nonexistent/sgb.csv")


def test_custom_initial_value_scales_output() -> None:
    s100 = create_sgb_daily_returns(csv_path=FIXTURE, initial_value=100)
    s200 = create_sgb_daily_returns(csv_path=FIXTURE, initial_value=200)
    # Compounded series scale linearly with initial_value.
    assert s200.iloc[-1] == pytest.approx(2 * s100.iloc[-1], rel=1e-9)
