"""Tests for the benchmark CSV loader.

Contract: load_timeseries_csv(file_path, date_format, max_delay_days)
returns a TimeseriesReturn whose underlying Series has a DatetimeIndex,
float values (commas stripped), is sorted ascending, and raises
RuntimeError when stale-data check (max_delay_days != None) fails.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from benchmark_loader import load_timeseries_csv
from timeseries import TimeseriesReturn

FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures" / "data"


def test_returns_timeseries_return() -> None:
    ts = load_timeseries_csv(str(FIXTURES / "benchmark_tiny.csv"), "%d/%m/%Y")
    assert isinstance(ts, TimeseriesReturn)


def test_underlying_series_is_sorted_datetime() -> None:
    ts = load_timeseries_csv(str(FIXTURES / "benchmark_tiny.csv"), "%d/%m/%Y")
    s = ts.value_series()
    assert isinstance(s.index, pd.DatetimeIndex)
    assert s.index.is_monotonic_increasing


def test_commas_stripped_from_values() -> None:
    ts = load_timeseries_csv(str(FIXTURES / "benchmark_tiny.csv"), "%d/%m/%Y")
    s = ts.value_series()
    assert s.iloc[0] == pytest.approx(21000.50)
    assert s.iloc[-1] == pytest.approx(21400.10)
    assert s.dtype == float


def test_no_age_check_when_max_delay_none() -> None:
    # The fixture's latest date is 2024-02-07; today is years later.
    # Should NOT raise because max_delay_days is None.
    ts = load_timeseries_csv(str(FIXTURES / "benchmark_tiny.csv"), "%d/%m/%Y", max_delay_days=None)
    assert ts is not None


def test_raises_when_data_stale() -> None:
    with pytest.raises(RuntimeError, match="outdated"):
        load_timeseries_csv(str(FIXTURES / "benchmark_tiny.csv"), "%d/%m/%Y", max_delay_days=3)


def test_raises_on_bad_date_format() -> None:
    # Fixture is dd/mm/yyyy; passing yyyy-mm-dd should fail.
    with pytest.raises(ValueError, match="date parsing failed"):
        load_timeseries_csv(str(FIXTURES / "benchmark_tiny.csv"), "%Y-%m-%d")
