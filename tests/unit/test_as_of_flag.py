"""Tests for the ``--as-of YYYY-MM-DD`` plumbing.

The flag pins the reference (as-of) date so a run is deterministic
regardless of when the underlying CSVs/NAVs were captured. The plumbing
surfaces in three places that previously called ``pd.Timestamp.today()``
directly:

- ``loaders.benchmark.load_timeseries_csv`` (staleness gate)
- ``utils.to_cutoff_date`` (lookback computation)
- ``fund_lifecycle.fund_dates`` (already accepted ``as_of``; verified here)

The trimming of every NAV/CIV series to ``<= as_of`` lives in ``main.py``
and is exercised end-to-end by the golden tests.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from fund_lifecycle import fund_dates
from loaders.benchmark import load_timeseries_csv
from utils import to_cutoff_date

FIXTURES = Path(__file__).parent / "loaders" / "fixtures"


def test_to_cutoff_date_uses_pinned_as_of() -> None:
    """``--lookback 1Y`` against a pinned as-of date must give 'one year
    before that date', not 'one year before the wall clock'."""
    pinned = pd.Timestamp("2025-06-15")
    result = to_cutoff_date("1Y", as_of=pinned)
    assert result == pd.Timestamp("2024-06-15")


def test_to_cutoff_date_ytd_uses_pinned_as_of() -> None:
    pinned = pd.Timestamp("2025-06-15")
    result = to_cutoff_date("YTD", as_of=pinned)
    assert result == pd.Timestamp("2025-01-01")


def test_to_cutoff_date_defaults_to_wall_clock() -> None:
    """Omitting ``as_of`` falls back to ``pd.Timestamp.today()`` — sanity
    check that the new optional arg didn't break the legacy code path."""
    result = to_cutoff_date("1Y")
    # 1Y back from now is within the past year; sufficient pin without
    # racing the wall clock.
    delta = (pd.Timestamp.today() - result).days
    assert 364 <= delta <= 367


def test_benchmark_staleness_gate_uses_pinned_as_of(tmp_path: Path) -> None:
    """A CSV whose last date is 30 days before the pinned as-of date must
    NOT trip the 3-day staleness gate when ``as_of`` is set to a date
    within that window."""
    csv = tmp_path / "tiny.csv"
    csv.write_text(
        "Date,Price\n01/06/2025,100.0\n02/06/2025,101.0\n03/06/2025,102.0\n"
    )
    # 4 June 2025 is within the gate: last date 3 June is 1 day old.
    ts = load_timeseries_csv(
        str(csv),
        "%d/%m/%Y",
        max_delay_days=3,
        as_of=pd.Timestamp("2025-06-04"),
    )
    assert ts is not None

    # But the same CSV against as_of = 2026-06-04 (a year later) must
    # trip the gate.
    with pytest.raises(RuntimeError, match="outdated"):
        load_timeseries_csv(
            str(csv),
            "%d/%m/%Y",
            max_delay_days=3,
            as_of=pd.Timestamp("2026-06-04"),
        )


def test_fund_dates_uses_pinned_as_of() -> None:
    """The DEFUNCT-status threshold is days-since-last-NAV; pinning the
    as-of date must make the verdict deterministic. (Pre-existing
    behavior; pinned here because main.py now propagates
    settings['as_of'].)"""
    df = pd.DataFrame(
        {"nav": [100.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2025-12-01")], name="date"),
    )
    fresh = fund_dates(df, as_of=dt.date(2025, 12, 5), defunct_threshold_days=30)
    stale = fund_dates(df, as_of=dt.date(2026, 6, 1), defunct_threshold_days=30)
    assert fresh["status"] == "LIVE"
    assert stale["status"] == "DEFUNCT"
