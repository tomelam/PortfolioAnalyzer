"""Tests for the PPF synthetic-CIV loader.

Contract: load_ppf_civ() returns a pd.Series indexed by daily dates with
no NaN values and monotonically increasing values (compounding interest).
Previously returned a monthly Series with chained-inplace ffill that was
a silent no-op since pandas 2.x; downstream from_civ() in
asset_timeseries.py rejected the NaNs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import data_loader
from data_loader import load_ppf_civ

FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures" / "data"


@pytest.fixture
def patch_ppf_rates_path(monkeypatch):
    """Point load_ppf_interest_rates at the tiny fixture, not data/."""
    fixture_path = str(FIXTURES / "ppf_rates_tiny.csv")
    real_loader = data_loader.load_ppf_interest_rates
    monkeypatch.setattr(
        data_loader,
        "load_ppf_interest_rates",
        lambda csv_file_path=fixture_path: real_loader(csv_file_path),
    )


def test_returns_series(patch_ppf_rates_path) -> None:
    s = load_ppf_civ()
    assert isinstance(s, pd.Series)


def test_no_nans(patch_ppf_rates_path) -> None:
    s = load_ppf_civ()
    assert not s.isnull().any(), f"PPF series has {s.isnull().sum()} NaNs"


def test_daily_index(patch_ppf_rates_path) -> None:
    s = load_ppf_civ()
    assert isinstance(s.index, pd.DatetimeIndex)
    # Consecutive index entries should be 1 day apart.
    diffs = s.index.to_series().diff().dropna().unique()
    assert len(diffs) == 1 and diffs[0] == pd.Timedelta(days=1), (
        f"index not daily-spaced; diffs: {diffs}"
    )


def test_monotonic_increasing(patch_ppf_rates_path) -> None:
    s = load_ppf_civ()
    assert s.is_monotonic_increasing, "PPF CIV must monotonically increase (compounding interest)"


def test_starts_near_unity(patch_ppf_rates_path) -> None:
    """The first CIV should be normalized to ~1000 or ~1 depending on the
    function; we just assert it is positive and small relative to the end."""
    s = load_ppf_civ()
    assert s.iloc[0] > 0
    assert s.iloc[-1] > s.iloc[0]
