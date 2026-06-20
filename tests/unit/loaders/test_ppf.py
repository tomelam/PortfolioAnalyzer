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

import portfolioanalyzer.loaders.ppf as ppf_loader
from portfolioanalyzer.loaders.ppf import load_ppf_civ

FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures" / "data"


@pytest.fixture
def patch_ppf_rates_path(monkeypatch):
    """Point load_ppf_interest_rates at the tiny fixture, not data/."""
    fixture_path = str(FIXTURES / "ppf_rates_tiny.csv")
    real_loader = ppf_loader.load_ppf_interest_rates
    monkeypatch.setattr(
        ppf_loader,
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


# Direct tests for the rate-CSV loader (no monkeypatch needed).


def test_rates_loader_returns_dataframe() -> None:
    from portfolioanalyzer.loaders.ppf import load_ppf_interest_rates

    df = load_ppf_interest_rates(str(FIXTURES / "ppf_rates_tiny.csv"))
    assert isinstance(df, pd.DataFrame)
    assert "rate" in df.columns


def test_rates_loader_indexed_by_datetime() -> None:
    from portfolioanalyzer.loaders.ppf import load_ppf_interest_rates

    df = load_ppf_interest_rates(str(FIXTURES / "ppf_rates_tiny.csv"))
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing


def test_rates_loader_missing_file_raises() -> None:
    from portfolioanalyzer.loaders.ppf import load_ppf_interest_rates

    with pytest.raises(FileNotFoundError):
        load_ppf_interest_rates("/nonexistent/ppf.csv")


def test_rates_loader_unparseable_date_fails_fast(tmp_path) -> None:
    """A malformed date must raise (naming the bad value), not be silently
    dropped — a dropped rate change would corrupt the CIV invisibly. Regression
    for the real `2025=01-01` typo found in data/funds/ppf_interest_rates.csv."""
    from portfolioanalyzer.loaders.ppf import load_ppf_interest_rates

    bad = tmp_path / "ppf.csv"
    bad.write_text("date,rate\n2024-10-01,7.1\n2025=01-01,7.1\n")
    with pytest.raises(ValueError, match=r"Unparseable date.*2025=01-01"):
        load_ppf_interest_rates(str(bad))
