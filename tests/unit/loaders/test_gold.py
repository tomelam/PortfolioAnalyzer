"""Tests for the gold-price loader.

Contract: gold_loader.load_gold_prices() returns a pd.Series indexed by
date and named 'price'. main.py expects every asset's nav_inputs entry
to be a pd.Series (see main.py:200 assertion). The previous DataFrame
return + dict-style conversion at the call site was broken and silently
asserted out for any portfolio with a [gold] section.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from portfolioanalyzer.loaders.gold import (
    GRAMS_PER_TROY_OUNCE,
    load_gold_prices,
    load_gold_prices_per_gram,
)

FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures" / "data"


def test_returns_series() -> None:
    s = load_gold_prices(csv_path=str(FIXTURES / "gold_tiny.csv"))
    assert isinstance(s, pd.Series), f"expected pd.Series, got {type(s)}"


def test_series_name_is_price() -> None:
    s = load_gold_prices(csv_path=str(FIXTURES / "gold_tiny.csv"))
    assert s.name == "price"


def test_datetime_index() -> None:
    s = load_gold_prices(csv_path=str(FIXTURES / "gold_tiny.csv"))
    assert isinstance(s.index, pd.DatetimeIndex)


def test_sorted_ascending() -> None:
    s = load_gold_prices(csv_path=str(FIXTURES / "gold_tiny.csv"))
    assert s.index.is_monotonic_increasing


def test_values_are_float() -> None:
    s = load_gold_prices(csv_path=str(FIXTURES / "gold_tiny.csv"))
    assert s.dtype == float


def test_first_and_last_values() -> None:
    s = load_gold_prices(csv_path=str(FIXTURES / "gold_tiny.csv"))
    assert s.iloc[0] == pytest.approx(62000.50)
    assert s.iloc[-1] == pytest.approx(65000.00)


def test_missing_file_raises() -> None:
    with pytest.raises(RuntimeError, match="not found"):
        load_gold_prices(csv_path="/nonexistent/path/gold.csv")


def test_unparseable_date_fails_fast(tmp_path) -> None:
    """A malformed date must raise (naming the bad value), not be silently
    coerced to NaT and bend the series."""
    bad = tmp_path / "gold.csv"
    bad.write_text("Date,Spot Price\n2025-01-31,100.0\nnot-a-date,200.0\n")
    with pytest.raises(RuntimeError, match=r"Unparseable date.*not-a-date"):
        load_gold_prices(csv_path=str(bad))


def test_per_gram_is_per_ounce_divided_by_31_103() -> None:
    """Per-gram series = per-ounce series / GRAMS_PER_TROY_OUNCE, point by point."""
    per_oz = load_gold_prices(csv_path=str(FIXTURES / "gold_tiny.csv"))
    per_g = load_gold_prices_per_gram(csv_path=str(FIXTURES / "gold_tiny.csv"))
    expected = per_oz / GRAMS_PER_TROY_OUNCE
    pd.testing.assert_series_equal(per_g, expected, check_names=False)
