"""Tests for the REC-bond series builder.

REC tax-free bonds are fixed-coupon instruments. The loader produces a
daily-frequency cumulative-gain Series using ``calculate_variable_bond_
cumulative_gain`` with a constant rate equal to the coupon declared in
the portfolio TOML.

Previously inlined at main.py:97-105 with a hard-coded 5.25% rate that
ignored the TOML's ``coupon`` field.
"""

from __future__ import annotations

import pandas as pd
import pytest

from loaders.rec_bond import load_rec_bond_series


def test_returns_series_or_dataframe() -> None:
    out = load_rec_bond_series({"coupon": 5.0})
    # calculate_variable_bond_cumulative_gain returns a DataFrame.
    assert isinstance(out, (pd.Series, pd.DataFrame))


def test_uses_coupon_from_spec() -> None:
    """Two different coupons should produce different cumulative values."""
    high = load_rec_bond_series({"coupon": 8.0})
    low = load_rec_bond_series({"coupon": 2.0})

    def last_value(obj):
        return obj.iloc[-1, 0] if isinstance(obj, pd.DataFrame) else obj.iloc[-1]

    assert last_value(high) > last_value(low)


def test_datetime_index_recent() -> None:
    out = load_rec_bond_series({"coupon": 5.0})
    assert isinstance(out.index, pd.DatetimeIndex)
    # Spans roughly 25 years up to "today"
    span = (out.index.max() - out.index.min()).days
    assert span > 20 * 365


def test_default_coupon_when_unset() -> None:
    """If the spec omits 'coupon', fall back to a documented default."""
    out = load_rec_bond_series({})
    assert len(out) > 0  # still produces a series


def test_invalid_coupon_raises() -> None:
    with pytest.raises((ValueError, TypeError)):
        load_rec_bond_series({"coupon": "not-a-number"})
