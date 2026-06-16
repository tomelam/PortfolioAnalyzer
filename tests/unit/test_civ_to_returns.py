"""Tests for ``civ_to_returns`` — the CIV → returns conversion utility.

The round-trip identity test (CIV → returns → reconstructed CIV) is
the most valuable here: it catches off-by-one indexing bugs and any
implicit shift the conversion might introduce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from civ_to_returns import civ_to_returns


def _daily_civ(values: list[float], start: str = "2024-01-01") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), name="value")


def test_daily_returns_match_pct_change() -> None:
    civ = _daily_civ([100.0, 101.0, 102.01, 100.99, 102.0])
    returns = civ_to_returns(civ, frequency="daily")

    # length: CIV - 1 (first return is NaN dropped)
    assert len(returns) == len(civ) - 1
    pd.testing.assert_series_equal(
        returns,
        civ.pct_change().dropna(),
        check_names=False,
    )


def test_daily_round_trip_reconstructs_civ_ratio() -> None:
    """CIV → returns → cumprod(1+r) recovers the *normalized* CIV
    (i.e. ratio relative to first value)."""
    civ = _daily_civ([100.0 * (1.0005**i) for i in range(50)])
    returns = civ_to_returns(civ, frequency="daily")
    reconstructed = (1 + returns).cumprod()

    expected = (civ / civ.iloc[0]).iloc[1:]
    np.testing.assert_allclose(reconstructed.values, expected.values, rtol=1e-12)


def test_monthly_resamples_to_month_end() -> None:
    daily = _daily_civ([100.0 * (1.001**i) for i in range(45)], start="2024-01-01")
    monthly = civ_to_returns(daily, frequency="monthly")

    # 45 business days from 2024-01-01 spans Jan + Feb partially → 1 return
    # (Jan-end → Feb-end). Length ≥ 1.
    assert len(monthly) >= 1
    # All values are positive (monotonically increasing input).
    assert (monthly > 0).all()


def test_unsupported_frequency_raises() -> None:
    civ = _daily_civ([100.0, 101.0])
    with pytest.raises(ValueError, match="Unsupported frequency"):
        civ_to_returns(civ, frequency="weekly")


def test_non_series_input_raises() -> None:
    with pytest.raises(TypeError, match="Expected pd.Series"):
        civ_to_returns([100.0, 101.0])  # type: ignore[arg-type]


def test_constant_civ_produces_zero_returns() -> None:
    civ = _daily_civ([100.0] * 10)
    returns = civ_to_returns(civ, frequency="daily")
    assert (returns == 0.0).all()
