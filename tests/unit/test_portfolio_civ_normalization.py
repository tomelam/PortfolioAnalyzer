"""Test that combined_civ_series is unit-free (scale-invariant).

The pre-existing implementation summed raw NAVs weighted by allocation,
so a per-unit price of 2500 (PPF) would dominate a per-unit price of 18
(MF NAV) — even though the *fractional* portfolio share was supposed
to be 15% PPF / 30% MF etc. This made the CIV reflect PPF's daily
movements at ~95% instead of the intended 15%, distorting every
metric (Sharpe in particular).

The correct behavior: each asset's CIV is normalized to 1.0 at the
common start date, *then* summed with weights. The result is a
portfolio CIV that starts at 1.0 and reflects the *fractional*
contribution of each asset.
"""

from __future__ import annotations

import pandas as pd
import pytest

from asset_timeseries import from_civ
from portfolio_timeseries import PortfolioTimeseries


def _series(name, values, start="2024-01-01"):
    s = pd.Series(values, index=pd.bdate_range(start, periods=len(values)), name="value")
    return s


def test_scale_invariance_two_assets() -> None:
    """Two assets at the same return profile but very different NAV scales
    should contribute according to their weights, not their scales."""
    nav_high = _series("high", [2500.0 * (1.01**i) for i in range(5)])  # +1% per day
    nav_low = _series("low", [18.0 * (1.01**i) for i in range(5)])  # +1% per day

    assets = {"high": from_civ(nav_high), "low": from_civ(nav_low)}
    portfolio = PortfolioTimeseries(assets=assets, weights={"high": 0.5, "low": 0.5})

    civ = portfolio.combined_civ_series().series
    # Both assets had identical %-returns; portfolio CIV should also have
    # identical %-returns. Starting value should be close to 1 (scale-free).
    daily_returns = civ.pct_change().dropna()
    assert (daily_returns - 0.01).abs().max() < 1e-9
    assert 0.5 < civ.iloc[0] < 2.0


def test_no_single_asset_dominates_by_raw_nav_scale() -> None:
    """If asset A has raw NAV 2500 and asset B has raw NAV 18, but both
    have equal weight, neither should be >70% of the CIV at any point."""
    # Asset A: PPF-like, monotonic +0.02% per day
    nav_high = _series("high", [2500.0 * (1.0002**i) for i in range(10)])
    # Asset B: MF-like, monotonic +0.5% per day
    nav_low = _series("low", [18.0 * (1.005**i) for i in range(10)])

    assets = {"high": from_civ(nav_high), "low": from_civ(nav_low)}
    portfolio = PortfolioTimeseries(assets=assets, weights={"high": 0.5, "low": 0.5})

    civ = portfolio.combined_civ_series().series
    # Track each asset's fractional contribution
    # (we re-compute normalized contributions here to check)
    norm_high = nav_high / nav_high.iloc[0]
    norm_low = nav_low / nav_low.iloc[0]
    expected = 0.5 * norm_high + 0.5 * norm_low

    # CIV should equal the weighted *normalized* sum, regardless of raw NAVs.
    assert civ.iloc[0] == pytest.approx(expected.iloc[0], rel=1e-9)
    assert civ.iloc[-1] == pytest.approx(expected.iloc[-1], rel=1e-9)
