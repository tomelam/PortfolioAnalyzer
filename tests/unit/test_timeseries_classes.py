"""Tests for the Timeseries{CIV,Return,Asset,Portfolio} class interfaces.

The metric *math* is exercised in tests/unit/test_metrics.py (pure functions)
and tests/test_timeseries.py (legacy delegates). This file pins the
class-level surface: constructors, validation, daily-returns aggregation,
and the hand-rolled ``TimeseriesCIV`` methods that don't delegate to
``metrics``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from timeseries.asset import from_civ
from timeseries.civ import TimeseriesCIV
from timeseries.portfolio import PortfolioTimeseries


def _values_series(values: list[float], start: str = "2024-01-01") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), name="value")


# --- TimeseriesCIV --------------------------------------------------------


def test_timeseries_civ_rejects_non_series() -> None:
    with pytest.raises(TypeError, match="expects a pandas Series"):
        TimeseriesCIV([100.0, 101.0])  # type: ignore[arg-type]


def test_timeseries_civ_rejects_wrong_name() -> None:
    s = pd.Series([100.0, 101.0], index=pd.bdate_range("2024-01-01", periods=2), name="nav")
    with pytest.raises(ValueError, match="Expected series name 'value'"):
        TimeseriesCIV(s)


def test_timeseries_civ_to_returns_daily() -> None:
    civ = TimeseriesCIV(_values_series([100.0, 101.0, 102.0, 103.0]))
    returns = civ.to_returns(frequency="daily")
    assert len(returns) == 3
    assert returns.iloc[0] == pytest.approx(0.01)


def test_timeseries_civ_max_drawdowns_finds_recovered_dip() -> None:
    # Peak 110, trough 100, recovery to 115 → one ~9.09% drawdown.
    civ = TimeseriesCIV(_values_series([100, 110, 105, 100, 115]))
    drawdowns = civ.max_drawdowns(threshold=0.05)
    assert len(drawdowns) == 1
    dd = drawdowns[0]
    assert dd["drawdown"] == pytest.approx((110 - 100) / 110, rel=1e-9)
    assert dd["drawdown_days"] == (civ.series.index[3] - civ.series.index[1]).days


def test_timeseries_civ_max_drawdowns_below_threshold_dropped() -> None:
    # 2% dip, threshold 5% → no drawdown reported.
    civ = TimeseriesCIV(_values_series([100, 102, 100, 102, 105]))
    assert civ.max_drawdowns(threshold=0.05) == []


def test_timeseries_civ_max_drawdowns_unrecovered_reported_with_no_recovery_days() -> None:
    # Peak 110, drops to 95, never recovers → reported with recovery_days=None.
    civ = TimeseriesCIV(_values_series([100, 110, 105, 98, 95]))
    drawdowns = civ.max_drawdowns(threshold=0.05)
    assert len(drawdowns) == 1
    assert drawdowns[0]["recovery_days"] is None


# --- Portfolio daily returns (now derived from CIV pct_change) ---------------


def test_portfolio_daily_returns_via_civ_pct_change() -> None:
    """Daily returns from pct_change of combined_civ_series — the canonical
    path used in main.py for alpha/beta calculation. Two assets with
    identical 1%-per-day exponential growth must produce a uniform daily
    return series at exactly 1%."""
    a = _values_series([100.0 * (1.01**i) for i in range(20)])  # +1%/day
    b = _values_series([200.0 * (1.01**i) for i in range(20)])  # +1%/day too

    portfolio = PortfolioTimeseries(
        assets={"a": from_civ(a), "b": from_civ(b)},
        weights={"a": 0.4, "b": 0.6},
    )
    civ = portfolio.combined_civ_series().series
    daily_returns = civ.pct_change().dropna()

    # 19 returns (20 CIV points minus 1), all non-NaN, all exactly 1%.
    assert len(daily_returns) == 19
    assert daily_returns.notna().all()
    assert (daily_returns - 0.01).abs().max() < 1e-12


# --- PortfolioTimeseries weight validation -------------------------------


def test_portfolio_rejects_weights_that_dont_sum_to_one() -> None:
    a = _values_series([100.0, 101.0])
    b = _values_series([200.0, 201.0])
    with pytest.raises(ValueError, match="weights must sum to 1"):
        PortfolioTimeseries(
            assets={"a": from_civ(a), "b": from_civ(b)},
            weights={"a": 0.4, "b": 0.4},  # sums to 0.8
        )


def test_portfolio_accepts_weights_off_by_floating_point_noise() -> None:
    """Real-world portfolios constructed by hand-typed fractions in TOML
    routinely sum to 0.9999999999999999 (FP rounding) or 1.0000000000000002.
    A strict ``!= 1`` check rejects them spuriously. Use a tolerance band
    matching ``validate_allocations``'s ``tol=0.01``.
    """
    a = _values_series([100.0, 101.0])
    b = _values_series([200.0, 201.0])
    # Six-fund split that sums to 0.9999999999999999 in IEEE-754.
    weights = {
        "a": 0.10526315789473684,
        "b": 0.2631578947368421,
        "c": 0.26842105263157887,
        "d": 0.21052631578947367,
        "e": 0.05263157894736842,
        "f": 0.10,
    }
    # Build six assets via copying a/b so the check can run.
    c = _values_series([300.0, 301.0])
    d = _values_series([400.0, 401.0])
    e = _values_series([500.0, 501.0])
    f = _values_series([600.0, 601.0])
    assets = {
        "a": from_civ(a),
        "b": from_civ(b),
        "c": from_civ(c),
        "d": from_civ(d),
        "e": from_civ(e),
        "f": from_civ(f),
    }
    # Should not raise even though sum != 1 in strict FP terms.
    portfolio = PortfolioTimeseries(assets=assets, weights=weights)
    assert sum(portfolio.weights.values()) == pytest.approx(1.0)


def test_portfolio_rejects_weights_off_by_more_than_tolerance() -> None:
    """One percentage point off is a real schema error, not FP noise."""
    a = _values_series([100.0, 101.0])
    b = _values_series([200.0, 201.0])
    with pytest.raises(ValueError, match="weights must sum to 1"):
        PortfolioTimeseries(
            assets={"a": from_civ(a), "b": from_civ(b)},
            weights={"a": 0.50, "b": 0.49},  # 0.99 — 1% off
        )


def test_portfolio_rejects_empty_assets() -> None:
    with pytest.raises(ValueError, match="at least one asset"):
        PortfolioTimeseries(assets={}, weights={})


# --- AssetTimeseries.summary ---------------------------------------------


def test_asset_timeseries_summary_reports_endpoints_and_total_return() -> None:
    navs = _values_series([100.0, 110.0, 121.0])
    ts = from_civ(navs)
    summary = ts.summary()
    assert summary["Start Value"] == 100.0
    assert summary["End Value"] == 121.0
    # cumret = cumprod(1 + pct_change) → final ratio ≈ 1.21
    assert summary["Total Return"] == pytest.approx(1.21, rel=1e-9)
