"""The plotted cumulative-portfolio line must agree with the headline
metrics (CAGR / Sharpe / etc.).

Background: ``main.py`` builds two parallel portfolio series:

- ``combined_daily_returns()`` → ``cumprod(1+r)``  — fed to ``visualizer.plot_cumulative_returns``
- ``combined_civ_series()`` → ``TimeseriesReturn`` → CAGR/Sharpe/Sortino

The Phase D CIV-normalization + frequency fix only touched
``combined_civ_series``. ``combined_daily_returns`` was left alone, so
when a mixed-frequency portfolio (daily MFs + monthly gold) is run, the
plot's blue line and the metrics box disagree — the headline CAGR
reports ~12.5%/year while the plotted endpoint shows ~2%/year.

The two paths must agree: end / start of the plotted series must equal
(1 + CAGR) ** years, within tolerance.
"""

from __future__ import annotations

import pandas as pd
import pytest

import metrics
from asset_timeseries import from_civ
from portfolio_timeseries import PortfolioTimeseries


def _bdaily(values: list[float], start: str = "2020-01-01") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), name="value")


def _monthly(values: list[float], start: str = "2020-01-31") -> pd.Series:
    return pd.Series(
        values,
        index=pd.date_range(start, periods=len(values), freq="ME"),
        name="value",
    )


def _plot_series_matches_cagr(portfolio: PortfolioTimeseries, rel_tol: float) -> None:
    """Helper: pin the contract that ``main.py`` uses for the blue line.

    main.py feeds ``portfolio.combined_civ_series().series`` to the plotter
    as ``cumulative_historical``. That series must satisfy
    ``end/start ≈ (1+CAGR)^years``, so the plotted endpoint and the
    metrics-box CAGR describe the same growth.
    """
    civ_series = portfolio.combined_civ_series().series
    cagr = metrics.cagr(civ_series)

    years = (civ_series.index[-1] - civ_series.index[0]).days / 365.25
    plot_ratio = civ_series.iloc[-1] / civ_series.iloc[0]
    expected_ratio = (1 + cagr) ** years

    assert plot_ratio == pytest.approx(expected_ratio, rel=rel_tol), (
        f"Plot endpoint ratio {plot_ratio:.4f} disagrees with CAGR-implied "
        f"{expected_ratio:.4f} (CAGR={cagr:.4f}, years={years:.2f}). "
        "The plotted blue line will mismatch the headline metrics box."
    )


def test_plot_endpoint_matches_cagr_for_mf_only_portfolio() -> None:
    """Single-frequency case (daily MFs only)."""
    mf1 = _bdaily([100.0 * (1.0005**i) for i in range(252)])  # +0.05%/day
    mf2 = _bdaily([200.0 * (1.0007**i) for i in range(252)])  # +0.07%/day

    portfolio = PortfolioTimeseries(
        assets={"mf1": from_civ(mf1), "mf2": from_civ(mf2)},
        weights={"mf1": 0.5, "mf2": 0.5},
    )
    _plot_series_matches_cagr(portfolio, rel_tol=1e-9)


def test_plot_endpoint_matches_cagr_for_mixed_frequency_portfolio() -> None:
    """Mixed-frequency case (daily MFs + monthly gold) — the case that
    surfaced the bug originally in the live ``port-everything`` plot."""
    mf = _bdaily([100.0 * (1.0005**i) for i in range(252)])
    gold = _monthly([5000.0 * (1.01**i) for i in range(12)])

    portfolio = PortfolioTimeseries(
        assets={"mf": from_civ(mf), "gold": from_civ(gold)},
        weights={"mf": 0.7, "gold": 0.3},
    )
    _plot_series_matches_cagr(portfolio, rel_tol=1e-9)


def test_old_cumprod_path_disagrees_for_mixed_frequency() -> None:
    """Pin the *reason* the old plot path was wrong: cumprod(1+r) on the
    weighted-sum-of-asset-returns series materially disagrees with the
    portfolio CIV / CAGR when assets are at different frequencies.

    Documents the bug for future contributors; if this ever passes, it
    means ``combined_daily_returns`` has been independently re-aligned
    and both aggregation paths agree."""
    mf = _bdaily([100.0 * (1.0005**i) for i in range(252)])
    gold = _monthly([5000.0 * (1.01**i) for i in range(12)])

    portfolio = PortfolioTimeseries(
        assets={"mf": from_civ(mf), "gold": from_civ(gold)},
        weights={"mf": 0.7, "gold": 0.3},
    )

    civ_series = portfolio.combined_civ_series().series
    cagr = metrics.cagr(civ_series)

    daily_returns = portfolio.combined_daily_returns()
    old_plot_series = (1 + daily_returns).cumprod()

    years = (old_plot_series.index[-1] - old_plot_series.index[0]).days / 365.25
    old_ratio = old_plot_series.iloc[-1]
    expected_ratio = (1 + cagr) ** years

    # The old path is materially off — assert disagreement of >5%.
    assert abs(old_ratio - expected_ratio) / expected_ratio > 0.05, (
        "combined_daily_returns now agrees with combined_civ_series — "
        "delete this regression-documentation test and pin the consistency "
        "as an invariant on combined_daily_returns instead."
    )
