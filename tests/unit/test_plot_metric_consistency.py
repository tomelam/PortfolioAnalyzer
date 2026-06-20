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

from portfolioanalyzer import metrics
from portfolioanalyzer.timeseries.asset import from_civ
from portfolioanalyzer.timeseries.portfolio import PortfolioTimeseries


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


def test_alpha_capm_is_sensible_for_mixed_frequency_portfolio() -> None:
    """``alpha_capm`` annualizes its inputs by ``^ 252`` (treating them as
    daily returns). If a caller feeds it mean-monthly returns by mistake,
    the alpha balloons by a factor of ~21 (monthly→daily annualization).

    main.py used to derive ``portfolio_daily_ret`` from
    ``combined_daily_returns()`` which silently collapsed to the *monthly*
    intersection when any monthly asset (gold) was present, producing
    Alpha = 139% for the ``port-everything`` portfolio (with Beta ~0.1).

    Fixed path: derive the daily-return series from
    ``combined_civ_series().series.pct_change()`` — that CIV is daily by
    construction (Phase D frequency fix). This test pins the contract by
    building a mixed-frequency portfolio with realistic noise, computing
    alpha via the new path, and asserting the magnitude is in the same
    order of magnitude as the portfolio's CAGR — not the 20×-inflated
    monthly-treated-as-daily figure.
    """
    import numpy as np

    from portfolioanalyzer.timeseries.returns import TimeseriesReturn

    rng = np.random.default_rng(seed=42)

    # Daily MF with modest drift + noise; ~10%/yr CAGR target.
    mf_idx = pd.bdate_range("2020-01-01", periods=500)
    mf_rets = 0.0004 + 0.005 * rng.standard_normal(len(mf_idx))
    mf = pd.Series(100.0 * np.cumprod(1 + mf_rets), index=mf_idx, name="value")

    # Monthly gold with noise; ~7%/yr CAGR target.
    gold_idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    gold_rets = 0.006 + 0.02 * rng.standard_normal(len(gold_idx))
    gold = pd.Series(5000.0 * np.cumprod(1 + gold_rets), index=gold_idx, name="value")

    portfolio = PortfolioTimeseries(
        assets={"mf": from_civ(mf), "gold": from_civ(gold)},
        weights={"mf": 0.7, "gold": 0.3},
    )

    # Benchmark with realistic noise; ~12%/yr CAGR.
    bench_idx = pd.bdate_range(mf.index.min(), mf.index.max())
    bench_rets = 0.0005 + 0.008 * rng.standard_normal(len(bench_idx))
    bench_ret = pd.Series(bench_rets, index=bench_idx, name="value").iloc[1:]

    # The NEW path — daily returns from combined_civ_series.
    civ = portfolio.combined_civ_series().series
    pf_daily_ret = civ.pct_change().dropna().rename("value")

    alpha = TimeseriesReturn(pf_daily_ret).alpha_capm(
        TimeseriesReturn(bench_ret),
        risk_free_rate=0.0,
    )

    # With ~10% portfolio CAGR vs ~12% benchmark CAGR, |alpha| should be
    # in the low-single-digit percent range — not 100%+.
    assert abs(alpha) < 0.30, (
        f"alpha_capm produced |{alpha:.4f}| > 30% on a portfolio whose "
        "components have realistic single-digit-percent annual returns. "
        "This indicates the annualization is being applied to monthly-"
        "frequency inputs, not daily. Check that the caller is feeding "
        "pct_change of combined_civ_series (daily) rather than "
        "combined_daily_returns (which inner-joins to the monthly "
        "intersection)."
    )


def test_weighted_sum_of_returns_diverges_from_return_of_weighted_sum() -> None:
    """Pin the mathematical reason a previous plot bug existed:
    cumprod((1 + Σ w_i r_i)) ≠ Σ w_i (1 + r_i)^t in general.

    The legacy ``PortfolioTimeseries.combined_daily_returns`` was
    weighted-sum-of-asset-returns. Cumulative-product of that path
    materially diverges from the portfolio CIV (= weighted sum of
    asset *values*) whenever assets have different frequencies or
    different growth rates. This test exhibits the gap so anyone
    proposing to re-introduce the old aggregation has to confront the
    math first.
    """
    # Construct asset returns by hand, no PortfolioTimeseries needed.
    # MF: daily +0.05%/business-day for ~1 year, lifted to monthly via
    # cumprod sampling at month-ends. Then compute returns of the
    # monthly-sampled series. That's "weighted sum of asset returns" on
    # the monthly intersection (because join="inner" with monthly gold
    # would force that frequency in the legacy code).
    mf_dates = pd.bdate_range("2024-01-01", periods=252)
    mf_civ = pd.Series([100.0 * (1.0005**i) for i in range(252)], index=mf_dates)
    gold_dates = pd.date_range("2024-01-31", periods=12, freq="ME")
    gold_civ = pd.Series([5000.0 * (1.01**i) for i in range(12)], index=gold_dates)

    # Align both to the monthly intersection (what the legacy inner-join did)
    mf_monthly = mf_civ.reindex(gold_dates, method="ffill")
    mf_ret = mf_monthly.pct_change().dropna()
    gold_ret = gold_civ.pct_change().dropna()
    # Weighted sum of *returns*.
    w_mf, w_gold = 0.7, 0.3
    weighted_returns = w_mf * mf_ret + w_gold * gold_ret
    cumprod_path = (1 + weighted_returns).cumprod()

    # Now build the *correct* CIV: weighted sum of asset values on a
    # common business-day calendar with ffill on gold.
    bday = pd.bdate_range(gold_dates.min(), gold_dates.max())
    mf_aligned = mf_civ.reindex(bday, method="ffill")
    gold_aligned = gold_civ.reindex(bday, method="ffill")
    mf_norm = mf_aligned / mf_aligned.iloc[0]
    gold_norm = gold_aligned / gold_aligned.iloc[0]
    civ_correct = w_mf * mf_norm + w_gold * gold_norm

    # Annualized growth implied by each path over its respective window.
    cumprod_years = (cumprod_path.index[-1] - cumprod_path.index[0]).days / 365.25
    civ_years = (civ_correct.index[-1] - civ_correct.index[0]).days / 365.25
    cumprod_cagr = cumprod_path.iloc[-1] ** (1 / cumprod_years) - 1
    civ_cagr = (civ_correct.iloc[-1] / civ_correct.iloc[0]) ** (1 / civ_years) - 1

    # The two paths describe different growth. With synthetic data we
    # demonstrate ~1pp divergence; with the real port-everything portfolio
    # the gap was ~10pp. The math fact (≠ in general) holds at any scale.
    assert abs(cumprod_cagr - civ_cagr) > 0.01, (
        f"cumprod path CAGR {cumprod_cagr:.4f} and CIV-path CAGR "
        f"{civ_cagr:.4f} agree more closely than expected. Either the "
        "math is wrong here or the synthetic returns happened to land "
        "such that ≠ became ≈. Adjust the synthetic data."
    )
