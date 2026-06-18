"""Pure-function portfolio metrics.

Each function takes a Series, applies no implicit conversion (returns
are returns; prices are prices) and returns a scalar or list. Intended
as the math layer that ``timeseries.TimeseriesReturn`` delegates to.

Naming convention:
- ``cagr`` and ``max_drawdown*`` consume **price** series (cumulative,
  values like 100, 110, 95, ...).
- ``volatility``, ``sharpe``, ``sortino`` consume **return** series
  (period-over-period deltas, values like 0.01 = 1%).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd


def cagr(prices: pd.Series) -> float:
    """Compound annual growth rate of a price series.

    Uses calendar days / 365.25 for the time-span denominator so leap
    years and irregular sampling are handled correctly.
    """
    s = prices.dropna()
    if len(s) < 2:
        raise ValueError("CAGR calculation requires at least two data points.")

    start_value = s.iloc[0]
    end_value = s.iloc[-1]
    years = (s.index[-1] - s.index[0]).days / 365.25
    if years <= 0:
        raise ValueError("CAGR calculation requires a positive time span.")
    if start_value == 0:
        raise ValueError("CAGR calculation invalid: start value is zero.")
    return (end_value / start_value) ** (1 / years) - 1


def volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized standard deviation of a return series.

    Pass ``periods_per_year=1`` to get the raw (unannualized) std.
    """
    return returns.std() * math.sqrt(periods_per_year)


def sharpe(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio.

    Handles the degenerate cases:
    - Zero volatility + positive excess → +inf
    - Zero volatility + negative excess → -inf
    - Zero volatility + zero excess     → 0.0
    """
    excess = returns - risk_free_rate
    std = returns.std()
    if std is None or math.isnan(std) or std < 1e-12:
        mean_excess = excess.mean()
        if mean_excess > 0:
            return float("inf")
        if mean_excess < 0:
            return float("-inf")
        return 0.0
    return (excess.mean() * periods_per_year) / (std * math.sqrt(periods_per_year))


def sortino(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio. Uses only negative excess returns in the
    denominator. Degenerate cases handled symmetrically to ``sharpe``."""
    excess = returns - risk_free_rate
    fuzz = -1e-10
    downside = excess[excess < fuzz]
    downside_std = downside.std()

    if (
        len(downside) == 0
        or downside_std is None
        or math.isnan(downside_std)
        or downside_std < 1e-12
    ):
        mean_excess = excess.mean()
        if mean_excess > 0:
            return float("inf")
        if mean_excess < 0:
            return float("-inf")
        return 0.0
    return (excess.mean() * periods_per_year) / (downside_std * math.sqrt(periods_per_year))


def max_drawdown(prices: pd.Series) -> float:
    """Maximum (most negative) drawdown of a price series, as a fraction."""
    s = prices.dropna()
    if s.empty:
        raise ValueError("Max drawdown requires at least one data point.")
    cumulative_max = s.cummax()
    drawdowns = (s - cumulative_max) / cumulative_max
    return drawdowns.min()


def max_drawdowns(prices: pd.Series, threshold: float = 0.05) -> list[dict]:
    """Find all drawdowns at least ``threshold`` deep that fully recover.

    A drawdown begins when the series falls below its running peak and
    ends when the series reaches (or exceeds) that prior peak. Only
    fully-recovered drawdowns whose magnitude meets the threshold are
    returned.

    Args:
        prices: A price/CIV-style Series indexed by date.
        threshold: Minimum drawdown magnitude as a fraction (0.05 = 5%).

    Returns:
        A list of dicts, each containing ``start_date``, ``trough_date``,
        ``recovery_date``, and ``drawdown`` (positive fraction).
    """
    s = prices.dropna()
    if s.empty:
        return []

    running_peak = s.cummax()
    out: list[dict] = []
    in_drawdown = False
    peak_value = None
    peak_date = None
    trough_value = None
    trough_date = None

    for date, value in s.items():
        current_peak = running_peak.at[date]
        if value < current_peak:
            if not in_drawdown:
                in_drawdown = True
                peak_value = current_peak
                peak_date = date
                trough_value = value
                trough_date = date
            elif value < trough_value:
                trough_value = value
                trough_date = date
        else:
            if in_drawdown and value >= peak_value:
                drawdown_pct = (peak_value - trough_value) / peak_value
                if drawdown_pct >= threshold:
                    out.append(
                        {
                            "start_date": peak_date,
                            "trough_date": trough_date,
                            "recovery_date": date,
                            "drawdown": drawdown_pct,
                            # Legacy aliases used by the pre-Phase-D test suite:
                            "depth_pct": -drawdown_pct * 100.0,
                            "trough_value": trough_value,
                            "recovery_value": value,
                        }
                    )
                in_drawdown = False
                peak_value = peak_date = trough_value = trough_date = None
    return out


# --- CAPM / regression alpha & beta --------------------------------------
#
# These consume two **return** series (portfolio and benchmark), aligned
# on dates by inner join. ``risk_free_rate`` is a scalar per-period rate.


def beta_capm(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """CAPM beta: ``Cov(Rp - Rf, Rb - Rf) / Var(Rb - Rf)``."""
    if portfolio_returns.empty or benchmark_returns.empty:
        raise ValueError("Cannot compute beta_capm: input series is empty.")

    benchmark_aligned, portfolio_aligned = benchmark_returns.align(
        portfolio_returns, join="inner"
    )
    if len(benchmark_aligned) < 3:
        raise ValueError("Too few aligned data points to compute beta_capm.")

    excess_benchmark = benchmark_aligned - risk_free_rate
    excess_portfolio = portfolio_aligned - risk_free_rate
    return excess_portfolio.cov(excess_benchmark) / excess_benchmark.var()


def beta_regression(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Regression beta: slope of ``Rp`` regressed on ``Rb``."""
    benchmark_aligned, portfolio_aligned = benchmark_returns.align(
        portfolio_returns, join="inner"
    )
    if len(benchmark_aligned) < 2:
        raise ValueError("Beta requires at least two aligned data points")
    coeffs = np.polyfit(benchmark_aligned, portfolio_aligned, 1)
    return float(coeffs[0])


def alpha_regression(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Regression alpha: the intercept of ``Rp`` regressed on ``Rb``.

    The intercept is the portfolio's average per-period return when the
    benchmark's return is zero. Not annualized (unlike ``alpha_capm``).
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        raise ValueError("Cannot compute alpha: input series is empty.")

    if np.allclose(benchmark_returns, benchmark_returns.iloc[0]) or np.allclose(
        portfolio_returns, portfolio_returns.iloc[0]
    ):
        raise ValueError("Cannot compute alpha: no variation in returns.")

    benchmark_aligned, portfolio_aligned = benchmark_returns.align(
        portfolio_returns, join="inner"
    )
    if len(benchmark_aligned) < 3:
        raise ValueError("Too few aligned data points after trimming. Cannot compute alpha.")

    with warnings.catch_warnings():
        warnings.simplefilter("error", np.RankWarning)
        coeffs = np.polyfit(benchmark_aligned, portfolio_aligned, 1)

    return float(coeffs[1])  # intercept = alpha


def alpha_capm(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    fallback_to_simple_beta: bool = False,
) -> float:
    """Annualized Jensen's alpha via CAPM.

        alpha_per_period = mean(Rp) - [Rf + beta·(mean(Rb) - Rf)]
        alpha            = (1 + alpha_per_period) ** periods_per_year - 1

    ``beta`` comes from :func:`beta_capm`; if that raises and
    ``fallback_to_simple_beta`` is set, :func:`beta_regression` is used.
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        raise ValueError("Cannot compute alpha_capm: input series is empty.")

    benchmark_aligned, portfolio_aligned = benchmark_returns.align(
        portfolio_returns, join="inner"
    )
    if len(benchmark_aligned) < 3:
        raise ValueError("Too few aligned data points to compute alpha_capm.")

    try:
        beta = beta_capm(portfolio_returns, benchmark_returns, risk_free_rate=risk_free_rate)
    except Exception:
        if not fallback_to_simple_beta:
            raise
        beta = beta_regression(portfolio_returns, benchmark_returns)

    mean_portfolio = portfolio_aligned.mean()
    mean_benchmark = benchmark_aligned.mean()
    alpha_per_period = mean_portfolio - (risk_free_rate + beta * (mean_benchmark - risk_free_rate))
    return (1 + alpha_per_period) ** periods_per_year - 1
