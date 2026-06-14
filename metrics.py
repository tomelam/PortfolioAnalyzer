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
