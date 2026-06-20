"""Pure-function metric tests.

Each function takes a Series and returns a scalar (or list, for the
drawdown-list case). They contain no I/O, no class state, no implicit
price→return conversion — what you pass is what you get.

Test cases are mostly hand-computed: one-shot inputs whose answer is
obvious by inspection. Together they pin the math layer that the
existing TimeseriesReturn methods will delegate to.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from portfolioanalyzer import metrics
from portfolioanalyzer.metrics import (
    alpha_capm,
    alpha_regression,
    beta_capm,
    beta_regression,
    cagr,
    max_drawdown,
    max_drawdowns,
    sharpe,
    sortino,
    volatility,
)

# ---- cagr ---------------------------------------------------------------


def test_cagr_doubles_in_one_year() -> None:
    # 365 calendar days = 1.0 years under the 365.25 convention; close enough.
    prices = pd.Series(
        [100.0, 200.0],
        index=pd.to_datetime(["2023-01-01", "2024-01-01"]),
    )
    # 100 → 200 in ~one year ⇒ CAGR ≈ 1.0 (100%).
    assert cagr(prices) == pytest.approx(1.0, rel=1e-2)


def test_cagr_known_two_year_return() -> None:
    # 100 → 121 over 2 years ⇒ CAGR = 10%.
    prices = pd.Series(
        [100.0, 121.0],
        index=pd.to_datetime(["2023-01-01", "2025-01-01"]),
    )
    assert cagr(prices) == pytest.approx(0.10, abs=1e-3)


def test_cagr_requires_two_points() -> None:
    s = pd.Series([100.0], index=pd.to_datetime(["2024-01-01"]))
    with pytest.raises(ValueError, match="two data points"):
        cagr(s)


def test_cagr_zero_start_value_raises() -> None:
    prices = pd.Series(
        [0.0, 100.0],
        index=pd.to_datetime(["2024-01-01", "2025-01-01"]),
    )
    with pytest.raises(ValueError, match="start value"):
        cagr(prices)


# ---- volatility ---------------------------------------------------------


def test_volatility_constant_returns_zero() -> None:
    returns = pd.Series([0.01] * 10, index=pd.bdate_range("2024-01-01", periods=10))
    assert volatility(returns, periods_per_year=1) == pytest.approx(0.0, abs=1e-12)


def test_volatility_known_std() -> None:
    returns = pd.Series([0.0, 0.01, 0.02], index=pd.bdate_range("2024-01-01", periods=3))
    # std (ddof=1) of [0, 0.01, 0.02] = 0.01.
    assert volatility(returns, periods_per_year=1) == pytest.approx(0.01, abs=1e-9)


def test_volatility_annualizes_by_sqrt_n() -> None:
    returns = pd.Series([0.0, 0.01, 0.02], index=pd.bdate_range("2024-01-01", periods=3))
    daily_vol = 0.01
    assert volatility(returns, periods_per_year=252) == pytest.approx(daily_vol * math.sqrt(252))


# ---- sharpe -------------------------------------------------------------


def test_sharpe_exactly_one() -> None:
    # mean = 0.01, std = 0.01, rf = 0 ⇒ Sharpe = 1.0
    returns = pd.Series([0.0, 0.01, 0.02], index=pd.bdate_range("2024-01-01", periods=3))
    assert sharpe(returns, risk_free_rate=0.0, periods_per_year=1) == pytest.approx(1.0, abs=1e-6)


def test_sharpe_zero_volatility_infinite_when_positive() -> None:
    returns = pd.Series([0.01] * 10, index=pd.bdate_range("2024-01-01", periods=10))
    assert sharpe(returns, risk_free_rate=0.0, periods_per_year=1) == float("inf")


def test_sharpe_zero_volatility_zero_when_zero_mean() -> None:
    returns = pd.Series([0.0] * 10, index=pd.bdate_range("2024-01-01", periods=10))
    assert sharpe(returns, risk_free_rate=0.0, periods_per_year=1) == 0.0


def test_sharpe_annualized() -> None:
    returns = pd.Series([0.0, 0.01, 0.02], index=pd.bdate_range("2024-01-01", periods=3))
    daily_sharpe = 1.0
    assert sharpe(returns, risk_free_rate=0.0, periods_per_year=252) == pytest.approx(
        daily_sharpe * math.sqrt(252)
    )


# ---- sortino ------------------------------------------------------------


def test_sortino_no_downside_is_infinite() -> None:
    """If no returns are negative, downside std is 0; positive mean ⇒ +inf."""
    returns = pd.Series([0.01, 0.02, 0.03], index=pd.bdate_range("2024-01-01", periods=3))
    assert sortino(returns, risk_free_rate=0.0, periods_per_year=1) == float("inf")


def test_sortino_known_downside() -> None:
    # Mix of -0.01, 0.01, 0.02. Downside (< 0) = [-0.01], std ~ NaN (1 sample).
    # If there's exactly one downside observation pandas returns NaN for std;
    # the function must handle that without crashing.
    returns = pd.Series(
        [-0.01, 0.01, 0.02], index=pd.bdate_range("2024-01-01", periods=3)
    )
    result = sortino(returns, risk_free_rate=0.0, periods_per_year=1)
    # We don't assert a specific value — just non-NaN.
    assert not math.isnan(result)


# ---- max_drawdown -------------------------------------------------------


def test_max_drawdown_simple_dip() -> None:
    # Prices 100 → 50 → 100 → 200. Min drawdown = -0.5 at the trough.
    prices = pd.Series(
        [100.0, 50.0, 100.0, 200.0],
        index=pd.bdate_range("2024-01-01", periods=4),
    )
    assert max_drawdown(prices) == pytest.approx(-0.5, abs=1e-9)


def test_max_drawdown_monotone_up_is_zero() -> None:
    prices = pd.Series(
        [100.0, 110.0, 120.0, 130.0],
        index=pd.bdate_range("2024-01-01", periods=4),
    )
    assert max_drawdown(prices) == pytest.approx(0.0, abs=1e-9)


def test_max_drawdown_empty_raises() -> None:
    with pytest.raises(ValueError):
        max_drawdown(pd.Series([], dtype=float))


# ---- max_drawdowns (the threshold + full-recovery version) --------------


def test_max_drawdowns_finds_single_recovered_drawdown() -> None:
    # 100 → 80 (down 20%) → 100 (recovered). One drawdown above the 5% threshold.
    prices = pd.Series(
        [100.0, 80.0, 100.0, 110.0],
        index=pd.bdate_range("2024-01-01", periods=4),
    )
    result = max_drawdowns(prices, threshold=0.05)
    assert len(result) == 1
    drawdown_pct = result[0]["drawdown"]
    assert drawdown_pct == pytest.approx(0.20, abs=1e-9)


def test_max_drawdowns_ignores_below_threshold() -> None:
    # 100 → 99 (1%) → 100 — below 5% threshold.
    prices = pd.Series(
        [100.0, 99.0, 100.0],
        index=pd.bdate_range("2024-01-01", periods=3),
    )
    assert max_drawdowns(prices, threshold=0.05) == []


def test_max_drawdowns_skips_unrecovered() -> None:
    """A drawdown that never recovers should be omitted (per docstring)."""
    prices = pd.Series(
        [100.0, 50.0, 60.0, 65.0],
        index=pd.bdate_range("2024-01-01", periods=4),
    )
    # Series never returns to 100 ⇒ no fully-recovered drawdown.
    assert max_drawdowns(prices, threshold=0.05) == []


def test_max_drawdowns_empty_input_returns_empty() -> None:
    assert max_drawdowns(pd.Series([], dtype=float), threshold=0.05) == []


# ---- alpha / beta (return series in, scalar out) ------------------------

_BENCH = pd.Series(
    [0.01, -0.02, 0.03, 0.0, 0.015, -0.01],
    index=pd.bdate_range("2024-01-01", periods=6),
)


def test_beta_capm_two_x_benchmark_is_two() -> None:
    # A portfolio that is exactly 2× the benchmark has CAPM beta = 2.
    port = 2.0 * _BENCH
    assert beta_capm(port, _BENCH) == pytest.approx(2.0, abs=1e-9)


def test_beta_capm_empty_raises() -> None:
    empty = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="input series is empty"):
        beta_capm(empty, empty)


def test_beta_capm_too_few_aligned_raises() -> None:
    two = pd.Series([0.01, 0.02], index=pd.bdate_range("2024-01-01", periods=2))
    with pytest.raises(ValueError, match="Too few aligned data points to compute beta_capm"):
        beta_capm(two, two)


def test_beta_regression_three_x_benchmark_is_three() -> None:
    # A portfolio that is exactly 3× the benchmark has regression beta = 3.
    port = 3.0 * _BENCH
    assert beta_regression(port, _BENCH) == pytest.approx(3.0, abs=1e-9)


def test_beta_regression_too_few_points_raises() -> None:
    one = pd.Series([0.01], index=pd.bdate_range("2024-01-01", periods=1))
    with pytest.raises(ValueError, match="at least two aligned data points"):
        beta_regression(one, one)


def test_alpha_regression_exact_line_has_zero_intercept() -> None:
    # port = 2×bench exactly ⇒ regression line passes through origin ⇒ alpha ≈ 0.
    port = 2.0 * _BENCH
    assert alpha_regression(port, _BENCH) == pytest.approx(0.0, abs=1e-9)


def test_alpha_regression_empty_raises() -> None:
    empty = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="input series is empty"):
        alpha_regression(empty, empty)


def test_alpha_regression_no_variation_raises() -> None:
    flat = pd.Series([0.01] * 5, index=pd.bdate_range("2024-01-01", periods=5))
    with pytest.raises(ValueError, match="no variation"):
        alpha_regression(flat, flat)


def test_alpha_capm_identical_series_is_zero() -> None:
    # port == bench ⇒ beta = 1 and mean(Rp) - (Rf + beta·(mean(Rb) - Rf)) = 0.
    assert alpha_capm(_BENCH, _BENCH) == pytest.approx(0.0, abs=1e-9)


def test_alpha_capm_empty_raises() -> None:
    empty = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="input series is empty"):
        alpha_capm(empty, empty)


def test_alpha_capm_falls_back_to_regression_beta(monkeypatch) -> None:
    # When beta_capm raises and fallback is enabled, alpha_capm uses
    # beta_regression instead and still returns a finite number.
    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(metrics, "beta_capm", boom)
    result = alpha_capm(2.0 * _BENCH, _BENCH, fallback_to_simple_beta=True)
    assert math.isfinite(result)
