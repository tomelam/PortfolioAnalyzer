import numpy as np
import pandas as pd
import pytest

import metrics
from asset_timeseries import AssetTimeseries, from_civ
from timeseries import TimeseriesReturn


@pytest.mark.order(10)
def test_timeseriesframe_cagr_two_years():
    """
    TimeseriesReturn.annualized() should return 10% for NAVs growing from 100 to 121 in exactly 2 years.
    """
    dates = pd.to_datetime(["2022-01-01", "2024-01-01"])
    navs = pd.Series([100.0, 121.0], index=dates)
    frame = TimeseriesReturn(navs.rename("value"))

    result = frame.cagr()
    expected = 0.10

    assert abs(result - expected) < 1e-4


@pytest.mark.order(11)
def test_volatility_alternating_returns():
    """Direct call to ``metrics.volatility`` on a return series.

    Rewritten from the legacy ``TimeseriesReturn(returns).volatility()``
    form, which treated the input as prices and applied ``pct_change``
    internally — incorrect when the caller already has returns.
    """
    dates = pd.bdate_range("2023-01-01", periods=252)
    returns = pd.Series([0.01 if i % 2 == 0 else -0.01 for i in range(252)], index=dates)

    expected = returns.std() * (252 ** 0.5)
    assert abs(metrics.volatility(returns, periods_per_year=252) - expected) < 1e-10


@pytest.mark.order(12)
def test_volatility_with_normalized_sample():
    """A return series normalized to std=1% must yield volatility 0.01
    when no annualization is applied."""
    np.random.seed(42)
    raw = np.random.normal(0.01, 0.01, 25200)

    normalized = (raw - raw.mean()) / raw.std(ddof=1)
    normalized = normalized * 0.01 + 0.01
    returns = pd.Series(normalized, index=pd.bdate_range("2023-01-01", periods=25200))

    assert abs(metrics.volatility(returns, periods_per_year=1) - 0.01) < 1e-6


@pytest.mark.order(13)
def test_sortino_positive_skew():
    """Returns whose mean exceeds their downside deviation → positive Sortino."""
    dates = pd.bdate_range("2023-01-01", periods=5)
    returns = pd.Series([0.02, 0.01, -0.01, 0.03, -0.005], index=dates)

    assert metrics.sortino(returns, risk_free_rate=0.0) > 0


@pytest.mark.order(14)
def test_sortino_ratio_no_downside():
    """No values fall below the risk-free rate → Sortino = +inf."""
    returns = pd.Series(
        np.full(252, 0.02),
        index=pd.bdate_range("2023-01-01", periods=252),
    )
    result = metrics.sortino(returns, risk_free_rate=0.01, periods_per_year=252)
    assert result == float("inf")


@pytest.mark.order(15)
def test_sharpe_ratio_positive_returns():
    """Tight, consistently positive returns → large positive Sharpe."""
    dates = pd.bdate_range("2023-01-01", periods=5)
    returns = pd.Series([0.01, 0.012, 0.009, 0.011, 0.010], index=dates)

    assert metrics.sharpe(returns, risk_free_rate=0.0, periods_per_year=252) > 5


@pytest.mark.order(16)
def test_sharpe_ratio_exact_one():
    """For returns where mean = std = 1%, unannualized Sharpe is exactly 1."""
    returns = pd.Series([0.0, 0.01, 0.02], index=pd.bdate_range("2023-01-01", periods=3))

    result = metrics.sharpe(returns, risk_free_rate=0.0, periods_per_year=1)
    assert abs(result - 1.0) < 1e-6


@pytest.mark.order(17)
def test_sharpe_ratio_long_series_mean_equals_std():
    """For a long sample with mean≈std≈1%, unannualized Sharpe ≈ 1."""
    np.random.seed(3)
    raw = np.random.normal(loc=0.01, scale=0.01, size=25200)
    returns = pd.Series(raw, index=pd.bdate_range("2023-01-01", periods=25200))

    result = metrics.sharpe(returns, risk_free_rate=0.0, periods_per_year=1)
    assert abs(result - 1.0) < 0.05


@pytest.mark.order(18)
def test_sharpe_ratio_with_normalized_sample():
    """Returns normalized to mean=std=1% give Sharpe exactly 1.0."""
    np.random.seed(42)
    raw = np.random.normal(0.01, 0.01, 25200)
    normalized = (raw - raw.mean()) / raw.std(ddof=1)
    normalized = normalized * 0.01 + 0.01
    returns = pd.Series(normalized, index=pd.bdate_range("2023-01-01", periods=25200))

    result = metrics.sharpe(returns, risk_free_rate=0.0, periods_per_year=1)
    assert abs(result - 1.0) < 1e-6


@pytest.mark.order(21)
def test_from_civ_creates_asset_timeseries_correctly():
    """
    Validate from_civ() creates an AssetTimeseries with aligned Series.
    """
    dates = pd.date_range("2022-01-01", periods=5, freq="D")
    navs = pd.Series([100.0, 102.0, 101.0, 104.0, 106.0], index=dates)

    ts = from_civ(navs)

    # Ensure correct types
    assert isinstance(ts, AssetTimeseries)
    assert isinstance(ts.civ, TimeseriesReturn)
    assert isinstance(ts.ret, TimeseriesReturn)
    assert isinstance(ts.cumret, TimeseriesReturn)

    # Check index alignment
    assert len(ts.ret.index) == len(ts.civ.index) - 1
    assert len(ts.cumret.index) == len(ts.civ.index) - 1

    # Check value content
    assert ts.civ.value_series().iloc[0] == 100.0
    assert ts.civ.value_series().iloc[-1] == 106.0
    assert ts.ret.value_series().iloc[0] == pytest.approx(0.02)
    assert ts.cumret.value_series().iloc[-1] == pytest.approx(106/100)


@pytest.mark.order(22)
def test_asset_timeseries_metrics_cagr_and_sharpe():
    """``AssetTimeseries.civ`` is price-like (cagr OK); ``ret`` is returns
    so we call ``metrics.sharpe`` directly on the underlying series."""
    # 2-year NAV growth 100 → 121 implies CAGR ≈ 10%.
    dates = pd.to_datetime(["2020-01-01", "2022-01-01"])
    navs = pd.Series([100.0, 121.0], index=dates)
    ts = from_civ(navs)

    assert abs(ts.civ.cagr() - 0.10) < 1e-4

    # 252 daily NAVs, +0.05%/day → daily mean ≈ 0.0005, std ≈ 0 → sharpe > 0.
    navs_full = pd.Series(
        [100.0 * (1.0005) ** i for i in range(252)],
        index=pd.bdate_range(start="2023-01-01", periods=252),
    )
    ts_full = from_civ(navs_full)

    # ts_full.ret is the *return* series, so use metrics.sharpe directly.
    sharpe = metrics.sharpe(ts_full.ret.value_series(), risk_free_rate=0.0, periods_per_year=1)
    assert sharpe > 0.0


@pytest.mark.order(23)
def test_asset_timeseries_sortino_ratio():
    """Steady +1% daily returns with risk-free 0.5% → no downside → +inf."""
    returns = pd.Series(
        [0.01] * 252,
        index=pd.bdate_range("2023-01-01", periods=252),
    )
    navs = (1 + returns).cumprod()
    nav_series = pd.Series(navs.values, index=returns.index)
    ts = from_civ(nav_series)

    # rf=0.5% daily would be massive; use per-period rate ≈ 0.5%/252 here
    # since we're passing the daily return series to metrics.sortino directly.
    sortino = metrics.sortino(
        ts.ret.value_series(), risk_free_rate=0.005 / 252, periods_per_year=252
    )
    assert sortino > 0
    assert sortino == float("inf")


@pytest.mark.order(40)
def test_alpha_capm_known_series():
    """
    A portfolio with 1% daily returns and benchmark with 0.5% daily returns
    should have a positive alpha_capm.
    """
    dates = pd.bdate_range(start="2023-01-01", periods=5)
    # Slightly rising returns
    port_returns = [0.009, 0.010, 0.011, 0.010, 0.012]
    bench_returns = [0.004, 0.005, 0.005, 0.006, 0.005]

    portfolio = TimeseriesReturn(pd.Series(port_returns, index=dates, name="value"))
    benchmark = TimeseriesReturn(pd.Series(bench_returns, index=dates, name="value"))

    alpha_capm = portfolio.alpha_capm(benchmark)
    assert alpha_capm > 0


@pytest.mark.order(41)
def test_alpha_capm_with_external_benchmark():
    """
    Alpha = Portfolio excess return - Beta × Benchmark excess return
    This test isolates the logic: Portfolio and Benchmark are independent.
    """

    dates = pd.bdate_range("2023-01-01", periods=5)
    
    # Simulate daily NAVs for the portfolio (steady growth)
    port_navs = pd.Series([100, 101, 102.01, 103.03, 104.06], index=dates)
    portfolio = from_civ(port_navs)

    # Simulate daily NAVs for the benchmark (slightly less growth)
    bench_navs = pd.Series([100, 100.5, 101.0, 101.5, 102.0], index=dates)
    benchmark = from_civ(bench_navs)

    alpha_capm = portfolio.ret.alpha_capm(benchmark.ret)
    beta = portfolio.ret.beta_capm(benchmark.ret)

    assert isinstance(alpha_capm, float)
    assert isinstance(beta, float)
    assert alpha_capm > 0    # Our synthetic portfolio outperformed the benchmark
    assert 0 < beta < 2  # Should be a reasonable positive correlation


@pytest.mark.order(42)
def test_beta_capm_known_series():
    """
    A portfolio that is exactly 2x the benchmark should have beta = 2.0.
    """
    dates = pd.bdate_range(start="2023-01-01", periods=5)
    bench_returns = np.random.normal(0.005, 0.001, size=5)
    port_returns = 2 * bench_returns

    bench = TimeseriesReturn(pd.Series(bench_returns, index=dates, name="value"))
    port = TimeseriesReturn(pd.Series(port_returns, index=dates, name="value"))

    beta = port.beta_capm(bench)
    assert abs(beta - 2.0) < 0.01


@pytest.mark.order(43)
def test_alpha_beta_capm_empty_inputs():
    """
    Empty return series should raise an error.
    """
    empty = TimeseriesReturn(pd.Series(dtype=float))
    with pytest.raises(ValueError):
        empty.alpha_capm(empty)
    with pytest.raises(ValueError):
        empty.beta_capm(empty)



@pytest.mark.order(44)
def test_alpha_regression_known_series():
    """A portfolio with consistently higher returns than the benchmark
    should have a positive regression alpha.

    NOTE: minimal coverage of ``alpha_regression`` — only the sign of the
    result is checked. Edge cases and robustness tracked in KANBAN.
    """
    dates = pd.bdate_range("2023-01-01", periods=5)
    port_returns = [0.012, 0.011, 0.013, 0.012, 0.014]
    bench_returns = [0.007, 0.006, 0.008, 0.007, 0.006]

    portfolio = TimeseriesReturn(pd.Series(port_returns, index=dates, name="value"))
    benchmark = TimeseriesReturn(pd.Series(bench_returns, index=dates, name="value"))

    alpha = portfolio.alpha_regression(benchmark)
    assert isinstance(alpha, float)
    assert alpha > 0


@pytest.mark.order(45)
def test_beta_regression_known_series():
    """A portfolio that is exactly 3x the benchmark should have regression beta = 3.

    NOTE: minimal coverage of ``beta_regression`` — tracked in KANBAN.
    """
    dates = pd.bdate_range("2023-01-01", periods=5)
    bench_returns = np.random.normal(0.005, 0.001, size=5)
    port_returns = 3 * bench_returns

    benchmark = TimeseriesReturn(pd.Series(bench_returns, index=dates, name="value"))
    portfolio = TimeseriesReturn(pd.Series(port_returns, index=dates, name="value"))

    beta = portfolio.beta_regression(benchmark)
    assert isinstance(beta, float)
    assert abs(beta - 3.0) < 0.01


@pytest.mark.order(50)
def test_max_drawdowns_basic():
    """
    Ensure .max_drawdowns() correctly identifies drawdown periods.
    """
    values = [100, 110, 105, 102, 108, 101, 100, 115]
    dates = pd.bdate_range("2023-01-01", periods=len(values))
    tsf = TimeseriesReturn(pd.Series(values, index=dates, name="value"))

    result = tsf.max_drawdowns(threshold=0.05)

    # There should be one drawdown exceeding 5%: from 110 → 100
    assert isinstance(result, list)
    assert len(result) == 1

    dd = result[0]
    assert dd["start_date"] == pd.Timestamp("2023-01-04")
    assert dd["trough_date"] == pd.Timestamp("2023-01-10")
    assert abs(dd["depth_pct"] + 9.09) < 0.2  # -9.09% drop (percent × 100 convention)
    assert dd["recovery_date"] == pd.Timestamp("2023-01-11")


@pytest.mark.order(51)
def test_max_drawdowns_behavioral():
    """
    Behavioral test: Confirm correct drawdown detection without relying on fixed dates.
    """
    values = [100, 110, 105, 102, 108, 101, 100, 115]  # Peak at 110 → trough at 100 → recovery at 115
    dates = pd.bdate_range("2023-01-01", periods=len(values))
    tsf = TimeseriesReturn(pd.Series(values, index=dates, name="value"))

    result = tsf.max_drawdowns(threshold=0.05)

    assert isinstance(result, list)
    assert len(result) == 1
    dd = result[0]

    expected_drop_pct = -((110 - 100) / 110) * 100
    assert abs(dd["depth_pct"] - expected_drop_pct) < 1e-6
    assert dd["trough_value"] == 100
    assert dd["recovery_value"] == 115


@pytest.mark.order(52)
def test_max_drawdowns_no_recovery():
    """
    If the series never fully recovers to a previous peak, no drawdown is recorded.
    """
    values = [100, 110, 108, 105, 102, 101]  # Never gets back to 110
    dates = pd.bdate_range("2023-02-01", periods=len(values))
    tsf = TimeseriesReturn(pd.Series(values, index=dates, name="value"))

    result = tsf.max_drawdowns(threshold=0.05)
    assert isinstance(result, list)
    assert len(result) == 0


@pytest.mark.order(53)
def test_max_drawdowns_none():
    """
    A steady upward trend should produce no drawdowns.
    """
    navs = pd.Series(np.linspace(100, 120, 20), index=pd.bdate_range("2023-01-01", periods=20))
    tsf = TimeseriesReturn(navs.rename("value"))

    result = tsf.max_drawdowns(threshold=5.0)
    assert result == []
