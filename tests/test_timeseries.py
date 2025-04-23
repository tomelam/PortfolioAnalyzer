import pandas as pd
import numpy as np
import pytest
from timeseries import TimeseriesFrame
from asset_timeseries import from_nav, AssetTimeseries

@pytest.mark.order(10)
def test_timeseriesframe_cagr_two_years():
    """
    TimeseriesFrame.annualized() should return 10% for NAVs growing from 100 to 121 in exactly 2 years.
    """
    dates = pd.to_datetime(["2022-01-01", "2024-01-01"])
    navs = pd.Series([100.0, 121.0], index=dates)
    frame = TimeseriesFrame(navs.rename("value"))

    result = frame.cagr()
    expected = 0.10

    assert abs(result - expected) < 1e-6


@pytest.mark.order(11)
def test_volatility_alternating_returns():
    dates = pd.bdate_range("2023-01-01", periods=252)
    returns = pd.Series([0.01 if i % 2 == 0 else -0.01 for i in range(252)], index=dates)
    frame = TimeseriesFrame(returns.rename("value"))
    
    expected = returns.std() * (252 ** 0.5)
    assert abs(frame.volatility() - expected) < 1e-10


@pytest.mark.order(12)
def test_volatility_with_normalized_sample():
    """
    Use a normalized return series with std dev = 1%.
    Test that TimeseriesFrame.volatility() returns 0.01 exactly.
    """
    np.random.seed(42)
    raw = np.random.normal(0.01, 0.01, 25200)

    # Normalize using sample std dev (ddof=1) to match Pandas default behavior
    normalized = (raw - raw.mean()) / raw.std(ddof=1)
    normalized = normalized * 0.01 + 0.01

    frame = TimeseriesFrame(pd.Series(normalized, index=pd.bdate_range("2023-01-01", periods=25200), name="value"))
    result = frame.volatility(periods_per_year=1)  # no annualization
    assert abs(result - 0.01) < 1e-6


@pytest.mark.order(13)
def test_sortino_positive_skew():
    dates = pd.bdate_range("2023-01-01", periods=5)
    returns = pd.Series([0.02, 0.01, -0.01, 0.03, -0.005], index=dates)
    frame = TimeseriesFrame(returns.rename("value"))
    
    result = frame.sortino(risk_free_rate=0.0)
    assert result > 0  # Should be positive since average return > downside deviation


@pytest.mark.order(14)
def test_sortino_ratio_no_downside():
    """
    Construct a return series with no negative excess returns.
    All values above the risk-free rate → Sortino ratio = ∞.
    """
    returns = np.full(252, 0.02)  # all returns = 2%
    dates = pd.bdate_range("2023-01-01", periods=252)
    frame = TimeseriesFrame(pd.Series(returns, index=dates, name="value"))

    result = frame.sortino(risk_free_rate=0.01, frequency="daily")
    assert result == float("inf")


@pytest.mark.order(15)
def test_sharpe_ratio_positive_returns():
    """
    Test Sharpe ratio with consistent positive excess returns and low volatility.
    """
    dates = pd.bdate_range("2023-01-01", periods=5)
    returns = pd.Series([0.01, 0.012, 0.009, 0.011, 0.010], index=dates)
    frame = TimeseriesFrame(returns.rename("value"))

    result = frame.sharpe(risk_free_rate=0.0)

    # Since the returns are tight and positive, Sharpe should be a large positive number
    assert result > 5


@pytest.mark.order(16)
def test_sharpe_ratio_exact_one():
    """
    Construct a return series where mean and std dev are both 1%,
    resulting in a Sharpe ratio of exactly 1.0 (ignoring annualization).
    """
    dates = pd.bdate_range("2023-01-01", periods=3)
    returns = pd.Series([0.0, 0.01, 0.02], index=dates)
    frame = TimeseriesFrame(returns.rename("value"))

    result = frame.sharpe(risk_free_rate=0.0, periods_per_year=1)
    assert abs(result - 1.0) < 1e-6


@pytest.mark.order(17)
def test_sharpe_ratio_long_series_mean_equals_std():
    """
    Use a longer synthetic return series (252 points) where mean ≈ std dev ≈ 1%.
    Sharpe ratio should be close to 1.0 when annualization is disabled.
    """
    np.random.seed(3)
    # Generate 252 daily returns from a normal distribution centered at 1% with std dev 1%
    returns = np.random.normal(loc=0.01, scale=0.01, size=25200)
    dates = pd.bdate_range(start="2023-01-01", periods=25200)
    series = pd.Series(returns, index=dates)
    frame = TimeseriesFrame(series.rename("value"))

    result = frame.sharpe(risk_free_rate=0.0, periods_per_year=1)
    print("Sample size:", len(returns))
    print("Mean:       ", returns.mean())
    print("Std dev:    ", returns.std())
    print("Sharpe:     ", result)
    print("Expected:   ", returns.mean() / returns.std())
    assert abs(result - 1.0) < 0.01  # Allow 10% margin for sampling noise


@pytest.mark.order(18)
def test_sharpe_ratio_with_normalized_sample():
    """
    Create a synthetic return series where mean and std dev are *exactly* 1%.
    This guarantees a Sharpe ratio of 1.0 (without annualization).
    
    We "normalize" the randomly generated returns by:
    1. Centering them to mean 0 and scaling to std dev 1
    2. Rescaling to mean = 0.01, std dev = 0.01
    """

    np.random.seed(42)
    raw = np.random.normal(0.01, 0.01, 25200)

    # Normalize: mean = 0.01, std dev = 0.01 (sample std, ddof=1)
    normalized = (raw - raw.mean()) / raw.std(ddof=1)
    normalized = normalized * 0.01 + 0.01

    dates = pd.bdate_range("2023-01-01", periods=25200)
    frame = TimeseriesFrame(pd.Series(normalized, index=dates, name="value"))

    result = frame.sharpe(risk_free_rate=0.0, periods_per_year=1)
    assert abs(result - 1.0) < 1e-6  # Very tight: test must be deterministic


@pytest.mark.order(21)
def test_from_nav_creates_asset_timeseries_correctly():
    """
    Validate from_nav() creates an AssetTimeseries with aligned Series.
    """
    dates = pd.date_range("2022-01-01", periods=5, freq="D")
    navs = pd.Series([100.0, 102.0, 101.0, 104.0, 106.0], index=dates)

    ts = from_nav(navs)

    # Ensure correct types
    assert isinstance(ts, AssetTimeseries)
    assert isinstance(ts.civ, TimeseriesFrame)
    assert isinstance(ts.ret, TimeseriesFrame)
    assert isinstance(ts.cumret, TimeseriesFrame)

    # Check index alignment
    assert (ts.civ.index == ts.ret.index).all()
    assert (ts.civ.index == ts.cumret.index).all()

    # Check value content
    assert ts.civ.value_series().iloc[0] == 100.0
    assert ts.civ.value_series().iloc[-1] == 106.0
    assert ts.ret.value_series().iloc[1] == pytest.approx(0.02)
    assert ts.cumret.value_series().iloc[-1] == pytest.approx((106/100))


@pytest.mark.order(22)
def test_asset_timeseries_metrics_cagr_and_sharpe():
    """
    Confirm that the AssetTimeseries object works with .cagr() and .sharpe() metrics.
    """
    # Create a 2-year NAV growth from 100 to 121 (implies CAGR ≈ 10%)
    dates = pd.to_datetime(["2020-01-01", "2022-01-01"])
    navs = pd.Series([100.0, 121.0], index=dates)

    ts = from_nav(navs)

    cagr = ts.civ.cagr()
    assert abs(cagr - 0.10) < 1e-6

    # Rebuild with enough data to make Sharpe meaningful
    navs_full = pd.Series(
        [100.0 * (1.0005) ** i for i in range(252)],
        index=pd.bdate_range(start="2023-01-01", periods=252)
    )
    ts_full = from_nav(navs_full)

    sharpe = ts_full.ret.sharpe(risk_free_rate=0.0, periods_per_year=1)
    assert sharpe > 0.0  # Loose sanity check


@pytest.mark.order(23)
def test_asset_timeseries_sortino_ratio():
    """
    Test .sortino() method on the return series of AssetTimeseries.
    Construct a series with steady positive returns and zero downside.
    """
    # 252 business days, 1% constant daily return (exceeds any realistic risk-free rate)
    returns = pd.Series(
        [0.01] * 252,
        index=pd.bdate_range("2023-01-01", periods=252)
    )
    navs = (1 + returns).cumprod()
    nav_series = pd.Series(navs.values, index=returns.index)

    ts = from_nav(nav_series)

    sortino = ts.ret.sortino(risk_free_rate=0.005, frequency="daily")
    print("Min excess return:", (ts.ret.value_series() - 0.005 / 252).min())
    print("Downside stddev:", (ts.ret.value_series() - 0.005 / 252)[(ts.ret.value_series() - 0.005 / 252) < 0].std())
    assert sortino > 0
    assert sortino != float("nan")
