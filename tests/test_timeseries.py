import pandas as pd
import numpy as np
import pytest
from timeseries import TimeseriesFrame

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

