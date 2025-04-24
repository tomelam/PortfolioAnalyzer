import pandas as pd
import pytest
from timeseries import TimeseriesFrame

# === Edge Case Extensions ===

@pytest.mark.order(24)
def test_max_drawdown_empty():
    """Verify that .max_drawdown() raises on empty series."""
    ts = TimeseriesFrame(pd.Series([], name="value"))
    with pytest.raises(ValueError, match="Max drawdown requires at least one data point"):
        ts.max_drawdown()

@pytest.mark.order(28)
def test_cagr_single_point():
    """CAGR should raise if fewer than two data points."""
    s = pd.Series([100.0], index=pd.to_datetime(["2022-01-01"]))
    ts = TimeseriesFrame(s.rename("value"))
    with pytest.raises(ValueError, match="requires at least two data points"):
        ts.cagr()

@pytest.mark.order(29)
def test_cagr_zero_start():
    """CAGR should raise if starting value is zero."""
    dates = pd.to_datetime(["2022-01-01", "2023-01-01"])
    ts = TimeseriesFrame(pd.Series([0.0, 100.0], index=dates, name="value"))
    with pytest.raises(ValueError, match="start value is zero"):
        ts.cagr()

@pytest.mark.order(30)
def test_volatility_empty():
    """Volatility should raise on empty series."""
    ts = TimeseriesFrame(pd.Series([], name="value"))
    with pytest.raises(ValueError, match="at least two data points"):
        ts.volatility()

@pytest.mark.order(31)
def test_sharpe_zero_volatility():
    """Sharpe should return +inf if returns are constant and positive."""
    s = pd.Series([0.01] * 100, index=pd.bdate_range("2023-01-01", periods=100))
    ts = TimeseriesFrame(s.rename("value"))
    result = ts.sharpe()
    assert result == float("inf")
