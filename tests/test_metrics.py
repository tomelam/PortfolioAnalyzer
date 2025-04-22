import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from data_loader import align_portfolio_civs
from metrics_calculator import (
    calculate_portfolio_metrics,
    geometric_annualized_return,
)
import pickle


@pytest.mark.order(10)  # test_calculate_portfolio_metrics()
def test_calculate_portfolio_metrics():
    """Test calculation of portfolio metrics."""
    portfolio_returns = pd.DataFrame(
        {"ppf_value": [0.01, -0.02, 0.03]},
        index=pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-04"])
    )
    risk_free_rate = 0.02 / 252
    dummy_portfolio = {}
    metrics, drawdowns = calculate_portfolio_metrics(
        portfolio_returns,
        dummy_portfolio,
        risk_free_rate
    )

    assert "Annualized Return" in metrics
    assert "Sharpe Ratio" in metrics


@pytest.mark.order(11)
def test_geometric_annualized_return_from_navs():
    """
    CAGR should be exactly 10% when NAV grows at a constant rate over 252 business days.
    """
    # Simulate 253 NAVs → 252 returns (1 trading year)
    dates = pd.bdate_range(start="2020-01-01", periods=253)
    initial_nav = 100.0
    annual_return = 0.10
    daily_return = (1 + annual_return) ** (1 / 252) - 1  # constant daily compounding

    # Generate NAVs from daily compounding
    navs = pd.Series(
        [initial_nav * (1 + daily_return) ** i for i in range(len(dates))],
        index=dates
    )

    daily_returns = navs.pct_change().dropna()  # 252 returns
    computed = geometric_annualized_return(daily_returns)
    expected = 0.10

    assert abs(computed - expected) < 1e-10  # tighter threshold now justified
