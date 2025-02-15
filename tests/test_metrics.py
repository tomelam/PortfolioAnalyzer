import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from data_loader import align_portfolio_civs
from metrics_calculator import calculate_portfolio_metrics
from test_utils import load_json
import pickle


@pytest.mark.order(9)  # test_calculate_portfolio_metrics()
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
