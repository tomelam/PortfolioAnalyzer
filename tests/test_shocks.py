import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from data_loader import align_portfolio_civs
from metrics_calculator import calculate_portfolio_metrics
from stress_test import simulate_multiple_shocks
from test_utils import load_json
import pickle


@pytest.mark.order(10)  # test_simulate_multiple_shocks()
def test_simulate_multiple_shocks():
    """Test shock scenario simulation."""
    shock_scenarios = [
        {
            "name": "Mild Shock",
            "equity_shock": -0.05,
            "debt_shock": -0.02,
            "shock_day": 1,
            "projection_days": 5,
            "plot_color": "orange",
        }
    ]
    portfolio_allocations = {"equity": 0.6, "debt": 0.3, "cash": 0.1}
    result = simulate_multiple_shocks(
        "Test Portfolio", 100, 0.1, portfolio_allocations, shock_scenarios
    )

    assert "Mild Shock" in result
    assert len(result["Mild Shock"][1]) > 0
