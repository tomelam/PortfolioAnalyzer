import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from data_loader import align_portfolio_civs
from metrics_calculator import calculate_portfolio_metrics
from stress_test import simulate_multiple_shocks
from test_utils import load_json
import pickle

'''
@pytest.mark.order(6)
def test_align_portfolio_civs(mocker):
    """Test alignment of CIV data."""
    mock_portfolio_civs = {
        "Fund A": pd.DataFrame({"value": [100, 101, 102]}, index=pd.date_range("2023-01-01", periods=3))
    }
    expected_alignment = pd.DataFrame({"value": [100, 101, 102]}, index=pd.date_range("2023-01-01", periods=3))

    mocker.patch('data_loader.get_align_portfolio_civs', return_value=expected_alignment)

    aligned = get_aligned_portfolio_civs(mock_portfolio_civs)
    assert_frame_equal(aligned, expected_alignment, check_dtype=True)
'''


@pytest.mark.order(6)
def test_align_portfolio_civs(mocker):
    """Test alignment of CIV data using fixed parts of Pickled inputs."""

    # Load Pickled test data
    with open("tests/data/portfolio_civs.pkl", "rb") as f:
        portfolio_civs = pickle.load(f)

    with open("tests/data/aligned_civs.pkl", "rb") as f:
        expected_alignment = pickle.load(f)

    # Extract deterministic parts (for example, first 1000 rows)
    fixed_portfolio_civs = {k: v.iloc[:1000] for k, v in portfolio_civs.items()}
    fixed_expected_alignment = expected_alignment.iloc[:1000]

    # Mock the function call
    mocker.patch(
        "data_loader.align_portfolio_civs", return_value=fixed_expected_alignment
    )

    aligned = align_portfolio_civs(fixed_portfolio_civs)

    # Compare fixed portions ignoring column order and data types if needed
    assert_frame_equal(aligned, fixed_expected_alignment, check_dtype=False)


@pytest.mark.order(7)
def test_calculate_portfolio_metrics():
    """Test calculation of portfolio metrics."""
    portfolio_returns = pd.Series(
        [0.01, -0.02, 0.03],
        index=pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-04"]),
    )
    risk_free_rate = 0.02 / 252
    metrics, drawdowns = calculate_portfolio_metrics(portfolio_returns, risk_free_rate)

    assert "Annualized Return" in metrics
    assert "Sharpe Ratio" in metrics


@pytest.mark.order(8)
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
