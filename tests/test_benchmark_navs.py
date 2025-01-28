import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from data_loader import align_portfolio_civs
from metrics_calculator import calculate_portfolio_metrics
from stress_test import simulate_multiple_shocks
from test_utils import load_json
import pickle


@pytest.mark.order(8)  # test_get_benchmark_navs(mocker)
def test_get_benchmark_navs(mocker):
    """Test getting of usefully indexed benchmark historical NAV using Yahoo Financ."""
    mock_get = mocker.patch('requests.get')

    mock_response_data = {
        "data": [
            {"date": date.strftime("%d-%m-%Y"), "nav": f"{nav:5f}"}
            for date, nav in mock_nav_data["nav"].items()
        ]
    }
    mock_response = mock_get.return_value
    mock_response.json.return_value = mock_response_data
    mock_response.status_code = 200

    nav_data = fetch_navs_of_mutual_fund("https://mock.url")

    assert len(nav_data) > 0, "No data returned from API!"
    assert isinstance(nav_data.index[0], pd.Timestamp), "Index is not datetime64[ns]!"
    pd.testing.assert_frame_equal(nav_data.iloc[:2], mock_nav_data.iloc[:2])
