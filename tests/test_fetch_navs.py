import pytest
import pandas as pd
from test_utils import mock_nav_data
from data_loader import fetch_navs_of_mutual_fund

@pytest.mark.order(5)
def test_fetch_navs_of_mutual_fund(mocker):
    """Test fetching NAV data."""
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
