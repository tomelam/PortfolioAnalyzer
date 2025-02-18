import pytest
import pandas as pd
#from test_utils import mock_nav_data
from data_loader import fetch_navs_of_mutual_fund


@pytest.mark.order(5)  # test_fetch_navs_of_mutual_funds
def test_fetch_navs_of_mutual_fund(mocker):
    """Test fetching NAV data."""
    mock_get = mocker.patch('requests.get')

    mock_nav_data = pd.DataFrame(
        {"nav": [100.0, 101.0, 102.0]},
        index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
    )
    mock_nav_data.index.name = "date"

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
