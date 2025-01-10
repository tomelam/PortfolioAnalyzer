import pytest
import unittest
from unittest.mock import patch
import pandas as pd
from test_utils import mock_nav_data  # Import the mocked DataFrame
from data_loader import fetch_navs_of_mutual_fund

class TestFetchNavs(unittest.TestCase):
    @pytest.mark.order(5)
    @patch('requests.get')
    def test_fetch_navs_of_mutual_fund(self, mock_get):
        """Test fetching NAV data."""
        USE_MOCK_DATA = True  # Toggle between mock and real data

        if USE_MOCK_DATA:
            # Convert mock_nav_data to a JSON-like structure expected by requests.get().json()
            mock_response_data = {
                "data": [
                    {"date": date.strftime("%d-%m-%Y"), "nav": f"{nav:5f}"}
                    for date, nav in mock_nav_data["nav"].items()
                ]
            }

            # Configure the mock response
            mock_response = mock_get.return_value
            mock_response.json.return_value = mock_response_data
            mock_response.status_code = 200

            # Call the function being tested
            nav_data = fetch_navs_of_mutual_fund("https://mock.url")  # Call the mocked function
        else:
            # Use real API data
            nav_data = fetch_navs_of_mutual_fund("https://api.mfapi.in/mf/120586")

        # Debugging: Print processed DataFrame
        print("\n[DEBUG] Processed NAV Data (First 2 rows):")
        print(nav_data.head(2))
        print("\n[DEBUG] Processed NAV Data (Last 2 rows):")
        print(nav_data.tail(2))

        # Example assertions (adjust as needed based on real data)
        assert len(nav_data) > 0, "No data returned from API!"
        assert isinstance(nav_data.index[0], pd.Timestamp), "Index is not datetime64[ns]!"

        if USE_MOCK_DATA:
            # Validate the first 2 rows
            pd.testing.assert_frame_equal(nav_data.iloc[:2], mock_nav_data.iloc[:2])
