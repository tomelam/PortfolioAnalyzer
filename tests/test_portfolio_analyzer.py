import sys
import os
import requests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from data_loader import load_portfolio, fetch_nav_data, align_fund_data
from metrics_calculator import calculate_metrics
from stress_test import simulate_multiple_shocks

class TestPortfolioAnalyzer(unittest.TestCase):

    def test_load_portfolio(self):
        """Test loading of portfolio data from TOML file."""
        portfolio = load_portfolio("port-x.toml")
        self.assertIn("label", portfolio)
        self.assertIn("funds", portfolio)
        self.assertIsInstance(portfolio["funds"], list)

    USE_MOCK_DATA = True  # Set this to False to use real data

    def test_fetch_nav_data(self):
        """Test fetching NAV data."""
        USE_MOCK_DATA = False  # Toggle between mock and real data

        if USE_MOCK_DATA:
            # Use patch as a context manager for mocking
            with patch("data_loader.requests.get") as mock_get:
                # Mocked API response data
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "data": [
                        {"date": "2013-01-02", "nav": 18.66},
                        {"date": "2013-01-03", "nav": 18.73},
                        # ... include intermediate dates as needed
                        {"date": "2025-01-06", "nav": 113.40},
                        {"date": "2025-01-07", "nav": 113.91},
                    ]
                }
                mock_get.return_value = mock_response

                nav_data = fetch_nav_data("http://example.com/api")
        else:
            # Use real API data
            url = "https://api.mfapi.in/mf/120586"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # Debugging: Log raw API response
            print("\n[DEBUG] Raw API Response:")
            raw_data = response.json()
            #print(raw_data)

            # Call fetch_nav_data to process the response
            nav_data = fetch_nav_data(url)

        # Debugging: Print processed DataFrame
        print("\n[DEBUG] Processed NAV Data (First 2 rows):")
        print(nav_data.head(2))
        print("\n[DEBUG] Processed NAV Data (Last 2 rows):")
        print(nav_data.tail(2))

        # Example assertions (adjust as needed based on real data)
        assert len(nav_data) > 0, "No data returned from API!"
        assert isinstance(nav_data.index[0], pd.Timestamp), "Index is not datetime64[ns]!"

        if USE_MOCK_DATA:
            # Expected DataFrame for mocked data
            expected_first_two = pd.DataFrame(
                {"nav": [100.0, 101.0]},
                index=pd.to_datetime(["2023-01-01", "2023-01-02"]),
            )
            expected_first_two.index.name = "date"

            # Validate the first 2 rows
            pd.testing.assert_frame_equal(nav_data.iloc[:2], expected_first_two)

    def test_align_fund_data(self):
        """Test alignment of NAV data."""
        fund_navs = {
            "FundA": pd.DataFrame({"nav": [100, 101]}, index=pd.to_datetime(["2023-01-01", "2023-01-02"])),
            "FundB": pd.DataFrame({"nav": [200, 202]}, index=pd.to_datetime(["2023-01-01", "2023-01-02"]))
        }
        aligned = align_fund_data(fund_navs)
        self.assertEqual(aligned.shape, (2, 2))

    def test_calculate_metrics(self):
        """Test calculation of portfolio metrics."""
        portfolio_returns = pd.Series([0.01, -0.02, 0.03], index=pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-04"]))
        risk_free_rate = 0.02 / 252
        metrics, drawdowns = calculate_metrics(portfolio_returns, risk_free_rate)
        self.assertIn("Annualized Return", metrics)
        self.assertIn("Sharpe Ratio", metrics)

    def test_simulate_multiple_shocks(self):
        """Test shock scenario simulation."""
        shock_scenarios = [
            {"name": "Mild Shock", "equity_shock": -0.05, "debt_shock": -0.02, "shock_day": 1, "projection_days": 5, "plot_color": "orange"}
        ]
        portfolio_allocations = {"equity": 0.6, "debt": 0.3, "cash": 0.1}
        result = simulate_multiple_shocks("Test Portfolio", 100, 0.1, portfolio_allocations, shock_scenarios)
        self.assertIn("Mild Shock", result)
        self.assertTrue(len(result["Mild Shock"][1]) > 0)

if __name__ == "__main__":
    unittest.main()
