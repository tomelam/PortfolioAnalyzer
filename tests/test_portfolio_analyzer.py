import sys
import os
import requests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from test_utils import mock_portfolio_toml
from data_loader import load_portfolio_details, get_aligned_portfolio_civs
from metrics_calculator import calculate_portfolio_metrics
from stress_test import simulate_multiple_shocks

class TestPortfolioAnalyzer(unittest.TestCase):

    def test_load_portfolio(self):
        """Test loading of portfolio data from TOML file."""
        portfolio = load_portfolio_details("port-x.toml")
        self.assertIn("label", portfolio)
        self.assertIn("funds", portfolio)
        self.assertIsInstance(portfolio["funds"], list)

    def test_get_aligned_portfolio_civs(self):
        """Test alignment of CIV data."""
        fund_civs = {
            "FundA": pd.DataFrame({"nav": [100, 101]}, index=pd.to_datetime(["2023-01-01", "2023-01-02"])),
            "FundB": pd.DataFrame({"nav": [200, 202]}, index=pd.to_datetime(["2023-01-01", "2023-01-02"]))
        }
        aligned = get_aligned_portfolio_civs(portfolio)
        self.assertEqual(aligned.shape, (2, 2))

    def test_calculate_portfolio_metrics(self):
        """Test calculation of portfolio metrics."""
        portfolio_returns = pd.Series([0.01, -0.02, 0.03], index=pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-04"]))
        risk_free_rate = 0.02 / 252
        metrics, drawdowns = calculate_portfolio_metrics(portfolio_returns, risk_free_rate)
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
