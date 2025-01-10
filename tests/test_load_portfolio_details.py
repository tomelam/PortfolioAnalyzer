import pytest
import unittest
from unittest.mock import patch
import toml
from data_loader import load_portfolio_details

# Valid TOML data for testing
mock_valid_toml = {
    "label": "Portfolio X: 60% NIFTY50 + 40% Bluechip",
    "funds": [
        {
            "name": "ICICI_Nifty_50_Index_Fund",
            "url": "https://api.mfapi.in/mf/120620",
            "allocation": 0.60,
            "asset_allocation": {
                "equity": 99.89,
                "debt": 0.00,
                "real_estate": 0.00,
                "commodities": 0.00,
                "cash": 0.11,
            },
        },
        {
            "name": "ICICI_Prudential_Bluechip_Fund",
            "url": "https://api.mfapi.in/mf/120586",
            "allocation": 0.40,
            "asset_allocation": {
                "equity": 92.69,
                "debt": 0.35,
                "real_estate": 0.00,
                "commodities": 0.00,
                "cash": 6.96,
            },
        },
    ],
}

class TestLoadPortfolioDetails(unittest.TestCase):
    @pytest.mark.order(1)
    @patch('toml.load')  # Mock toml.load
    def test_valid_toml(self, mock_toml_load):
        """Test loading a valid TOML file."""
        mock_toml_load.return_value = mock_valid_toml
        portfolio = load_portfolio_details("port-x.toml")
        
        # Assertions for valid data
        self.assertIn("label", portfolio)
        self.assertIn("funds", portfolio)
        self.assertIsInstance(portfolio["funds"], list)
        self.assertEqual(portfolio["label"], "Portfolio X: 60% NIFTY50 + 40% Bluechip")

    @pytest.mark.order(2)
    @patch('toml.load')  # Mock toml.load
    def test_invalid_toml_missing_top_level_key(self, mock_toml_load):
        """Test loading a TOML file missing top-level keys."""
        # Simulate a missing "label" in the TOML data
        mock_toml_load.return_value = {
            "funds": mock_valid_toml["funds"]
        }
        with self.assertRaises(ValueError) as context:
            load_portfolio_details("port-x.toml")
        self.assertEqual(str(context.exception), "Missing required top-level key: 'label'")

    @pytest.mark.order(3)
    @patch('toml.load')  # Mock toml.load
    def test_invalid_toml_invalid_fund_allocation(self, mock_toml_load):
        """Test loading a TOML file with an invalid fund allocation."""
        # Simulate an invalid "allocation" value
        invalid_toml = {
            "label": "Portfolio X",
            "funds": [
                {
                    "name": "Fund A",
                    "url": "https://example.com",
                    "allocation": 1.5,  # Invalid allocation > 1
                    "asset_allocation": {
                        "equity": 50.0,
                        "debt": 50.0,
                        "real_estate": 0.0,
                        "commodities": 0.0,
                        "cash": 0.0,
                    },
                }
            ],
        }
        mock_toml_load.return_value = invalid_toml
        with self.assertRaises(ValueError) as context:
            load_portfolio_details("port-x.toml")
        self.assertEqual(
            str(context.exception),
            "Invalid allocation value for fund 'Fund A': Must be between 0 and 1"
        )

    @pytest.mark.order(4)
    @patch('toml.load')  # Mock toml.load
    def test_invalid_toml_asset_allocation_missing_key(self, mock_toml_load):
        """Test loading a TOML file with missing keys in asset_allocation."""
        # Simulate missing "equity" in asset_allocation
        invalid_toml = {
            "label": "Portfolio X",
            "funds": [
                {
                    "name": "Fund A",
                    "url": "https://example.com",
                    "allocation": 0.5,
                    "asset_allocation": {
                        "debt": 50.0,  # Missing "equity"
                        "real_estate": 0.0,
                        "commodities": 0.0,
                        "cash": 0.0,
                    },
                }
            ],
        }
        mock_toml_load.return_value = invalid_toml
        with self.assertRaises(ValueError) as context:
            load_portfolio_details("port-x.toml")
        self.assertEqual(
            str(context.exception),
            "Missing key in 'asset_allocation' for fund 'Fund A': 'equity'"
        )
