import pytest
import toml
from data_loader import load_portfolio_details

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
    ],
}


@pytest.mark.order(1)  # test_valid_toml(mocker)
def test_valid_toml(mocker):
    mocker.patch("toml.load", return_value=mock_valid_toml)
    portfolio = load_portfolio_details("port-x.toml")

    assert "label" in portfolio
    assert portfolio["label"] == "Portfolio X: 60% NIFTY50 + 40% Bluechip"


@pytest.mark.order(2)  # test_invalid_toml_missing_top_level_key(mocker)
def test_invalid_toml_missing_top_level_key(mocker):
    """Test loading a TOML file missing top-level keys."""
    mocker.patch("toml.load", return_value={"funds": mock_valid_toml["funds"]})

    with pytest.raises(ValueError, match="Missing required top-level key: 'label'"):
        load_portfolio_details("port-x.toml")


@pytest.mark.order(3)  # test_invalid_toml_invalid_fund_allocation(mocker)
def test_invalid_toml_invalid_fund_allocation(mocker):
    """Test loading a TOML file with an invalid fund allocation."""
    invalid_toml = {
        "label": "Portfolio X",
        "funds": [
            {
                "name": "Fund A",
                "url": "https://api.mfapi.in/mf/120620",
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
    mocker.patch("toml.load", return_value=invalid_toml)

    with pytest.raises(ValueError, match="Invalid allocation value for fund 'Fund A'"):
        load_portfolio_details("port-x.toml")
