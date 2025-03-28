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
            "allocation": 0.40,
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

mock_valid_ppf_toml = {
    "label": "PPF",
    "ppf": {
        "name": "Public Provident Fund",
        "allocation": 1.0,
        "ppf_interest_rates_file": "ppf_interest_rates.csv"
    }
}


@pytest.mark.order(1)  # test_valid_toml(mocker)
def test_valid_toml(mocker):
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("toml.load", return_value=mock_valid_toml)
    portfolio = load_portfolio_details("port-x.toml")

    assert "label" in portfolio
    assert portfolio["label"] == "Portfolio X: 60% NIFTY50 + 40% Bluechip"

    mocker.patch("toml.load", return_value=mock_valid_ppf_toml)
    portfolio = load_portfolio_details("port-ppf.toml")

    assert "label" in portfolio
    assert portfolio["label"] == "PPF"


@pytest.mark.order(2)  # test_invalid_toml_missing_top_level_key(mocker)
def test_invalid_toml_missing_top_level_key(mocker):
    """Test loading a TOML file missing top-level keys."""
    mocker.patch("os.path.exists", return_value=True)
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
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("toml.load", return_value=invalid_toml)

    # Use (?s) to allow '.' to match newline characters
    with pytest.raises(
        ValueError,
        match=r"(?s)TOML file errors detected:.*investment 'Fund A': Must be between 0 and 1"
    ):
        load_portfolio_details("port-x.toml")
