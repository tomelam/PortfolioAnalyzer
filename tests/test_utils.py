# tests/test_utils.py
import pandas as pd

# Mock data mimicking the structure of the TOML data
mock_portfolio_toml = {
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
                "cash": 0.11
            }
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
                "cash": 6.96
            }
        }
    ]
}

mock_nav_data = pd.DataFrame(
    {"nav": [100.0, 101.0, 102.0]},
    index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
)
mock_nav_data.index.name = "date"
