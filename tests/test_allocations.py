import pytest
from metrics_calculator import calculate_portfolio_allocations


@pytest.mark.order(4)  # test_calculate_portfolio_allocations(mocker)
def test_calculate_portfolio_allocations(mocker):
    mock_fund_allocations = [
        {
            "name": "ICICI_Nifty_50_Index_Fund",
            "url": "https://api.mfapi.in/mf/120620",
            "allocation": 0.6,
            "equity": 99.89,
            "debt": 0.0,
            "real_estate": 0.0,
            "commodities": 0.0,
            "cash": 0.11,
        },
        {
            "name": "ICICI_Prudential_Bluechip_Fund",
            "url": "https://api.mfapi.in/mf/120586",
            "allocation": 0.4,
            "equity": 92.69,
            "debt": 0.35,
            "real_estate": 0.0,
            "commodities": 0.0,
            "cash": 6.96,
        },
    ]
    mock_expected_allocations = {"equity": 0.9701, "debt": 0.0014}

    mocker.patch(
        "test_utils.load_json",
        side_effect=[mock_fund_allocations, mock_expected_allocations],
    )

    calculated_portfolio_allocations = calculate_portfolio_allocations(
        mock_fund_allocations
    )

    for key in mock_expected_allocations:
        assert calculated_portfolio_allocations[key] == mock_expected_allocations[key]
