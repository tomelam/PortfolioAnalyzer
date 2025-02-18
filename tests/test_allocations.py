import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from portfolio_calculator import calculate_portfolio_allocations


@pytest.mark.order(4)  # test_calculate_portfolio_allocations(mocker)
def test_calculate_portfolio_allocations(mocker):
    portfolio = {
        'label': 'Portfolio X: 60% NIFTY50 + 40% Bluechip',
        'funds': [
            {
                'name': 'ICICI_Nifty_50_Index_Fund',
                'url': 'https://api.mfapi.in/mf/120620',
                'allocation': 0.6,
                'asset_allocation': {
                    'equity': 99.89,
                    'debt': 0.0,
                    'real_estate': 0.0,
                    'commodities': 0.0,
                    'cash': 0.11
                }
            },
            {
                'name': 'ICICI_Prudential_Bluechip_Fund',
                'url': 'https://api.mfapi.in/mf/120586',
                'allocation': 0.4,
                'asset_allocation': {
                    'equity': 92.69,
                    'debt': 0.35,
                    'real_estate': 0.0,
                    'commodities': 0.0,
                    'cash': 6.96
                }
            }
        ]
    }

    expected = pd.Series({
        'equity': 0.9701,
        'debt': 0.0014,
        'real_estate': 0.0000,
        'commodities': 0.0000,
        'cash': 0.0285
    })

    result = calculate_portfolio_allocations(portfolio)

    # Check that the result is a pandas Series
    assert isinstance(result, pd.Series), f"Result is not a pandas Series, got {type(result)}"

    # Check that the values and indices match the expected output
    assert_series_equal(result, expected, atol=1e-4)

