import pytest
import pandas as pd
from portfolio_calculator import calculate_portfolio_allocations

@pytest.mark.order(4)
def test_calculate_portfolio_allocations():
    """
    Test asset allocation calculation with mocked portfolio.
    """

    class MockPortfolio:
        def __init__(self):
            self.assets = {
                'ICICI_Nifty_50_Index_Fund': None,
                'ICICI_Prudential_Bluechip_Fund': None,
            }
            self.weights = {
                'ICICI_Nifty_50_Index_Fund': 0.6,
                'ICICI_Prudential_Bluechip_Fund': 0.4,
            }
            self.funds = [
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
            self.label = 'Portfolio X: 60% NIFTY50 + 40% Bluechip'

    portfolio = MockPortfolio()

    expected = pd.Series({
        'equity': 0.9701,
        'debt': 0.0014,
        'real_estate': 0.0000,
        'commodities': 0.0000,
        'cash': 0.0285
    })

    result = calculate_portfolio_allocations(portfolio)

    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index(), rtol=1e-4)
