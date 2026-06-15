import pytest
import pandas as pd
from types import SimpleNamespace
from portfolio_calculator import calculate_portfolio_allocations


@pytest.mark.order(4)
def test_calculate_portfolio_allocations():
    """``calculate_portfolio_allocations`` aggregates each asset's
    ``asset_allocation`` mapping, weighted by its portfolio weight.

    The legacy mock used ``self.assets = {name: None}`` because an
    earlier API read from ``portfolio.funds`` (a list of dicts). The
    current API reads from ``portfolio.assets`` and pulls
    ``asset.asset_allocation``, so each mocked asset must expose that
    attribute.
    """

    class MockPortfolio:
        def __init__(self):
            nifty = SimpleNamespace(
                asset_allocation={
                    "equity": 99.89,
                    "debt": 0.0,
                    "real_estate": 0.0,
                    "commodities": 0.0,
                    "cash": 0.11,
                }
            )
            bluechip = SimpleNamespace(
                asset_allocation={
                    "equity": 92.69,
                    "debt": 0.35,
                    "real_estate": 0.0,
                    "commodities": 0.0,
                    "cash": 6.96,
                }
            )
            self.assets = {
                "ICICI_Nifty_50_Index_Fund": nifty,
                "ICICI_Prudential_Bluechip_Fund": bluechip,
            }
            self.weights = {
                "ICICI_Nifty_50_Index_Fund": 0.6,
                "ICICI_Prudential_Bluechip_Fund": 0.4,
            }

    expected = pd.Series(
        {
            "equity": 0.6 * 0.9989 + 0.4 * 0.9269,
            "debt": 0.6 * 0.0 + 0.4 * 0.0035,
            "real_estate": 0.0,
            "commodities": 0.0,
            "cash": 0.6 * 0.0011 + 0.4 * 0.0696,
        }
    )

    result = calculate_portfolio_allocations(MockPortfolio())

    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index(), rtol=1e-9)
