from types import SimpleNamespace

import pandas as pd
import pytest

from portfolioanalyzer.portfolio_calculator import (
    calculate_gains_cumulative,
    calculate_portfolio_allocations,
)


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


def test_calculate_portfolio_allocations_skips_assets_without_breakdown():
    """Assets lacking an ``asset_allocation`` attribute (e.g. PPF, gold, SGB)
    contribute nothing to the asset-class breakdown rather than erroring."""
    fund = SimpleNamespace(asset_allocation={"equity": 100.0})
    gold = SimpleNamespace()  # no asset_allocation attribute → skipped
    portfolio = SimpleNamespace(
        assets={"Fund": fund, "Gold": gold},
        weights={"Fund": 0.7, "Gold": 0.3},
    )
    result = calculate_portfolio_allocations(portfolio)
    assert list(result.index) == ["equity"]
    assert result["equity"] == pytest.approx(0.7)  # only the fund's weight counts


def test_calculate_gains_cumulative_with_benchmark():
    port = pd.Series([0.10, 0.0, -0.05])
    bench = pd.Series([0.05, 0.05, 0.0])
    cum_port, cum_bench = calculate_gains_cumulative(port, bench)
    assert cum_port.tolist() == pytest.approx([1.10, 1.10, 1.045])
    assert cum_bench.tolist() == pytest.approx([1.05, 1.1025, 1.1025])


def test_calculate_gains_cumulative_without_benchmark():
    port = pd.Series([0.10, 0.0])
    cum_port, cum_bench = calculate_gains_cumulative(port, None)
    assert cum_port.tolist() == pytest.approx([1.10, 1.10])
    assert cum_bench is None
