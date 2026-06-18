import pandas as pd


def calculate_portfolio_allocations(portfolio) -> pd.Series:
    """
    Calculate allocation percentages by asset class across the entire portfolio.
    """
    allocations: dict[str, float] = {}

    for name, asset in portfolio.assets.items():
        if not hasattr(asset, "asset_allocation"):
            continue  # skip if asset does not have allocation breakdown
        fund_weight = portfolio.weights.get(name, 0.0)
        for asset_class, percent in asset.asset_allocation.items():
            allocations[asset_class] = allocations.get(asset_class, 0.0) + (percent / 100.0) * fund_weight

    return pd.Series(allocations)


def calculate_gains_cumulative(gain_daily_portfolio_series, gain_daily_benchmark_series):
    gain_cumulative_returns = (1 + gain_daily_portfolio_series).cumprod()
    if gain_daily_benchmark_series is not None:
        gain_cumulative_benchmark = (1 + gain_daily_benchmark_series).cumprod()
    else:
        gain_cumulative_benchmark = None

    return gain_cumulative_returns, gain_cumulative_benchmark
