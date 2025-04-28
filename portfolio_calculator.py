import pandas as pd

"""
def calculate_portfolio_allocations(portfolio):
    aggregated = {"equity": 0.0, "debt": 0.0, "real_estate": 0.0, "commodities": 0.0, "cash": 0.0}

    if "funds" in portfolio:
        for fund in portfolio["funds"]:
            alloc = fund["allocation"]
            for asset_type in aggregated:
                aggregated[asset_type] += alloc * fund["asset_allocation"][asset_type] / 100.0

    for bond in ["ppf", "scss", "sgb", "rec_bond"]:
        if bond in portfolio:
            aggregated["debt"] += portfolio[bond].get("allocation", 0)

    if "gold" in portfolio:
        aggregated["commodities"] += portfolio["gold"].get("allocation", 0)

    return pd.Series(aggregated)
"""
def calculate_portfolio_allocations(portfolio) -> pd.Series:
    """
    Calculate allocation percentages by asset class across the entire portfolio.
    """
    allocations = {}

    for name, asset in portfolio.assets.items():
        if not hasattr(asset, "asset_allocation"):
            continue  # skip if asset does not have allocation breakdown
        fund_weight = portfolio.weights.get(name, 0.0)
        for asset_class, percent in asset.asset_allocation.items():
            allocations[asset_class] = allocations.get(asset_class, 0.0) + (percent / 100.0) * fund_weight

    return pd.Series(allocations)


def calculate_gain_daily_portfolio_series(
    portfolio,
    aligned_portfolio_civs,
    ppf_series=None,
    scss_series=None,
    rec_bond_series=None,
    sgb_series=None,
    gold_series=None,
):
    daily_returns_components = []

    # Funds (weighted correctly)
    if "funds" in portfolio and not aligned_portfolio_civs.empty:
        fund_weights = [fund["allocation"] for fund in portfolio["funds"]]
        funds_returns = aligned_portfolio_civs.pct_change().fillna(0).mul(fund_weights, axis=1).sum(axis=1)
        daily_returns_components.append(funds_returns.rename("funds"))

    # PPF
    if ppf_series is not None and "ppf" in portfolio:
        ppf_alloc = portfolio["ppf"]["allocation"]
        ppf_returns = ppf_series["ppf_value"].pct_change().fillna(0)
        daily_returns_components.append(ppf_returns.rename("ppf") * ppf_alloc)

    # SCSS
    if scss_series is not None and "scss" in portfolio:
        scss_alloc = portfolio["scss"]["allocation"]
        scss_returns = scss_series["var_rate_bond_value"].pct_change().fillna(0)
        daily_returns_components.append(scss_returns.rename("scss") * scss_alloc)

    # REC Bonds
    if rec_bond_series is not None and "rec_bond" in portfolio:
        rec_bond_alloc = portfolio["rec_bond"]["allocation"]
        rec_bond_returns = rec_bond_series["var_rate_bond_value"].pct_change().fillna(0)
        daily_returns_components.append(rec_bond_returns.rename("rec_bond") * rec_bond_alloc)

    # SGB (no daily_interest here; it's done in loader)
    if sgb_series is not None and "sgb" in portfolio:
        sgb_alloc = portfolio["sgb"]["allocation"]
        sgb_returns = sgb_series.iloc[:, 0]
        daily_returns_components.append(sgb_returns.rename("sgb") * sgb_alloc)

    # Gold
    if gold_series is not None and "gold" in portfolio:
        gold_alloc = portfolio["gold"]["allocation"]
        gold_returns = gold_series["Adjusted Spot Price"].pct_change().fillna(0)
        daily_returns_components.append(gold_returns.rename("gold") * gold_alloc)

    if not daily_returns_components:
        raise ValueError("No valid asset returns were found!")

    daily_returns = pd.concat(daily_returns_components, axis=1).fillna(0)
    daily_returns["portfolio_value"] = daily_returns.sum(axis=1)

    if daily_returns["portfolio_value"].abs().sum() == 0:
        raise ValueError("No valid asset returns were found!")

    return daily_returns[["portfolio_value"]]


def calculate_gains_cumulative(gain_daily_portfolio_series, gain_daily_benchmark_series):
    gain_cumulative_returns = (1 + gain_daily_portfolio_series).cumprod()
    if gain_daily_benchmark_series is not None:
        gain_cumulative_benchmark = (1 + gain_daily_benchmark_series).cumprod()
    else:
        gain_cumulative_benchmark = None

    return gain_cumulative_returns, gain_cumulative_benchmark
