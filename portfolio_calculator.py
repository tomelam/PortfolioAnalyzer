import pandas as pd

def calculate_portfolio_allocations(portfolio):
    aggregated = {"equity": 0.0, "debt": 0.0, "real_estate": 0.0, "commodities": 0.0, "cash": 0.0}

    if "funds" in portfolio:
        for fund in portfolio["funds"]:
            alloc = fund["allocation"]
            for asset_type in aggregated:
                aggregated[asset_type] += alloc * fund["asset_allocation"][asset_type] / 100.0

    for bond in ["ppf", "scss", "rec_bond"]:
        if bond in portfolio:
            aggregated["debt"] += portfolio[bond].get("allocation", 0)

    if "gold" in portfolio:
        aggregated["commodities"] += portfolio["gold"].get("allocation", 0)

    return pd.Series(aggregated)

def calculate_gain_daily_portfolio_series(portfolio, aligned_portfolio_civs,
                                          ppf_series=None, scss_series=None,
                                          rec_bond_series=None, gold_series=None):
    daily_returns_components = []

    # Funds
    if "funds" in portfolio and not aligned_portfolio_civs.empty:
        print("Funds daily returns head:", aligned_portfolio_civs.head())
        funds_returns = aligned_portfolio_civs.pct_change().fillna(0).mean(axis=1)
        daily_returns_components.append(funds_returns.rename("funds"))

    # PPF
    if ppf_series is not None:
        print("PPF series head:", ppf_series.head())
        ppf_returns = ppf_series.pct_change().fillna(0)
        daily_returns_components.append(ppf_returns.iloc[:, 0].rename("ppf"))

    # SCSS
    if scss_series is not None:
        print("SCSS series head:", scss_series.head())
        scss_returns = scss_series.pct_change().fillna(0)
        daily_returns_components.append(scss_returns.iloc[:, 0].rename("scss"))

    # TODO: Since we've added a fixed-rate bond to the calculation, don't refer to the
    # general calculations of bonds as being variable rate.
    # REC Bonds
    if rec_bond_series is not None:
        print("REC Bond series before pct_change:\n", rec_bond_series.head())
        rec_bond_returns = rec_bond_series["var_rate_bond_value"].pct_change().fillna(0)
        print("REC bond returns after pct_change:", rec_bond_returns.head(), rec_bond_returns.dropna().empty)
        daily_returns_components.append(rec_bond_returns.rename("rec_bond"))

    # Gold
    if gold_series is not None:
        print("Gold series head:", gold_series.head())
        gold_returns = gold_series["Adjusted Spot Price"].pct_change().fillna(0)
        daily_returns_components.append(gold_returns.rename("gold"))

    if not daily_returns_components:
        print("daily_returns_components length:", len(daily_returns_components))
        for i, comp in enumerate(daily_returns_components):
            print(f"Component {i} head:\n", comp.head())
        raise ValueError("No valid asset returns were found!")

    daily_returns = pd.concat(daily_returns_components, axis=1).fillna(0)
    daily_returns["portfolio_value"] = daily_returns.sum(axis=1)

    if daily_returns["portfolio_value"].abs().sum() == 0:
        raise ValueError("No valid asset returns were found!")

    print("daily_returns DataFrame head after combining:", daily_returns.head())
    return daily_returns[["portfolio_value"]]

def calculate_gains_cumulative(gain_daily_portfolio_series, gain_daily_benchmark_series):
    gain_cumulative_returns = (1 + gain_daily_portfolio_series).cumprod()
    gain_cumulative_benchmark = (1 + gain_daily_benchmark_series).cumprod()

    return gain_cumulative_returns, gain_cumulative_benchmark
