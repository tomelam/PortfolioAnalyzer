def calculate_portfolio_allocations(portfolio):
    import pandas as pd
    
    # Initialize the five classes at zero
    aggregated = {
        "equity": 0.0,
        "debt": 0.0,
        "real_estate": 0.0,
        "commodities": 0.0,
        "cash": 0.0
    }

    # If there are funds, aggregate their asset allocations
    if "funds" in portfolio:
        for fund in portfolio["funds"]:
            alloc = fund["allocation"]  # e.g. 0.25
            eq_pct = fund["asset_allocation"]["equity"] / 100.0
            dt_pct = fund["asset_allocation"]["debt"] / 100.0
            re_pct = fund["asset_allocation"]["real_estate"] / 100.0
            co_pct = fund["asset_allocation"]["commodities"] / 100.0
            cs_pct = fund["asset_allocation"]["cash"] / 100.0
            # Add them
            aggregated["equity"]      += alloc * eq_pct
            aggregated["debt"]        += alloc * dt_pct
            aggregated["real_estate"] += alloc * re_pct
            aggregated["commodities"] += alloc * co_pct
            aggregated["cash"]        += alloc * cs_pct

    # If there's ppf or scss, assume 100% debt.
    if "ppf" in portfolio:
         aggregated["debt"] += portfolio["ppf"].get("allocation", 0)
    if "scss" in portfolio:
         aggregated["debt"] += portfolio["scss"].get("allocation", 0)

    return pd.Series(aggregated)


def calculate_gain_daily_portfolio_series(portfolio, aligned_portfolio_civs, gold_series=None):
    """
    Calculate daily portfolio returns by combining returns from various assets,
    resulting in a single 'portfolio_value' column for the entire portfolio.
    """
    import pandas as pd

    # Start with an empty DataFrame or one indexed by aligned_portfolio_civs if present.
    if not aligned_portfolio_civs.empty:
        daily_returns = pd.DataFrame(index=aligned_portfolio_civs.index)
    else:
        daily_returns = pd.DataFrame()

    # ---------------------------------------------------
    # 1) Mutual Funds (if any)
    # ---------------------------------------------------
    funds_value = None
    if "funds" in portfolio and not aligned_portfolio_civs.empty:
        # Convert NAVs to daily returns, fill the initial NaN with 0
        funds_pct = aligned_portfolio_civs.pct_change().fillna(0)
        # Weighted sum of each fund by its allocation
        funds_alloc_sum = 0
        funds_return_series = pd.Series(0.0, index=funds_pct.index)
        for fund in portfolio["funds"]:
            alloc = fund["allocation"]
            funds_alloc_sum += alloc
            # Add (fund's daily return * its allocation) to the overall sum
            funds_return_series = funds_return_series.add(
                funds_pct[fund["name"]] * alloc, fill_value=0
            )
        # Put the final combined daily returns into a single column
        funds_value = funds_return_series.to_frame(name="funds_value")

    # ---------------------------------------------------
    # 2) PPF & SCSS -> single "var_bond_value" column
    # ---------------------------------------------------
    var_bond_value = None
    from bond_calculators import calculate_variable_bond_cumulative_gain

    # Helper function: convert a cumulative DataFrame to daily returns, fill first row with 0
    def cum_to_daily(cum_df, col_name):
        return cum_df[col_name].pct_change().fillna(0)

    # We'll accumulate PPF + SCSS daily returns into a single Series
    var_bond_series = pd.Series(0.0, index=daily_returns.index)

    # If PPF
    if "ppf" in portfolio:
        from data_loader import load_ppf_interest_rates
        ppf_rates = load_ppf_interest_rates(portfolio["ppf"]["ppf_interest_rates_file"])
        ppf_cum = calculate_variable_bond_cumulative_gain(ppf_rates, daily_returns.index[0] if not daily_returns.empty else pd.Timestamp("2000-01-01"))
        ppf_daily = cum_to_daily(ppf_cum, "var_rate_bond_value")
        ppf_alloc = portfolio["ppf"]["allocation"]
        var_bond_series = var_bond_series.add(ppf_daily * ppf_alloc, fill_value=0)

    # If SCSS
    if "scss" in portfolio:
        from data_loader import load_scss_interest_rates
        scss_rates = load_scss_interest_rates()
        scss_cum = calculate_variable_bond_cumulative_gain(scss_rates, daily_returns.index[0] if not daily_returns.empty else pd.Timestamp("2000-01-01"))
        scss_daily = cum_to_daily(scss_cum, "var_rate_bond_value")
        scss_alloc = portfolio["scss"]["allocation"]
        var_bond_series = var_bond_series.add(scss_daily * scss_alloc, fill_value=0)

    # If var_bond_series is non-zero, store it in a column
    if not var_bond_series.empty and var_bond_series.abs().sum() != 0:
        var_bond_value = var_bond_series.to_frame(name="var_bond_value")

    # ---------------------------------------------------
    # 3) Merge "funds_value" and "var_bond_value"
    # ---------------------------------------------------
    # If we have both, sum them into a single column
    # If we have just one, rename it to "portfolio_value"
    # If we have none, raise an error
    final_df = pd.DataFrame(index=daily_returns.index)  # ensures consistent index

    if funds_value is not None and var_bond_value is not None:
        # Merge them
        final_df["portfolio_value"] = funds_value["funds_value"].add(var_bond_value["var_bond_value"], fill_value=0)
    elif funds_value is not None:
        # Just funds
        final_df["portfolio_value"] = funds_value["funds_value"]
    elif var_bond_value is not None:
        # Just variable bonds
        final_df["portfolio_value"] = var_bond_value["var_bond_value"]
    else:
        raise ValueError("No valid asset returns were found! (No funds, PPF, or SCSS daily returns)")

    # ---------------------------------------------------
    # 4) If gold is present, add it in
    # ---------------------------------------------------
    if "gold" in portfolio and gold_series is not None:
        gold_daily = gold_series.pct_change().fillna(0)["Adjusted Spot Price"]
        gold_alloc = portfolio["gold"]["allocation"]
        # Weighted by gold allocation
        final_df["portfolio_value"] = final_df["portfolio_value"].add(gold_daily * gold_alloc, fill_value=0)

    # Check if final is still empty
    if final_df["portfolio_value"].abs().sum() == 0:
        raise ValueError("Error: No valid asset returns were found! Portfolio calculations failed.")

    return final_df


def calculate_gains_cumulative(gain_daily_portfolio_series, gain_daily_benchmark_series):
    if gain_daily_portfolio_series.empty or gain_daily_benchmark_series.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # DEBUGGING: Print first few rows of daily returns before cumulative gains
    print("\nDEBUG: First few rows of gain_daily_portfolio_series before .cumprod():")
    print(gain_daily_portfolio_series.head())
    #print("\nDEBUG: Last few rows of gain_daily_portfolio_series before .cumprod():")
    #print(gain_daily_portfolio_series.tail())

    '''
    # Convert daily returns to cumulative gains
    cum_port = (1 + gain_daily_portfolio_series).cumprod() - 1
    cum_bench = (1 + gain_daily_benchmark_series).cumprod() - 1

    return cum_port, cum_bench
    '''

    # Apply cumulative product
    gain_cumulative_returns = (1 + gain_daily_portfolio_series).cumprod()
    gain_cumulative_benchmark = (1 + gain_daily_benchmark_series).cumprod()

    # DEBUG: Print first few rows of computed cumulative gain
    print("\nDEBUG: First few rows of gain_cumulative_returns after .cumprod():")
    print(gain_cumulative_returns.head())
    print("\nDEBUG: First few rows of gain_cumulative_benchmark after .cumprod():")
    print(gain_cumulative_benchmark.head())

    return gain_cumulative_returns, gain_cumulative_benchmark
