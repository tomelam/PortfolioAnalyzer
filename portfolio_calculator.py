import pandas as pd

import pandas as pd

def calculate_portfolio_allocations(portfolio):
    """
    Calculate the portfolio's aggregate asset allocations for funds.
    
    For each fund, the asset percentages (e.g. 'equity', 'debt', etc.)
    are expected to be provided as top-level keys in the fund dict.
    The allocation from each fund is weighted by the fund's overall allocation,
    with the percentage divided by 100.
    
    Returns a Series with the aggregated asset allocations.
    """
    if "funds" in portfolio:
        # Define the asset types we expect
        asset_types = ["equity", "debt", "real_estate", "commodities", "cash"]
        aggregated = {}
        for asset in asset_types:
            aggregated[asset] = sum(
                fund.get("allocation", 0) * (fund.get(asset, 0) / 100)
                for fund in portfolio["funds"]
            )
        return pd.Series(aggregated)
    else:
        # For non-fund assets, return an empty Series (or handle as needed)
        return pd.Series({})


def calculate_gain_daily_portfolio_series(portfolio, aligned_portfolio_civs, gold_series=None):
    """
    Calculate daily portfolio returns by combining returns from various assets.

    Returns:
        pd.DataFrame: DataFrame representing daily portfolio returns, with each asset as a column.
    """
    daily_returns = pd.DataFrame(
        index=aligned_portfolio_civs.index
        if not aligned_portfolio_civs.empty
        else None
    )

    # Gold
    if "gold" in portfolio:
        gold_returns = gold_series.pct_change().dropna()["Adjusted Spot Price"]
        # REQUIRED? gold_returns = gold_returns.rename(columns={"Adjusted Spot Price": "gold_value"}
        daily_returns = daily_returns.add(
            gold_returns * portfolio["gold"]["allocation"], fill_value=0
        )

    # PPF
    if "ppf" in portfolio:
        from data_loader import load_ppf_interest_rates
        from ppf_calculator import calculate_ppf_cumulative_gain
        ppf_rates = load_ppf_interest_rates(portfolio["ppf"]["ppf_interest_rates_file"])
        ppf_series = calculate_ppf_cumulative_gain(ppf_rates)
        # Ensure ppf_value is a DataFrame column
        #ppf_returns = ppf_series.pct_change().dropna()
        ppf_returns = ppf_series["ppf_value"].pct_change().dropna().to_frame(name="ppf_value")
        daily_returns = daily_returns.join(ppf_returns, how="outer")

    # Mutual Funds
    if "funds" in portfolio and not aligned_portfolio_civs.empty:
        funds_returns = sum(
            aligned_portfolio_civs.pct_change().dropna()[fund["name"]] * fund["allocation"]
            for fund in portfolio["funds"]
        ).to_frame(name="funds_value")
        #daily_returns = funds_returns if daily_returns is None else daily_returns.add(funds_returns, fill_value=0)
        daily_returns = daily_returns.join(funds_returns, how="outer")

    if daily_returns.empty:
        raise ValueError("Error: No valid asset returns were found! Portfolio calculations failed.")

    return daily_returns


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
