import numpy as np
import pandas as pd

def calculate_max_drawdowns(portfolio_gain_series, threshold=0.05):
    """
    Calculate maximum drawdowns with full retracements, printing debug info for key calculations only.

    Parameters:
        gain_cumulative_returns (pd.Series): Cumulative portfolio returns.
        threshold (float): Minimum drawdown percentage to report (e.g., 0.05 for 5%).

    Returns:
        list of dict: Each dict contains 'start_date', 'trough_date', 'end_date',
            and 'drawdown' (percentage).
    """
    print("\nDEBUG: Type of portfolio_gain_series:", type(portfolio_gain_series))
    if isinstance(portfolio_gain_series, pd.DataFrame):
        print("\nDEBUG: portfolio_gain_series is a DataFrame. Columns:", portfolio_gain_series.columns)
    elif isinstance(portfolio_gain_series, pd.Series):
        print("\nDEBUG: portfolio_gain_series is a Series.")
    
    max_drawdowns = []
    
    #gain_cumulative_returns = (1 + portfolio_gain_series).cumprod()

    # Convert gains back to absolute values for retracement checks
    #absolute_values = gain_cumulative_returns + 1  # Adjust gains to start from the base value
    
    #gain_peak = gain_cumulative_returns.cummax()  # Running record of the highest NAV
    max_drawdowns = []
    gain_peak = portfolio_gain_series.cummax()
    
    in_drawdown = False
    drawdown_start_date = None
    trough_date = None
    trough_value = None

    print("\nDEBUG: before `for` loop")
    #for date, value in gain_cumulative_returns.iteritems():
    for date, value in portfolio_gain_series.items():  # ✅ Works for Series
        peak_value = gain_peak.at[date]  # ✅ Ensures `gain_peak[date]` is a scalar
        if value < peak_value:  # Below peak, in drawdown
            if not in_drawdown:  # Start of a new drawdown
                in_drawdown = True
                drawdown_start_date = date
                trough_date = date
                trough_value = value
            elif value < trough_value:  # Update trough during drawdown
                trough_date = date
                trough_value = value
        elif in_drawdown and value >= gain_peak[drawdown_start_date]:  # Full retracement to peak
            # Calculate drawdown percentage
            drawdown_percentage = (trough_value - gain_peak[start_date]) / gain_peak[start_date]
            if (abs(drawdown_percentage) >= threshold):  # Only include significant drawdowns
                max_drawdowns.append(
                    {
                        "start_date": drawdown_start_date,
                        "trough_date": trough_date,
                        "end_date": date,
                        "drawdown": drawdown_percentage * 100,  # Convert to percentage
                    }
                )
            in_drawdown = False  # Reset drawdown state
        #else:  # Update peak if no drawdown is active
        #    gain_peak[date] = value

    return max_drawdowns


# Calculate annualized metrics
def calculate_annualized_metrics(portfolio_returns: pd.Series):
    """
    Calculate annualized return and volatility.
    
    Parameters:
        portfolio_returns (pd.Series): Daily portfolio returns

    Returns:
        tuple: Annualized return (float), volatility (float).
    """
    annualized_return = portfolio_returns.mean() * 252
    volatility = portfolio_returns.std() * (252**0.5)
    return annualized_return, volatility


# Calculate risk-adjusted metrics
def calculate_risk_adjusted_metrics(
    annualized_return, volatility, downside_risk, risk_free_rate
):
    """
    Calculate Sharpe and Sortino ratios.

    Parameters:
        annualized_return (float): Annualized portfolio return.
        volatility (float): Portfolio volatility.
        downside_risk (float): Portfolio downside risk.
        risk_free_rate (float): Risk-free rate.

    Returns:
        tuple: Sharpe ratio and Sortino ratio.
    """
    print("\nDEBUG: downside_risk before calculation:")
    print(downside_risk)
    print("\nDEBUG: downside_risk type:", type(downside_risk))
    sharpe_ratio = (
        (annualized_return - risk_free_rate) / volatility if volatility else np.nan
    )
    sortino_ratio = (
        (annualized_return - risk_free_rate) / downside_risk
        if downside_risk
        else np.nan
    )
    return sharpe_ratio, sortino_ratio


# Calculate Alpha and Beta
def calculate_alpha_beta(
    portfolio_returns, benchmark_returns, annualized_return, risk_free_rate
):
    """
    Calculate Alpha and Beta relative to a benchmark.

    Parameters:
        portfolio_returns (pd.Series): Portfolio daily returns.
        benchmark_returns (pd.Series): Benchmark daily returns.
        annualized_return (float): Annualized portfolio return.
        risk_free_rate (float): Risk-free rate.

    Returns:
        tuple: Alpha and Beta.
    """
    beta = portfolio_returns.corr(benchmark_returns) * (
        portfolio_returns.std() / benchmark_returns.std()
    )
    alpha = annualized_return - (
        risk_free_rate + beta * (benchmark_returns.mean() * 252 - risk_free_rate)
    )
    return alpha, beta


# Calculate downside risk
def calculate_downside_risk(gain_daily_portfolio_df, portfolio_weights):
    """
    Calculate downside risk of the portfolio.

    Parameters:
        portfolio_returns (pd.Series): Portfolio daily returns.

    Returns:
        float: Downside risk.
    """
    downside_returns = gain_daily_portfolio_df[gain_daily_portfolio_df < 0]

    # Always a Series, since gain_daily_portfolio_df is a DataFrame
    individual_downside_risks = downside_returns.std()

    # Compute weighted downside risk
    weighted_downside_risk = (individual_downside_risks * portfolio_weights).sum()
    
    #return downside_returns.std() * (252**0.5)
    return weighted_downside_risk


def calculate_benchmark_cumulative(benchmark_returns, earliest_datetime):
    benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
    benchmark_cumulative -= benchmark_cumulative.loc[earliest_datetime]
    return benchmark_cumulative


# Calculate all metrics
def calculate_portfolio_metrics(
        gain_daily_portfolio_series,
        portfolio,
        risk_free_rate,
        benchmark_returns=None,
        drawdown_threshold=0.05
):
    """
    Calculate key portfolio performance metrics.

    Parameters:
        gain_daily_portfolio_series (pd.DataFrame): Daily portfolio returns.
        portfolio (dict): The portfolio details from the TOML file.
        risk_free_rate (float): Risk-free rate.
        benchmark_returns (pd.Series, optional): Benchmark daily returns.
        drawdown_threshold (float): Minimum drawdown percentage to report.

    Returns:
        dict: Portfolio performance metrics.
        list: List of max drawdowns.
    """

    # Compute PPF cumulative returns if PPF exists
    if "ppf_value" in gain_daily_portfolio_series:
        ppf_cumulative_returns = (1 + gain_daily_portfolio_series["ppf_value"]).cumprod()
        start_value = ppf_cumulative_returns.iloc[0]
        end_value = ppf_cumulative_returns.iloc[-1]
        days = (ppf_cumulative_returns.index[-1] - ppf_cumulative_returns.index[0]).days
        ppf_annualized_return = (end_value / start_value) ** (365 / days) - 1
    else:
        ppf_annualized_return = 0

    # Compute mutual fund annualized return
    fund_columns = [col for col in gain_daily_portfolio_series.columns if col != "ppf_value"]
    if fund_columns:
        fund_daily_returns = gain_daily_portfolio_series[fund_columns]
        fund_annualized_return = fund_daily_returns.mean().sum() * 252
    else:
        fund_annualized_return = 0

    # Compute portfolio allocations
    ppf_allocation = portfolio.get("ppf", {}).get("allocation", 0)
    funds_allocation = sum(fund["allocation"] for fund in portfolio.get("funds", []))
    total_allocation = ppf_allocation + funds_allocation

    # Normalize allocations
    if total_allocation > 0:
        ppf_weight = ppf_allocation / total_allocation
        funds_weight = funds_allocation / total_allocation
    else:
        ppf_weight, funds_weight = 0, 0

    # Compute weighted portfolio annualized return
    annualized_return = (ppf_annualized_return * ppf_weight) + (fund_annualized_return * funds_weight)

    # Compute portfolio volatility
    volatility = gain_daily_portfolio_series.std().mean() * (252**0.5)

    # Compute Sharpe and Sortino ratios
    portfolio_weights = calculate_portfolio_allocations(portfolio)
    downside_risk = calculate_downside_risk(gain_daily_portfolio_series, portfolio_weights)
    sharpe_ratio, sortino_ratio = calculate_risk_adjusted_metrics(
        annualized_return, volatility, downside_risk, risk_free_rate
    )

    # Compute max drawdowns
    #max_drawdowns = calculate_max_drawdowns(gain_daily_portfolio_series, drawdown_threshold)
    # Compute a single weighted cumulative gain series for the portfolio
    gain_cumulative_portfolio_series = (gain_daily_portfolio_series * portfolio_weights).sum(axis=1)
    
    # Pass this correctly structured Series to `calculate_max_drawdowns()`
    max_drawdowns = calculate_max_drawdowns(gain_cumulative_portfolio_series, drawdown_threshold)


    metrics = {
        "Annualized Return": annualized_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Drawdowns": len(max_drawdowns),
    }

    if benchmark_returns is not None:
        alpha, beta = calculate_alpha_beta(
            gain_daily_portfolio_series, benchmark_returns, annualized_return, risk_free_rate
        )
        metrics.update({"Alpha": alpha, "Beta": beta})

    return metrics, max_drawdowns
    

def calculate_portfolio_allocations(portfolio):
    """
    Calculate the portfolio's aggregate asset allocations.

    Parameters:
        portfolio (dict): The full portfolio details (from the TOML file).
        fund_allocations (list): A list of dicts representing the allocations
            and asset breakdowns of each fund.
    
    Returns:
        dict: Aggregate portfolio allocations as percentages across asset classes.
    """
    # Define all asset types
    asset_types = ["ppf", "funds", "gold", "sgb", "scss", "rec_bond"]

    # Extract allocations dynamically
    portfolio_weights = pd.Series(
        {f"{asset}_value": portfolio.get(asset, {}).get("allocation", 0) for asset in asset_types}
    )

    return portfolio_weights


def calculate_ppf_relative_civ(ppf_interest_rates):
    """
    Calculate the monthly current investment value (CIV) gain for PPF, accruing monthly interest and crediting annually.

    Parameters:
        ppf_interest_rates (pd.DataFrame): DataFrame with 'rate' column and 'date' index.

    Returns:
        pd.DataFrame: DataFrame with 'civ' column indexed by date.
    """
    # Ensure the input is a Series
    if isinstance(ppf_interest_rates, pd.DataFrame):
        ppf_interest_rates = ppf_interest_rates["rate"]

    # Extend backward by one month
    first_date = ppf_interest_rates.index.min()
    extended_dates = pd.date_range(
        start=first_date - pd.DateOffset(months=1),
        end=ppf_interest_rates.index.max(),
        freq="M",
    )
    monthly_rates = ppf_interest_rates.reindex(extended_dates, method="pad")

    # Initialize CIV tracking
    yearly_civ = 1000.0  # Starting CIV for the financial year
    monthly_civ = yearly_civ  # Initialize monthly CIV
    accrued_interest = 0.0  # Accumulated interest during the year
    civs = []  # Store monthly CIV values

    # Iterate through dates and calculate CIV
    for i, (date, annual_rate) in enumerate(monthly_rates.items()):
        # Explicitly set the first CIV
        if i == 0:
            civs.append({"date": date, "civ": yearly_civ})
            continue

        # Calculate monthly interest
        monthly_interest = yearly_civ * ((annual_rate / 12) / 100)
        accrued_interest += monthly_interest
        # loader_logger.debug(f"date: {date}, annual_rate: {annual_rate}, monthly_interest: {monthly_interest}, accrued_interest: {accrued_interest}")

        # Update monthly CIV
        monthly_civ = yearly_civ + accrued_interest

        # At the end of the financial year (March), compound interest
        if date.month == 3:
            yearly_civ += accrued_interest  # Compound accrued interest
            accrued_interest = 0.0  # Reset accrued interest for the new year

        # Store monthly CIV
        civs.append(
            {
                "date": date,
                "civ": monthly_civ,  # Reflect monthly accruals
            }
        )

    # Create DataFrame for CIVs
    civ_df = pd.DataFrame(civs)
    civ_df["date"] = pd.to_datetime(civ_df["date"])
    civ_df.set_index("date", inplace=True)

    # Fill missing CIV values if any remain
    civ_df["civ"].fillna(method="ffill", inplace=True)

    return civ_df

import pandas as pd

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
        daily_returns = daily_returns.join(fund_returns, how="outer")

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

    # DEBUG: Print first few rows of computed cumulative gain
    print("\nDEBUG: First few rows of gain_cumulative_returns after .cumprod():")
    print(gain_cumulative_returns.head())

    return gain_cumulative_returns, gain_cumulative_returns  # Return twice just to keep compatibility
