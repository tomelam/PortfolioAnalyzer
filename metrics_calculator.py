import numpy as np
import pandas as pd

def calculate_max_drawdowns(portfolio_gain_series, threshold=0.05):
    """
    Identify drawdown periods from a cumulative portfolio series.
    A drawdown starts at a local peak, continues until the portfolio
    returns to or exceeds that peak, and is recorded if it exceeds
    the given threshold (e.g. 0.05 for 5%).
    
    Parameters:
        portfolio_gain_series (pd.Series): A cumulative series, e.g. (1 + returns).cumprod().
        threshold (float): Minimum drawdown fraction to record.

    Returns:
        list of dict: Each dict has 'start_date', 'trough_date', 'end_date', 'drawdown' (in %).
    """
    # We assume portfolio_gain_series is sorted by date ascending.
    # If it's not, sort it or ensure it is sorted before calling.
    portfolio_gain_series = portfolio_gain_series.sort_index()
    
    max_drawdowns = []
    
    peak_value = None
    peak_date = None
    in_drawdown = False
    trough_date = None
    trough_value = None

    for date, value in portfolio_gain_series.items():
        if peak_value is None or value > peak_value:
            # The portfolio has reached a new peak
            # If we were in a drawdown, it ends here.
            if in_drawdown:
                # Calculate how big the drawdown was
                dd_fraction = (trough_value - peak_value) / peak_value
                if abs(dd_fraction) >= threshold:
                    max_drawdowns.append({
                        "start_date": peak_date,
                        "trough_date": trough_date,
                        "end_date": date,
                        "drawdown": dd_fraction * 100,  # convert to %
                    })
                in_drawdown = False
            
            # Update the new peak
            peak_value = value
            peak_date = date
        else:
            # The portfolio is below the peak
            if not in_drawdown:
                # We are entering a drawdown
                in_drawdown = True
                trough_value = value
                trough_date = date
            else:
                # Update trough if this is a new low
                if value < trough_value:
                    trough_value = value
                    trough_date = date
    
    # If we end in a drawdown that never fully retraces, you can choose
    # whether or not to record it. For example:
    if in_drawdown:
        dd_fraction = (trough_value - peak_value) / peak_value
        if abs(dd_fraction) >= threshold:
            # We consider the end_date the last known date
            last_date = portfolio_gain_series.index[-1]
            max_drawdowns.append({
                "start_date": peak_date,
                "trough_date": trough_date,
                "end_date": last_date,
                "drawdown": dd_fraction * 100,
            })
    
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
    # Prevent a near-0 downside risk from making the denominator 0 by adding a small epsilon
    sharpe_ratio = (annualized_return - risk_free_rate) / (volatility + 1e-6)
    # Prevent a near-0 downside risk from making the denominator 0 by adding a small epsilon
    sortino_ratio = (annualized_return - risk_free_rate) / (downside_risk + 1e-6)
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


def calculate_downside_risk(portfolio_returns, portfolio_weights=None):
    """
    Calculate the standard deviation of negative daily returns (downside risk).
    Expects a single Series (not a DataFrame).
    """
    import numpy as np

    # Filter negative returns
    negative_returns = portfolio_returns[portfolio_returns < 0]

    # If there are no negative returns, downside risk is 0
    if negative_returns.empty:
        return 0.0

    # Convert to annualized downside risk
    return negative_returns.std() * np.sqrt(252)


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
    from portfolio_calculator import calculate_portfolio_allocations

    # Combine variable-rate bond returns from both PPF and SCSS.
    var_bond_keys = [k for k in gain_daily_portfolio_series.columns if k in ["ppf_value", "scss_value",]]
    if var_bond_keys:
        # Compute the daily returns as the average of the available variable bond returns.
        var_bond_daily_returns = gain_daily_portfolio_series[var_bond_keys].mean(axis=1)
        var_bond_cumulative = (1 + var_bond_daily_returns).cumprod()
        start_value = var_bond_cumulative.iloc[0]
        end_value = var_bond_cumulative.iloc[-1]
        days = (var_bond_cumulative.index[-1] - var_bond_cumulative.index[0]).days
        var_bond_annualized_return = (end_value / start_value) ** (365 / days) - 1
    else:
        var_bond_annualized_return = 0

    # Compute mutual fund annualized return
    fund_columns = [col for col in gain_daily_portfolio_series.columns if col != "ppf_value"]
    if fund_columns:
        fund_daily_returns = gain_daily_portfolio_series[fund_columns]
        fund_annualized_return = fund_daily_returns.mean().sum() * 252
    else:
        fund_annualized_return = 0

    # Compute portfolio allocations
    var_bond_allocation = portfolio.get("ppf", {}).get("allocation", 0) + portfolio.get("scss", {}).get("allocation", 0)
    funds_allocation = sum(fund.get("allocation", 0) for fund in portfolio.get("funds", []))
    total_allocation = var_bond_allocation + funds_allocation

    # Normalize allocations
    if total_allocation > 0:
        var_bond_weight = var_bond_allocation / total_allocation
        funds_weight = funds_allocation / total_allocation
    else:
        var_bond_weight, funds_weight = 0, 0

    # Compute weighted portfolio annualized return using the combined variable-rate bond return.
    annualized_return = (var_bond_annualized_return * var_bond_weight) + (fund_annualized_return * funds_weight)

    # Compute portfolio volatility
    volatility = gain_daily_portfolio_series.std().mean() * (252**0.5)

    # Compute Sharpe and Sortino ratios
    portfolio_weights = calculate_portfolio_allocations(portfolio)

    # Compute a single weighted cumulative gain series for the portfolio
    if "funds_value" in gain_daily_portfolio_series.columns and len(gain_daily_portfolio_series.columns) == 1:
        daily_return_series = gain_daily_portfolio_series["funds_value"]
    else:
        daily_return_series = (gain_daily_portfolio_series * portfolio_weights).sum(axis=1)
    # Convert daily returns into a cumulative factor starting at ~1.0.
    gain_cumulative_portfolio_series = (1 + daily_return_series).cumprod()

    downside_risk = calculate_downside_risk(daily_return_series, portfolio_weights=None)
    sharpe_ratio, sortino_ratio = calculate_risk_adjusted_metrics(
        annualized_return, volatility, downside_risk, risk_free_rate
    )

    # Compute maximum drawdowns
    max_drawdowns = calculate_max_drawdowns(gain_cumulative_portfolio_series, drawdown_threshold)

    metrics = {
        "Annualized Return": annualized_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Drawdowns": len(max_drawdowns),
    }

    if benchmark_returns is not None:
        '''
        alpha, beta = calculate_alpha_beta(
            gain_daily_portfolio_series, benchmark_returns, annualized_return, risk_free_rate
        )
        '''
        alpha, beta = calculate_alpha_beta(
            gain_cumulative_portfolio_series,
            benchmark_returns,
            annualized_return,
            risk_free_rate
        )
        metrics.update({"Alpha": alpha, "Beta": beta})

    return metrics, max_drawdowns
    

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
