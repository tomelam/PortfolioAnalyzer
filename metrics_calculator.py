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
    print("DEBUG: Type of portfolio_gain_series:", type(portfolio_gain_series))
    if isinstance(portfolio_gain_series, pd.DataFrame):
        print("DEBUG: portfolio_gain_series is a DataFrame. Columns:", portfolio_gain_series.columns)
    elif isinstance(portfolio_gain_series, pd.Series):
        print("DEBUG: portfolio_gain_series is a Series.")
    
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

    print("DEBUG: before `for` loop for finding maximum drawdowns-with-full-retracement")
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
    return

'''
    print("DEBUG: downside_risk before calculation: {downside_risk}")
    print("DEBUG: downside_risk type:", type(downside_risk))
    sharpe_ratio = (
        (annualized_return - risk_free_rate) / volatility if volatility else np.nan
    )
    sortino_ratio = (
        (annualized_return - risk_free_rate) / downside_risk
        if downside_risk
        else np.nan
    )
    return sharpe_ratio, sortino_ratio
'''


'''
# Calculate Alpha and Beta
def calculate_alpha_beta(
    gain_daily_portfolio_series, benchmark_returns, annualized_return, risk_free_rate
):
'''
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
    import pandas as pd

    debug = False

    # Ensure portfolio_returns is a Series.
    if isinstance(portfolio_returns, pd.DataFrame):
        port_ret = portfolio_returns.iloc[:, 0]
    else:
        port_ret = portfolio_returns

    # TODO: This is for debugging. I might have reversed the minuend and subtrahend in a calculation.
    port_ret = -port_ret

    # Squeeze benchmark_returns in case it's a DataFrame with one column.
    bench_ret = benchmark_returns.squeeze()

    if debug:
        print("DEBUG: After extraction, type(port_ret) =", type(port_ret))
        print("DEBUG: port_ret head:\n", port_ret.head())
        print("DEBUG: After squeeze, type(bench_ret) =", type(bench_ret))
        print("DEBUG: Portfolio returns index range:", port_ret.index.min(), "to", port_ret.index.max())
        print("DEBUG: Benchmark returns index range:", bench_ret.index.min(), "to", bench_ret.index.max())

    # Drop NaNs.
    port_ret = port_ret.dropna()
    bench_ret = bench_ret.dropna()

    # *** Sort both series in ascending order ***
    port_ret = port_ret.sort_index()
    bench_ret = bench_ret.sort_index()
    if debug:
        aligned = pd.DataFrame({
            "portfolio": port_ret,
            "benchmark": bench_ret
        })
        print(aligned.head(14))
        print(aligned.tail(14))

    # Determine the overlapping period:
    common_start = max(port_ret.index.min(), bench_ret.index.min())
    common_end = min(port_ret.index.max(), bench_ret.index.max())
    if debug:
        print("DEBUG: Overlap period from", common_start, "to", common_end)

    # Truncate both series to the overlapping period.
    port_ret = port_ret.loc[common_start:common_end]
    bench_ret = bench_ret.loc[common_start:common_end]
    if debug:
        print("DEBUG: After truncation, port_ret.shape =", port_ret.shape)
        print("DEBUG: After truncation, bench_ret.shape =", bench_ret.shape)

    # *** Align indices: reindex benchmark returns to match portfolio returns ***
    #bench_ret = bench_ret.reindex(port_ret.index)
    #bench_ret = bench_ret.dropna()
    bench_ret = bench_ret.reindex(port_ret.index, method='ffill')
    bench_ret = bench_ret.shift(1)
    if debug:
        print("DEBUG: After reindexing, port_ret.shape =", port_ret.shape)
        print("DEBUG: After reindexing, bench_ret.shape =", bench_ret.shape)

    # Calculate correlation and volatility ratio.
    corr = port_ret.corr(bench_ret)
    lagged_corr = port_ret.corr(bench_ret.shift(1))
    vol_ratio = port_ret.std() / bench_ret.std()
    if debug:
        print("DEBUG: Calculated correlation =", corr)
        print("Correlation with benchmark shifted by 1 day:", lagged_corr)
        print("DEBUG: Calculated volatility ratio =", vol_ratio)
    #corr = -lagged_corr

    if debug:
        print("Portfolio returns from", port_ret.index.min(), "to", port_ret.index.max())
        print("Benchmark returns from", bench_ret.index.min(), "to", bench_ret.index.max())
        print("Portfolio returns stats:", port_ret.describe())
        print("Benchmark returns stats:", bench_ret.describe())
        print("Correlation:", port_ret.corr(bench_ret))
        print("Portfolio std:", port_ret.std(), "Benchmark std:", bench_ret.std())

    beta = float(corr) * float(vol_ratio)
    alpha = float(port_ret.mean() - risk_free_rate - beta * (bench_ret.mean() - risk_free_rate))
    if debug:
        print("DEBUG: Calculated alpha =", alpha)
        print("DEBUG: Calculated beta =", beta)

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
        gain_daily_series,
        portfolio,
        risk_free_rate,
        benchmark_returns=None,
        drawdown_threshold=0.05
):
    """
    Calculate various performance metrics.

    Parameters:
        portfolio_returns (pd.DataFrame): Daily portfolio returns.
        portfolio (dict): The portfolio details from the TOML file.
        risk_free_rate (float): Risk-free rate.
        benchmark_returns (pd.Series, optional): Benchmark daily returns.
        drawdown_threshold (float): Minimum drawdown percentage to report.

    Returns:
        dict: Portfolio performance metrics.
        list: List of max drawdowns.
    """
    from portfolio_calculator import calculate_portfolio_allocations

    # Fill NaN values in the daily returns so that the cumulative product doesn't become NaN.
    gain_daily_series = gain_daily_series.fillna(0)

    # Compute cumulative returns from the daily gains.
    cumulative = (1 + gain_daily_series).cumprod()

    # Standard annualized return calculation:
    # 1. Total return over the period.
    total_return = cumulative.iloc[-1] / cumulative.iloc[0] - 1
    # 2. Number of years in the period.
    num_years = (gain_daily_series.index[-1] - gain_daily_series.index[0]).days / 365.25
    # 3. Annualized return.
    annualized_return = (1 + total_return) ** (1 / num_years) - 1

    # Compute annualized volatility (standard deviation scaled to 252 trading days).
    volatility = gain_daily_series.std() * (252 ** 0.5)

    # Ensure that annualized_return, risk_free_rate, and volatility are plain floats.
    annualized_return = float(annualized_return)
    risk_free_rate = float(risk_free_rate)
    volatility = float(volatility)

    # Compute Sharpe Ratio.
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility != 0 else float('nan')

    # Placeholder for Sortino Ratio (replace with actual calculation when ready).
    sortino_ratio = None

    # Placeholder for drawdown calculations using drawdown_threshold.
    max_drawdowns = None

    # Placeholders for Alpha and Beta calculations.
    alpha = None
    beta = None
    if benchmark_returns is not None:
        '''
        alpha, beta = calculate_alpha_beta(
            gain_daily_series, benchmark_returns, annualized_return, risk_free_rate
        )
        '''
        alpha, beta = calculate_alpha_beta(
            gain_daily_series,  # Is this the right data to use for the argument?
            benchmark_returns,
            annualized_return,
            risk_free_rate
        )


    metrics = {
        "Annualized Return": annualized_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Drawdowns": 0,  # TODO: Fix this.
        "Alpha": alpha,
        "Beta": beta,
    }

    print(f"type(benchmark_returns): {type(benchmark_returns)}")


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
