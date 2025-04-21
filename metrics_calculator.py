from logging import debug
import numpy as np
import pandas as pd
from utils import dbg


def geometric_annualized_return(daily_returns: pd.Series, periods_per_year=252):
    """
    CAGR from a Series of *daily* simple returns.
    Works for any length ≥ 1 day (handles holidays automatically).
    """
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        return float("nan")

    gross = (1 + daily_returns).prod()           # geometric link = end / start
    # Use calendar‑day span for exact annualisation
    days_span = (daily_returns.index[-1] - daily_returns.index[0]).days
    years = days_span / 365.25
    return gross**(1 / years) - 1


def calculate_max_drawdowns(portfolio_gain_series, threshold=0.05):
    """
    Calculate maximum drawdowns with full retracements.
    A drawdown is recorded only after the series has fully recovered to the peak from which it fell.

    Parameters:
        portfolio_gain_series (pd.Series): Cumulative portfolio returns.
        threshold (float): Minimum drawdown percentage to report (e.g., 0.05 for 5%).

    Returns:
        list of dict: Each dict contains 'start_date', 'trough_date', 'recovery_date',
            and 'drawdown' (percentage).
    """
    # Ensure we work with a Series.
    if isinstance(portfolio_gain_series, pd.DataFrame):
        portfolio_gain_series = portfolio_gain_series.iloc[:, 0]

    gain_peak = portfolio_gain_series.cummax()
    max_drawdowns = []
    in_drawdown = False
    drawdown_start_date = None
    drawdown_start_value = None
    trough_date = None
    trough_value = None

    for date, value in portfolio_gain_series.items():
        current_peak = gain_peak.at[date]
        if value < current_peak:
            # We're in a drawdown.
            if not in_drawdown:
                in_drawdown = True
                drawdown_start_date = date
                drawdown_start_value = current_peak  # Record the peak value at drawdown start
                trough_date = date
                trough_value = value
            else:
                # Update trough if value falls further.
                if value < trough_value:
                    trough_date = date
                    trough_value = value
        else:
            # Recovery: series has reached (or exceeded) the previous peak.
            if in_drawdown:
                if value >= drawdown_start_value:  # Full retracement to the initial peak
                    drawdown_percentage = (trough_value - drawdown_start_value) / drawdown_start_value
                    if abs(drawdown_percentage) >= threshold:
                        max_drawdowns.append({
                            "start_date": drawdown_start_date,
                            "trough_date": trough_date,
                            "recovery_date": date,
                            "drawdown": drawdown_percentage * 100,
                            "drawdown_days": (trough_date - drawdown_start_date).days + 1,
                            "recovery_days": (date - drawdown_start_date).days + 1,
                        })
                    in_drawdown = False
    # Do not record drawdowns that haven't been recovered.
    return max_drawdowns


def print_major_drawdowns(drawdowns):
    """
    Prints drawdowns.
    
    Args:
        drawdowns (list of dict): Each dict has "start_date", "recovery_date", "drawdown".
    """
    for dd in drawdowns:
        start = dd["start_date"].strftime("%Y-%m-%d")
        end = dd["recovery_date"].strftime("%Y-%m-%d")
        pct = dd["drawdown"]
        days = dd["recovery_days"]
        print(f"Drawdown from {start} to {end} ({days:>4} days): {pct:7.2f}%")


def calculate_downside_deviation(returns, target=0):
    """
    Calculate the daily downside deviation of returns below a target.
    
    Parameters:
        returns (pd.Series): Daily returns (as decimals).
        target (float): The target return (default 0).

    Returns:
        float: The daily downside deviation.
    """
    # Compute negative deviations (returns below target)
    negative_diff = np.minimum(0, returns - target)
    # Return the square root of the mean squared negative deviation
    return np.sqrt(np.mean(negative_diff**2))


def calculate_risk_adjusted_metrics(annualized_return, volatility, returns, risk_free_rate):
    """
    Calculate Sharpe and Sortino ratios on an annualized basis.

    Parameters:
        annualized_return (float): Annualized portfolio return.
        volatility (float): Annualized portfolio volatility.
        returns (pd.Series): Daily portfolio returns.
        risk_free_rate (float): Daily risk-free rate.

    Returns:
        tuple: (Sharpe ratio, Sortino ratio)
    """
    sharpe_ratio = (annualized_return - risk_free_rate * 252) / volatility if volatility != 0 else np.nan
    
    # Calculate the daily downside deviation of (returns - risk_free_rate)
    daily_downside_std = calculate_downside_deviation(returns - risk_free_rate)
    # Annualize the downside deviation assuming 252 trading days:
    annualized_downside_std = daily_downside_std * np.sqrt(252)
    
    sortino_ratio = ((annualized_return - risk_free_rate * 252) / annualized_downside_std
                     if annualized_downside_std and annualized_downside_std != 0 else np.nan)
    return sharpe_ratio, sortino_ratio


def calculate_alpha_beta(portfolio_returns, benchmark_returns, annualized_return, risk_free_rate):
    """
    Calculate Alpha and Beta relative to a benchmark using a one-day lag on the benchmark returns.
    This one-day shift is used because mutual fund NAVs are delayed relative to same-day benchmark data.
    
    Parameters:
        portfolio_returns (pd.Series): Portfolio daily returns.
        benchmark_returns (pd.Series): Benchmark daily returns.
        annualized_return (float): Annualized portfolio return (unused in this calculation).
        risk_free_rate (float): Risk-free rate.
    
    Returns:
        tuple: Alpha and Beta.
    """
    # Ensure portfolio_returns is a Series.
    if isinstance(portfolio_returns, pd.DataFrame):
        port_ret = portfolio_returns.iloc[:, 0]
    else:
        port_ret = portfolio_returns

    # Squeeze benchmark_returns in case it's a DataFrame with one column.
    bench_ret = benchmark_returns.squeeze()

    # Drop NaNs and sort indices.
    port_ret = port_ret.dropna().sort_index()
    bench_ret = bench_ret.dropna().sort_index()

    # Determine the overlapping period.
    common_start = max(port_ret.index.min(), bench_ret.index.min())
    common_end = min(port_ret.index.max(), bench_ret.index.max())
    port_ret = port_ret.loc[common_start:common_end]
    dbg(f"Last Overlapping Date of Portfolio and Benchmark: {common_end}")
    dbg(f"Last Benchmark Date: {bench_ret.index.max()}")
    bench_ret = bench_ret.loc[common_start:common_end]

    # Reintroduce the one-day shift on benchmark returns.
    # TODO: Rename `bench_ret_shifted` to `bench_ret`.
    #bench_ret_shifted = bench_ret.reindex(port_ret.index, method='ffill').shift(1)
    bench_ret_shifted = bench_ret.reindex(port_ret.index, method='ffill')

    # Drop any NaNs resulting from the shift.
    valid_idx = bench_ret_shifted.dropna().index
    port_ret = port_ret.loc[valid_idx]
    bench_ret_shifted = bench_ret_shifted.loc[valid_idx]

    # If after dropping NaNs we have insufficient data, return NaN.
    if len(bench_ret_shifted) < 2:
        return np.nan, np.nan

    # Calculate covariance and variance.
    covariance = np.cov(port_ret, bench_ret_shifted, ddof=1)[0, 1]
    variance = np.var(bench_ret_shifted, ddof=1)
    beta = covariance / variance if variance != 0 else np.nan
    alpha = port_ret.mean() - risk_free_rate - beta * (bench_ret_shifted.mean() - risk_free_rate)
    alpha *= 252  # Annualize alpha
    return alpha, beta


def calculate_benchmark_cumulative(benchmark_returns, earliest_datetime):
    benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
    benchmark_cumulative -= benchmark_cumulative.loc[earliest_datetime]
    return benchmark_cumulative


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
        gain_daily_series (pd.DataFrame or pd.Series): Daily portfolio returns.
        portfolio (dict): The portfolio details from the TOML file.
        risk_free_rate (float): Risk-free rate.
        benchmark_returns (pd.Series, optional): Benchmark daily returns.
        drawdown_threshold (float): Minimum drawdown percentage to report.

    Returns:
        dict: Portfolio performance metrics.
        list: List of max drawdowns.
    """

    # Fill NaN values in the daily returns.
    gain_daily_series = gain_daily_series.fillna(0)

    # Compute cumulative returns.
    cumulative = (1 + gain_daily_series).cumprod()

    # Annualised return.
    annualized_return = geometric_annualized_return(gain_daily_series)
    
    # Annualized volatility.
    volatility = gain_daily_series.std() * (252 ** 0.5)

    # Convert to floats.
    annualized_return = float(annualized_return.iloc[0])
    risk_free_rate = float(risk_free_rate)
    volatility = float(volatility.iloc[0])

    sharpe_ratio, sortino_ratio = calculate_risk_adjusted_metrics(
        annualized_return, volatility, gain_daily_series, risk_free_rate
    )

    # Calculate alpha and beta if benchmark returns are provided.
    alpha = None
    beta = None
    if benchmark_returns is not None:
        alpha, beta = calculate_alpha_beta(
            gain_daily_series, benchmark_returns, annualized_return, risk_free_rate
        )

    # Calculate drawdowns.
    max_drawdowns = calculate_max_drawdowns(cumulative, threshold=drawdown_threshold)

    metrics = {
        "Annualized Return": annualized_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Alpha": alpha,
        "Beta": beta,
        "Drawdowns": len(max_drawdowns),
    }

    return metrics, max_drawdowns
    

def calculate_ppf_relative_civ(ppf_interest_rates):
    """
    Calculate the monthly current investment value (CIV) gain for PPF, accruing monthly interest and
    crediting annually.

    Parameters:
        ppf_interest_rates (pd.DataFrame): DataFrame with 'rate' column and 'date' index.

    Returns:
        pd.DataFrame: DataFrame with 'civ' column indexed by date.
    """
    # Ensure the input is a Series.
    if isinstance(ppf_interest_rates, pd.DataFrame):
        ppf_interest_rates = ppf_interest_rates["rate"]

    first_date = ppf_interest_rates.index.min()
    extended_dates = pd.date_range(
        start=first_date - pd.DateOffset(months=1),
        end=ppf_interest_rates.index.max(),
        freq="M",
    )
    monthly_rates = ppf_interest_rates.reindex(extended_dates, method="pad")

    yearly_civ = 1000.0  # Starting CIV for the financial year
    monthly_civ = yearly_civ
    accrued_interest = 0.0
    civs = []

    for i, (date, annual_rate) in enumerate(monthly_rates.items()):
        if i == 0:
            civs.append({"date": date, "civ": yearly_civ})
            continue

        monthly_interest = yearly_civ * ((annual_rate / 12) / 100)
        accrued_interest += monthly_interest
        monthly_civ = yearly_civ + accrued_interest

        if date.month == 3:
            yearly_civ += accrued_interest
            accrued_interest = 0.0

        civs.append({"date": date, "civ": monthly_civ})

    civ_df = pd.DataFrame(civs)
    civ_df["date"] = pd.to_datetime(civ_df["date"])
    civ_df.set_index("date", inplace=True)
    civ_df["civ"].fillna(method="ffill", inplace=True)

    return civ_df
