import numpy as np
import pandas as pd


def calculate_max_drawdowns(gain_daily_returns, threshold=0.05):
    """
    Calculate maximum drawdowns with full retracements, printing debug info for key calculations only.

    Parameters:
        gain_cumulative_returns (pd.Series): Cumulative portfolio returns.
        threshold (float): Minimum drawdown percentage to report (e.g., 0.05 for 5%).

    Returns:
        list of dict: Each dict contains 'start_date', 'trough_date', 'end_date', and 'drawdown' (percentage).
    """
    gain_cumulative_returns = (1 + gain_daily_returns).cumprod()

    # Convert gains back to absolute values for retracement checks
    absolute_values = (
        gain_cumulative_returns + 1
    )  # Adjust gains to start from the base value

    gain_peak = gain_cumulative_returns.cummax()  # Running record of the highest NAV
    max_drawdowns = []
    in_drawdown = False
    drawdown_start_date = None
    trough_date = None
    trough_value = None

    for date, value in gain_cumulative_returns.items():
        if value < gain_peak[date]:  # Below peak, in drawdown
            if not in_drawdown:  # Start of a new drawdown
                in_drawdown = True
                start_date = date
                trough_date = date
                trough_value = value
            elif value < trough_value:  # Update trough during drawdown
                trough_date = date
                trough_value = value
        elif in_drawdown and value >= gain_peak[start_date]:  # Full retracement to peak
            # Calculate drawdown percentage
            drawdown_percentage = (trough_value - gain_peak[start_date]) / gain_peak[
                start_date
            ]
            if (
                abs(drawdown_percentage) >= threshold
            ):  # Only include significant drawdowns
                max_drawdowns.append(
                    {
                        "start_date": start_date,
                        "trough_date": trough_date,
                        "end_date": date,
                        "drawdown": drawdown_percentage * 100,  # Convert to percentage
                    }
                )
            in_drawdown = False  # Reset drawdown state
        else:  # Update peak if no drawdown is active
            gain_peak[date] = value

    return max_drawdowns


# Calculate annualized metrics
def calculate_annualized_metrics(portfolio_returns):
    """
    Calculate annualized return and volatility.

    Parameters:
        portfolio_returns (pd.Series): Daily portfolio returns.

    Returns:
        tuple: Annualized return and volatility.
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
    """
    metrics_logger.debug(f"Timezone of portfolio_returns: {portfolio_returns.index.tz}")
    metrics_logger.debug(f"Timezone of benchmark_returns: {benchmark_returns.index.tz}")
    """

    beta = portfolio_returns.corr(benchmark_returns) * (
        portfolio_returns.std() / benchmark_returns.std()
    )
    alpha = annualized_return - (
        risk_free_rate + beta * (benchmark_returns.mean() * 252 - risk_free_rate)
    )
    return alpha, beta


# Calculate downside risk
def calculate_downside_risk(portfolio_returns):
    """
    Calculate downside risk of the portfolio.

    Parameters:
        portfolio_returns (pd.Series): Portfolio daily returns.

    Returns:
        float: Downside risk.
    """
    downside_returns = portfolio_returns[portfolio_returns < 0]
    return downside_returns.std() * (252**0.5)


def calculate_gain_daily_portfolio_series(portfolio, aligned_portfolio_civs):
    gain_daily_returns = aligned_portfolio_civs.pct_change().dropna()
    # Calculate portfolio daily returns based on allocations
    gain_daily_portfolio_df = sum(
        gain_daily_returns[fund["name"]] * fund["allocation"]
        for fund in portfolio["funds"]
    )
    return gain_daily_portfolio_df


def calculate_gains_cumulative(
    gain_daily_portfolio_series, gain_daily_benchmark_series
):
    cumulative_historical = (1 + gain_daily_portfolio_series).cumprod() - 1
    cumulative_benchmark = gain_daily_benchmark_series.cumprod() - 1
    return cumulative_historical, cumulative_benchmark


def calculate_benchmark_cumulative(benchmark_returns, earliest_datetime):
    benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
    benchmark_cumulative -= benchmark_cumulative.loc[earliest_datetime]
    return benchmark_cumulative


# Calculate all metrics
def calculate_portfolio_metrics(
    gain_daily_portfolio_series,
    risk_free_rate,
    benchmark_returns=None,
    drawdown_threshold=0.05,
):
    """
    Calculate key portfolio performance metrics.

    Parameters:
        portfolio_returns (pd.Series): Daily portfolio returns.
        risk_free_rate (float): Risk-free rate.
        benchmark_returns (pd.Series, optional): Benchmark daily returns.

    Returns:
        dict: Portfolio performance metrics.
    """
    annualized_return, volatility = calculate_annualized_metrics(
        gain_daily_portfolio_series
    )
    downside_risk = calculate_downside_risk(gain_daily_portfolio_series)

    sharpe_ratio, sortino_ratio = calculate_risk_adjusted_metrics(
        annualized_return, volatility, downside_risk, risk_free_rate
    )

    metrics = {
        "Annualized Return": annualized_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
    }

    if benchmark_returns is not None:
        """
        metrics_logger.debug(f"gain_daily_portfolio_series.head(): {gain_daily_portfolio_series.head()}")
        metrics_logger.debug(f"benchmark_returns.head(): {benchmark_returns.head()}")
        """
        alpha, beta = calculate_alpha_beta(
            gain_daily_portfolio_series,
            benchmark_returns,
            annualized_return,
            risk_free_rate,
        )
        metrics.update(
            {
                "Alpha": alpha,
                "Beta": beta,
            }
        )

    """
    metrics.update({
            "Drawdown Threshold": drawdown_threshold,
    })
    """
    metrics.update(
        {
            "Drawdown Threshold": f"{drawdown_threshold * 100:.0f}%",  # Convert to percent and format
        }
    )

    max_drawdowns = calculate_max_drawdowns(
        gain_daily_portfolio_series, drawdown_threshold
    )

    num_max_drawdowns = len(max_drawdowns)
    if max_drawdowns is not None:
        metrics.update({"Drawdowns": num_max_drawdowns})

    return metrics, max_drawdowns


# def calculate_portfolio_allocations(portfolio, fund_allocations):
def calculate_portfolio_allocations(fund_allocations):
    """
    Calculate the portfolio's aggregate equity, debt, and cash allocations.

    Parameters:
        portfolio (dict): The portfolio data loaded from the TOML file.

    Returns:
        dict: Aggregate portfolio allocations as percentages.
    """

    total_equity = sum(
        fund["allocation"] * (fund["equity"] / 100) for fund in fund_allocations
    )
    total_debt = sum(
        fund["allocation"] * (fund["debt"] / 100) for fund in fund_allocations
    )
    total_real_estate = sum(
        fund["allocation"] * (fund["real_estate"] / 100) for fund in fund_allocations
    )
    total_commodities = sum(
        fund["allocation"] * (fund["commodities"] / 100) for fund in fund_allocations
    )
    total_cash = sum(
        fund["allocation"] * (fund["cash"] / 100) for fund in fund_allocations
    )

    return {
        "equity": total_equity,
        "debt": total_debt,
        "real estate": total_real_estate,
        "commodities": total_commodities,
        "cash": total_cash,
    }


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
        # TODO: Is the interest rate accrued monthly 1/12 of the annual rate, or is it compounded? Answer: 1/12.
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
