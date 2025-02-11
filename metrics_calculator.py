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
    import pandas as pd
    
    # Start with mutual funds returns, if any.
    daily_returns = None
    if "funds" in portfolio and not aligned_portfolio_civs.empty:
        daily_returns = sum(
            aligned_portfolio_civs.pct_change().dropna()[fund["name"]] * fund["allocation"]
            for fund in portfolio["funds"]
        )

    # PPF
    if "ppf" in portfolio:
        from data_loader import load_ppf_interest_rates
        from ppf_calculator import calculate_ppf_cumulative_gain
        ppf_rates = load_ppf_interest_rates(portfolio["ppf"]["ppf_interest_rates_file"])
        portfolio_start_date = aligned_portfolio_civs.index.min() if not aligned_portfolio_civs.empty else ppf_rates.index.min()
        ppf_series = calculate_ppf_cumulative_gain(ppf_rates, portfolio_start_date)
        ppf_returns = ppf_series.pct_change().dropna()["ppf_value"]
        daily_returns = ppf_returns if daily_returns is None else daily_returns.add(ppf_returns * portfolio["ppf"]["allocation"], fill_value=0)

    # Gold
    if "gold" in portfolio:
        from fetch_gold_spot import get_gold_adjusted_spot
        portfolio_start_date = aligned_portfolio_civs.index.min() if not aligned_portfolio_civs.empty else None
        gold_series = get_gold_adjusted_spot(start_date=portfolio_start_date.strftime("%Y-%m-%d") if portfolio_start_date else "2000-01-01")
        gold_returns = gold_series.pct_change().dropna()["Adjusted Spot Price"]
        daily_returns = gold_returns if daily_returns is None else daily_returns.add(gold_returns * portfolio["gold"]["allocation"], fill_value=0)

    # SGB. Assume matured values based upon projections of the spot price of gold.
    if "sgb" in portfolio:
        from sgb_extractor import extract_sgb_series
        from bond_calculators import calculate_realistic_sgb_series
        # Extract SGB tranche data.
        sgb_data = extract_sgb_series()  # Optionally add refresh_hours logic.
        
        # Retrieve the gold spot series (this function already exists in fetch_gold_spot.py).
        from fetch_gold_spot import get_gold_adjusted_spot
        # Use an appropriate start date (e.g., overall_start from the SGB data or a fixed date)
        gold_series = get_gold_adjusted_spot(start_date="2015-01-01")  # adjust as needed
        
        # Compute the realistic SGB cumulative series.
        sgb_cum_series = calculate_realistic_sgb_series(sgb_data, gold_series)
        
        # Compute daily returns from the cumulative series.
        sgb_returns = sgb_cum_series.pct_change().dropna()
        daily_returns = sgb_returns if daily_returns is None else daily_returns.add(sgb_returns * portfolio["sgb"]["allocation"], fill_value=0)

    # SCSS
    if "scss" in portfolio:
        from scss_rates_extractor import fetch_html, extract_rate_series
        url = "https://www.nsiindia.gov.in/(S(2xgxs555qwdlfb2p4ub03n3n))/InternalPage.aspx?Id_Pk=181"
        html = fetch_html(url)  # You can add a refresh_hours check similar to Yahoo Finance.
        scss_data = extract_rate_series(html)
        scss_df = pd.DataFrame(scss_data)
        # Standardize header names—assume the table includes 'YEAR' and 'INTEREST'.
        scss_df.columns = [col.strip().lower() for col in scss_df.columns]
        print("DEBUG: SCSS DataFrame columns:", scss_df.columns)
        scss_df.rename(columns={'rate of interest (%)': 'interest'}, inplace=True)
        if 'year' in scss_df.columns and 'interest' in scss_df.columns:
            print("Raw 'year' values:", scss_df['year'].head())
            #scss_df['year'] = pd.to_datetime(scss_df['year'], format='%Y', errors='coerce')
            #scss_df.set_index('year', inplace=True)            
            #scss_df['year'] = scss_df['year'].str.extract(r'(\d{2}-\d{2}-\d{4})')[0].str.strip()
            #scss_df['year'] = pd.to_datetime(scss_df['year'], format='%d-%m-%Y', errors='coerce')
            #scss_df.set_index('year', inplace=True)
            scss_df['year'] = scss_df['year'].str.extract(r'(\d{1,2}[-.]\d{1,2}[-.]\d{4})')[0].str.strip()
            scss_df['year'] = pd.to_datetime(scss_df['year'], dayfirst=True, errors='coerce')
            scss_df.set_index('year', inplace=True)

            # Convert the 'interest' column to numeric (this is crucial for later calculations)
            scss_df['interest'] = pd.to_numeric(scss_df['interest'], errors='coerce')

            # Print for debugging purposes:
            print("SCSS DataFrame head:")
            print(scss_df.head())
            print("SCSS DataFrame info:")
            print(scss_df.info())
    
            from bond_calculators import calculate_variable_bond_cumulative_gain
            if not aligned_portfolio_civs.empty:
                portfolio_start_date = aligned_portfolio_civs.index.min()
            else:
                portfolio_start_date = scss_df.index.min()
            scss_cum_series = calculate_variable_bond_cumulative_gain(scss_df, portfolio_start_date)
            scss_returns = scss_cum_series.pct_change().dropna()
            daily_returns = scss_returns if daily_returns is None else daily_returns.add(scss_returns * portfolio["scss"]["allocation"], fill_value=0)

    # REC Bond (fixed coupon)
    if "rec_bond" in portfolio:
        rec_coupon = portfolio["rec_bond"].get("coupon", 5.0)  # Fixed 5.0% coupon by default.
        from bond_calculators import calculate_bond_cumulative_gain
        #portfolio_start_date = aligned_portfolio_civs.index.min() if not aligned_portfolio_civs.empty else pd.Timestamp.today()
        portfolio_start_date = aligned_portfolio_civs.index.min() if not aligned_portfolio_civs.empty else pd.Timestamp("2000-01-01")
        rec_cum_series = calculate_bond_cumulative_gain(rec_coupon, portfolio_start_date)
        rec_returns = rec_cum_series.pct_change().dropna()
        daily_returns = rec_returns if daily_returns is None else daily_returns.add(rec_returns * portfolio["rec_bond"]["allocation"], fill_value=0)

    return daily_returns if daily_returns is not None else pd.Series(dtype=float)


def calculate_gains_cumulative(gain_daily_portfolio_series, gain_daily_benchmark_series):
    # Compute cumulative returns for portfolio and benchmark
    cum_port = (1 + gain_daily_portfolio_series).cumprod() - 1
    cum_bench = (1 + gain_daily_benchmark_series).cumprod() - 1

    port_start = gain_daily_portfolio_series.index.min()
    bench_start = cum_bench.index.min()

    # If either is NaN (e.g., due to an empty series), provide a fallback date.
    if pd.isna(port_start) or pd.isna(bench_start):
        common_start_date = pd.Timestamp("2000-01-01")  # or another sensible fallback
    else:
        common_start_date = max(port_start, bench_start)

    # Determine common start date (later of the two series’ first dates)
    common_start_date = max(gain_daily_portfolio_series.index.min(), cum_bench.index.min())

    # Restrict both series to dates from the common start date onward
    cum_port = cum_port[cum_port.index >= common_start_date]
    cum_bench = cum_bench[cum_bench.index >= common_start_date]

    # Reindex the benchmark series to the portfolio index (using forward fill)
    cum_bench = cum_bench.reindex(cum_port.index, method='ffill')

    # Rebase both series so that they start at 0
    cum_port = cum_port - cum_port.iloc[0]
    cum_bench = cum_bench - cum_bench.iloc[0]

    return cum_port, cum_bench


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


def calculate_portfolio_allocations(portfolio, fund_allocations):
    """
    Calculate the portfolio's aggregate asset allocations.

    Parameters:
        portfolio (dict): The full portfolio details (from the TOML file).
        fund_allocations (list): A list of dicts representing the allocations and asset breakdowns of each fund.
    
    Returns:
        dict: Aggregate portfolio allocations as percentages across asset classes.
    """
    total_equity = sum(fund["allocation"] * (fund["equity"] / 100) for fund in fund_allocations)
    total_debt = sum(fund["allocation"] * (fund["debt"] / 100) for fund in fund_allocations)
    total_real_estate = sum(fund["allocation"] * (fund["real_estate"] / 100) for fund in fund_allocations)
    total_commodities = sum(fund["allocation"] * (fund["commodities"] / 100) for fund in fund_allocations)
    total_cash = sum(fund["allocation"] * (fund["cash"] / 100) for fund in fund_allocations)

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
