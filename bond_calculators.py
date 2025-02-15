# bond_calculators.py
import pandas as pd
import numpy as np

def calculate_bond_cumulative_gain(annual_rate, portfolio_start_date, end_date=None):
    """
    Calculate a daily cumulative gain series for a bond with a constant annual rate.
    Uses simple daily accrual: daily_rate = (annual_rate/100)/252.
    """
    if end_date is None:
        end_date = pd.Timestamp.today()
    dates = pd.date_range(start=portfolio_start_date, end=end_date, freq='B')
    daily_rate = (annual_rate / 100) / 252
    # Create a constant daily return series and compound it:
    returns = pd.Series(daily_rate, index=dates)
    cum_gain = (1 + returns).cumprod()
    return cum_gain

def calculate_variable_bond_cumulative_gain(rate_df, portfolio_start_date):
    """
    Calculate a daily cumulative gain series from variable bond rates.
    
    Expects:
      - rate_df: a DataFrame with a DatetimeIndex and a column 'interest' representing the annual
        interest rate in percent.
      - portfolio_start_date: the start date for the calculation.
    
    The function reindexes the sparse rate data to a business day frequency, forward-fills missing
    values, converts the annual rate to a daily return (assuming 252 business days per year),
    and computes a cumulative product.
    
    Returns:
      pd.Series: A daily cumulative gain series.
    """

    # Ensure that the DataFrame's index is valid.
    rate_df = rate_df[rate_df.index.notna()]
    # Remove duplicate index labels, keeping the first occurrence.
    rate_df = rate_df[~rate_df.index.duplicated(keep='first')]
    
    # Ensure portfolio_start_date is a Timestamp.
    portfolio_start_date = pd.to_datetime(portfolio_start_date)
    # Define an end date; here we use today.
    end_date = pd.Timestamp.today()

    # NEW: Use the later of the portfolio start date and the earliest rate date.
    effective_start_date = (
        portfolio_start_date
        if portfolio_start_date >= rate_df.index.min()
        else rate_df.index.min()
    )
    
    # Create a business day date range.

    # DEBUG: Check the start and end dates used for the date range.
    print("DEBUG: portfolio_start_date =", portfolio_start_date)
    print("DEBUG: effective_start_date =", effective_start_date)
    print("DEBUG: end_date =", end_date)
    
    # Create a business day date range.
    dates = pd.date_range(start=portfolio_start_date, end=end_date, freq='B')
    
    # Reindex the rate data to the daily dates and forward-fill missing values.
    # Here, 'interest' should be numeric (in percent) – already converted outside.
    rate_series = rate_df['interest'].reindex(dates, method='ffill')

    print("DEBUG: rate_series shape:", rate_series.shape)
    
    # Convert annual rate (in percent) to daily return:
    # daily_rate = (1 + annual_rate/100)^(1/252) - 1
    daily_rate = (1 + rate_series / 100) ** (1/252) - 1
    
    # Compute the cumulative series by compounding daily returns.
    cum_series = (1 + daily_rate).cumprod()
    print("DEBUG: cum_series shape:", cum_series.shape)
    
    # Optionally, ensure the series is on a business day frequency.
    daily_cum_series = cum_series.asfreq('B', method='ffill')
    daily_cum_series = daily_cum_series.to_frame(name="scss_value")
    
    return daily_cum_series

# Calculate cumulative gain for a single-tranch SGB asset holding, if needed. Currently not used.
def calculate_sgb_cumulative_gain(sgb_data, portfolio_start_date, end_date=None):
    """
    Given a list of SGB tranche dictionaries (as extracted by sgb_extractor),
    compute a daily cumulative gain series for SGB.
    
    For simplicity, we select the tranche with the earliest Issue Date
    that is at or before the portfolio start date (or the earliest available if none qualify).
    
    We assume that the 'CAGRReturns (Absolute) (%)' column is formatted as
    "10.87 (128.46)" where the first number is the annualized return.
    
    Returns:
        pd.Series: A daily cumulative gain series from portfolio_start_date to end_date.
    """
    # Convert the list of dicts into a DataFrame.
    df = pd.DataFrame(sgb_data)
    # Parse the 'Issue Date' column.
    df['Issue Date'] = pd.to_datetime(df['Issue Date'], errors='coerce')
    df = df.dropna(subset=['Issue Date'])
    if df.empty:
        raise ValueError("No valid SGB tranches found.")
    
    portfolio_start_date = pd.to_datetime(portfolio_start_date)
    # Prefer a tranche that was issued on or before the portfolio start date.
    valid_df = df[df['Issue Date'] <= portfolio_start_date]
    if not valid_df.empty:
        chosen = valid_df.sort_values('Issue Date').iloc[0]
    else:
        chosen = df.sort_values('Issue Date').iloc[0]
    
    # Extract the annualized return from 'CAGRReturns (Absolute) (%)'
    cagr_str = chosen.get('CAGRReturns (Absolute) (%)', '')
    if cagr_str:
        try:
            # Expecting format like "10.87 (128.46)"; take the first number.
            annualized_return = float(cagr_str.split()[0])
        except Exception:
            annualized_return = 0.0
    else:
        annualized_return = 0.0
    
    # Compute an approximate daily rate using compound interest:
    daily_rate = (1 + annualized_return/100)**(1/252) - 1

    if end_date is None:
        end_date = pd.Timestamp.today()
    else:
        end_date = pd.to_datetime(end_date)
    
    dates = pd.date_range(start=portfolio_start_date, end=end_date, freq='B')
    # Generate the cumulative gain series via daily compounding.
    cum_gain = pd.Series((1 + daily_rate)**np.arange(len(dates)), index=dates)
    return cum_gain

# Calculate cumulative gain for merged tranches of SGB assets, using projections of
# the matured tranches. This is not very realistic, because those tranches matured
# many years ago. Not currently used.
def calculate_merged_sgb_series(sgb_data, end_date=None):
    """
    Given SGB tranche data (a list of dictionaries as extracted by sgb_extractor),
    build a daily time series that merges the tranches by averaging their cumulative
    values on days when more than one is active and forward-filling gaps.

    Each tranche is assumed to start at its Issue Date with a value of 1.0 and grow
    at its reported annualized return. The annualized return is extracted from the
    'CAGRReturns (Absolute) (%)' field (assumed to be in a format such as "10.87 (128.46)").

    Args:
        sgb_data (list): List of dictionaries, one per tranche.
        end_date (str or pd.Timestamp, optional): The end date of the series.
            Defaults to today.

    Returns:
        pd.Series: Daily merged cumulative series indexed by business day.
    """
    # Convert SGB data to DataFrame.
    df = pd.DataFrame(sgb_data)
    
    # Parse and clean the Issue Date.
    df['Issue Date'] = pd.to_datetime(df['Issue Date'], errors='coerce')
    df = df.dropna(subset=['Issue Date'])
    if df.empty:
        raise ValueError("No valid SGB tranches found.")
    
    # Extract the annualized return.
    def extract_annual_return(val):
        try:
            # Expect a format like "10.87 (128.46)" – take the first token.
            token = val.split()[0]
            # Remove any non-numeric characters (if needed)
            token = ''.join(c for c in token if c.isdigit() or c=='.' or c=='-')
            return float(token)
        except Exception:
            return np.nan

    df['AnnualReturn'] = df['CAGRReturns (Absolute) (%)'].apply(
        lambda x: extract_annual_return(x) if isinstance(x, str) else np.nan
    )
    df = df.dropna(subset=['AnnualReturn'])
    if df.empty:
        raise ValueError("No valid annual return data found in SGB tranches.")

    # Determine the overall start date as the earliest Issue Date.
    overall_start = df['Issue Date'].min()
    
    # Set end_date to today if not provided.
    if end_date is None:
        end_date = pd.Timestamp.today()
    else:
        end_date = pd.to_datetime(end_date)
    
    # Create a master daily (business days) date range.
    master_dates = pd.date_range(start=overall_start, end=end_date, freq='B')
    
    # Build a DataFrame to hold each tranche's cumulative series.
    tranche_series = pd.DataFrame(index=master_dates)
    
    for i, row in df.iterrows():
        issue_date = row['Issue Date']
        annual_return = row['AnnualReturn']
        # Compute daily rate from annual compounded over 252 business days.
        daily_rate = (1 + annual_return/100)**(1/252) - 1
        
        # Create a daily date range starting from the tranche's Issue Date.
        tranche_dates = pd.date_range(start=issue_date, end=end_date, freq='B')
        # Compute cumulative values: starting at 1.0 and compounding daily.
        n = np.arange(len(tranche_dates))
        values = (1 + daily_rate) ** n
        s = pd.Series(values, index=tranche_dates)
        
        # Reindex the series to the master date range and forward fill.
        s_reindexed = s.reindex(master_dates)
        s_reindexed = s_reindexed.ffill()
        # Before the tranche’s Issue Date the values remain NaN.
        tranche_series[f"tranche_{i}"] = s_reindexed

    # Average the active tranches on each day (ignoring NaNs).
    merged_series = tranche_series.mean(axis=1, skipna=True)
    # Forward fill any remaining gaps.
    merged_series = merged_series.ffill()
    return merged_series

def calculate_realistic_sgb_series(sgb_data, gold_series, default_coupon=2.50, end_date=None):
    """
    Build a realistic daily cumulative return series for SGBs by anchoring on the gold spot price.
    
    For each SGB tranche (from the sgb_data list), assume:
      - Capital appreciation equals the ratio of the gold spot price on a given day to the gold spot price
        on the tranche’s Issue Date.
      - A fixed coupon accrues at a daily rate (derived from default_coupon) and compounds over time.
      
    The function builds each tranche’s cumulative return series from its Issue Date until end_date,
    reindexes them to a master daily date range (using business days), and then averages overlapping periods.
    
    Args:
        sgb_data (list): List of dictionaries from sgb_extractor.
        gold_series (pd.Series): Gold spot price series (as returned by get_gold_adjusted_spot),
                                 indexed by date.
        default_coupon (float): Annual coupon yield (in percent) for SGBs. Default is 2.50.
        end_date (str or pd.Timestamp, optional): End date for the series. If None, uses the gold_series end.
    
    Returns:
        pd.Series: A merged daily cumulative return series for SGB.
    """
    # Convert the SGB data to a DataFrame and parse Issue Dates.
    df = pd.DataFrame(sgb_data)
    df['Issue Date'] = pd.to_datetime(df['Issue Date'], errors='coerce')
    df = df.dropna(subset=['Issue Date'])
    if df.empty:
        raise ValueError("No valid SGB tranche data found.")
    
    # Determine the overall start date as the earliest Issue Date.
    overall_start = df['Issue Date'].min()
    
    # Ensure gold_series is on a daily (business day) frequency.
    master_start = overall_start
    if end_date is None:
        end_date = gold_series.index.max()
    else:
        end_date = pd.to_datetime(end_date)
    master_dates = pd.date_range(start=master_start, end=end_date, freq='B')
    
    # Prepare a DataFrame to hold each tranche's cumulative series.
    tranche_series = pd.DataFrame(index=master_dates)
    
    # For each tranche, compute a daily series:
    for i, row in df.iterrows():
        issue_date = row['Issue Date']
        try:
            # Remove commas if any and convert to float.
            issue_price = float(str(row.get('Issue price/unit (₹)', '')).replace(',', ''))
        except Exception:
            continue  # Skip this tranche if the issue price is not available
        
        # Get the gold price at or after the issue date.
        gold_issue_candidates = gold_series[gold_series.index >= issue_date]
        if gold_issue_candidates.empty:
            continue  # Skip if we cannot determine a base gold price.
        #gold_at_issue = gold_issue_candidates.iloc[0]
        gold_at_issue = gold_issue_candidates.iloc[0]["Adjusted Spot Price"]
        
        # Create a daily date range for this tranche.
        tranche_dates = pd.date_range(start=issue_date, end=end_date, freq='B')
        
        # Capital appreciation factor: gold price today divided by gold price at issue.
        #gold_reindexed = gold_series.reindex(tranche_dates, method='ffill')
        gold_reindexed = gold_series["Adjusted Spot Price"].reindex(tranche_dates, method='ffill')
        capital_factor = gold_reindexed / gold_at_issue
        
        # Coupon accrual factor: use daily compounding.
        daily_coupon_rate = (1 + default_coupon/100)**(1/252) - 1
        n = np.arange(len(tranche_dates))
        coupon_factor = (1 + daily_coupon_rate) ** n
        
        # Total cumulative factor for this tranche:
        total_factor = capital_factor * coupon_factor
        
        # Reindex to the master date range and forward-fill.
        s = pd.Series(total_factor.values, index=tranche_dates)
        s = s.reindex(master_dates)
        s = s.ffill()
        tranche_series[f"tranche_{i}"] = s

    # Average the active tranches on each day (ignoring NaNs) and forward-fill.
    merged_series = tranche_series.mean(axis=1, skipna=True)
    merged_series = merged_series.ffill()
    return merged_series
