# ppf_calculator.py

import pandas as pd

def calculate_ppf_cumulative_gain(ppf_interest_rates, portfolio_start_date):
    """
    Calculate the relative cumulative gain series for a PPF investment.
    
    The series is computed assuming that on the portfolio start date the PPF value is 1.0
    (100%). Interest is accrued monthly (using annual_rate/12) and compounded annually (credited in March).
    
    This computed series serves as the equivalent of a historical NAV for the PPF,
    and it is used as an input for portfolio metrics and plots.
    
    Parameters:
        ppf_interest_rates (pd.DataFrame): DataFrame with a 'rate' column and datetime index (rates in percentages).
        portfolio_start_date (pd.Timestamp): The portfolio’s start date.
            
    Returns:
        pd.DataFrame: Daily relative cumulative gain series for the PPF (column 'ppf_value').
    """
    # Set up the calculation period: extend one month before the portfolio start date
    # to allow for accurate monthly accrual computation.
    extended_start = pd.to_datetime(portfolio_start_date) - pd.DateOffset(months=1)
    last_date = ppf_interest_rates.index.max()
    
    # Create a monthly date range for accrual calculation.
    monthly_dates = pd.date_range(start=extended_start, end=last_date, freq='ME')
    
    # Reindex the rates to this monthly frequency (using forward fill for missing values).
    monthly_rates = ppf_interest_rates['rate'].reindex(monthly_dates, method='pad')
    
    # Initialize baseline and accrual variables.
    baseline = 1.0  # Relative value at portfolio start.
    yearly_value = baseline
    accrued_interest = 0.0
    records = []
    
    for date, annual_rate in monthly_rates.items():
        # Skip months before the portfolio start date.
        if date < pd.to_datetime(portfolio_start_date):
            continue
        
        # Compute the monthly interest (using simple monthly accrual: annual_rate/12).
        monthly_interest = yearly_value * ((annual_rate / 12) / 100)
        accrued_interest += monthly_interest
        
        # The current value is the sum of the base value and accrued interest.
        current_value = yearly_value + accrued_interest
        
        # At the end of March (crediting point), compound the interest.
        if date.month == 3:
            yearly_value = current_value
            accrued_interest = 0.0
        
        records.append({'date': date, 'ppf_value': current_value})
    
    # Convert to DataFrame and reindex to daily frequency for alignment.
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.asfreq('D', method='ffill')
    
    return df

# --- Quick Test Section ---
if __name__ == '__main__':
    # For a quick test, simulate loading a CSV file of PPF interest rates.
    # The CSV should have columns "date" and "rate".
    csv_file = "ppf_interest_rates.csv"
    try:
        ppf_rates = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        exit(1)
    
    # Convert 'date' to datetime and set as index.
    ppf_rates['date'] = pd.to_datetime(ppf_rates['date'], format='%Y-%m-%d', errors='coerce')
    ppf_rates.dropna(subset=['date'], inplace=True)
    ppf_rates.set_index('date', inplace=True)
    
    # Define a portfolio start date (adjust as needed for testing).
    portfolio_start = pd.Timestamp("2023-01-01")
    
    # Calculate the PPF cumulative gain series.
    ppf_series = calculate_ppf_cumulative_gain(ppf_rates, portfolio_start)
    
    # Output a sample of the resulting series.
    print(ppf_series.head(10))
