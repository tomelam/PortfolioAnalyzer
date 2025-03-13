import pandas as pd
import numpy as np

#def calculate_ppf_cumulative_gain(ppf_interest_rates, portfolio_start_date):
def calculate_ppf_cumulative_gain(ppf_interest_rates):
    """
    Calculate the daily cumulative gain series for a PPF investment.
    
    - The annual interest rate is forward-filled to apply the correct rate each day.
    - The annual rate is converted to a daily effective rate using:
          daily_effective_rate = (1 + annual_rate / 100) ** (1 / 365) - 1
    - The cumulative gain is compounded daily.

    Parameters:
        ppf_interest_rates (pd.DataFrame): DataFrame with a 'rate' column and datetime index.
                                           Rates are in percentages.
        portfolio_start_date (str or pd.Timestamp): The portfolio’s start date.
    
    Returns:
        pd.DataFrame: A DataFrame with a daily cumulative gain series in the column 'ppf_value'.
    """
    print("\nDEBUG: Inside calculate_ppf_cumulative_gain()")
    #print("DEBUG: Received portfolio_start_date:", portfolio_start_date)
    # 1) Forward-fill daily rates from ppf_rates_df (which presumably has an index of date and a 'rate' column).
    #    If your code is daily compounding, watch out for an exact ~8% final outcome.
    #    For demonstration, I'm assuming you reindex to daily frequency and forward-fill:
    if not isinstance(ppf_interest_rates.index, pd.DatetimeIndex):
        raise ValueError("ppf_rates_df must have a DatetimeIndex")

    portfolio_start_date = "2001-03-01"
    portfolio_start_date = pd.to_datetime(portfolio_start_date)
    last_date = ppf_interest_rates.index.max()
    daily_dates = pd.date_range(start=portfolio_start_date, end=last_date, freq='D')

    # Forward-fill rates so that each day has an applicable interest rate
    daily_rates = ppf_interest_rates['rate'].reindex(daily_dates, method='ffill')

    print("PPF rates near early 2005:\n", daily_rates.loc["2005-01-01":"2005-01-10"].head(10))
    print("PPF rates near mid-2010:\n", daily_rates.loc["2010-06-01":"2010-06-10"].head(10))
    print("PPF rates near late 2020:\n", daily_rates.loc["2020-12-01":"2020-12-10"].head(10))

    print("\nDEBUG: First few rows of forward-filled daily rates:")
    print(daily_rates.head())
    print("\nDEBUG: Last few rows of forward-filled daily rates:")
    print(daily_rates.tail())

    # Convert annual rate to daily effective rate
    daily_effective_rate = (1 + daily_rates / 100) ** (1 / 365) - 1

    # Compute cumulative gain by compounding daily
    cumulative_gain = (1 + daily_effective_rate).cumprod()

    print("\nDEBUG: First few rows of computed cumulative gain:")
    print(cumulative_gain.head())
    print("\nDEBUG: Last few rows of computed cumulative gain:")
    print(cumulative_gain.tail())

    return cumulative_gain.to_frame(name='ppf_value')

if __name__ == '__main__':
    # Load the PPF rates CSV
    csv_file = "ppf_interest_rates.csv"
    try:
        ppf_rates = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        exit(1)

    # Convert 'date' column to datetime and set as index
    ppf_rates['date'] = pd.to_datetime(ppf_rates['date'], format='%Y-%m-%d', errors='coerce')
    ppf_rates.dropna(subset=['date'], inplace=True)
    ppf_rates.set_index('date', inplace=True)

    print("\nDEBUG: First few rows of PPF rates from file:")
    print(ppf_rates.head())
    print("\nDEBUG: Last few rows of PPF rates from file:")
    print(ppf_rates.tail())

    # Define the portfolio start date
    portfolio_start = "2001-03-01"
    print("\nDEBUG: Portfolio start date:", portfolio_start)

    # Compute cumulative gain series
    ppf_series = calculate_ppf_cumulative_gain(ppf_rates, portfolio_start)

    # Compute overall gain and annualized return
    start_value = ppf_series.iloc[0]['ppf_value']
    end_value = ppf_series.iloc[-1]['ppf_value']
    days = (ppf_series.index[-1] - ppf_series.index[0]).days
    print(f"days in test: {days}")
    annualized_return = (end_value / start_value) ** (365 / days) - 1

    print("\nDEBUG: Overall Gain: {:.4f}".format(end_value))
    print("DEBUG: Annualized Return: {:.4%}".format(annualized_return))
