import sys
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os

# Helper functions assumed to be in your module:
def file_modified_within(file_path, refresh_hours):
    """Return True if the file was modified within the last refresh_hours."""
    if not os.path.exists(file_path):
        return False
    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
    return datetime.now() - mod_time < timedelta(hours=refresh_hours)

def rename_yahoo_data_columns(df):
    """Rename columns to match your desired format (assuming Date index is desired)."""
    # For example, you might want to ensure the DataFrame's index is named "Date"
    if 'Date' not in df.columns:
        df.index.name = "Date"
    return df

# Fetch Yahoo Finance data
def fetch_yahoo_finance_data(ticker, refresh_hours=6, period="max"):
    """
    Fetch historical data from Yahoo Finance for a given ticker symbol.

    Parameters:
        ticker (str): Yahoo Finance ticker symbol.
        period (str): The time period for historical data (default is "max").
        refresh_hours (int): Number of hours before refreshing the data (default is 6).

    Returns:
        pd.DataFrame: Historical data indexed by date.
    """
    file_path = f"{ticker.replace('^', '').replace('/', '_')}.csv"
    if not file_modified_within(file_path, refresh_hours):
        print(f"The data for {ticker} is outdated or missing. Fetching it using yfinance...")
        yf_ticker = yf.Ticker(ticker)
        raw = yf_ticker.history(period=period)
        raw.to_csv(file_path)
        print(f"Data downloaded successfully and saved to {file_path}.")
    else:
        raw = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    data = rename_yahoo_data_columns(raw)
    assert data.index.name == "Date", "Index name mismatch: expected 'Date'"
    return data

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python fetch_yahoo_finance_data.py <YAHOO_FINANCE_TICKER>")
        sys.exit(1)
    
    ticker = sys.argv[1]
    data = fetch_yahoo_finance_data(ticker)
    print(data.head())
