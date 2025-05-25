import pandas as pd
import numpy as np
import argparse

def create_sgb_daily_returns(csv_path="sgb_data.csv", initial_value=100):
    """
    Creates a Pandas DataFrame with DateTime index based on SGB tranche issue dates.
    
    Args:
        csv_path (str): Path to the CSV file containing SGB tranche data.

    Returns:
        pd.DataFrame: DataFrame with DateTime index (issue dates) and issue prices as values.
    """
    df = pd.read_csv(csv_path, parse_dates=["Issue Date"])
    df.set_index("Issue Date", inplace=True)
    df.sort_index(inplace=True)

    df_daily = df.resample('D').ffill()
    first_valid_idx = df_daily['Issue Price (₹/gram)'].first_valid_index()
    df_daily = df_daily.loc[first_valid_idx:]

    # Compute daily price returns
    price_returns = df_daily['Issue Price (₹/gram)'].pct_change().fillna(0)

    # Include the fixed annual interest rate of 2.5% for SGBs
    daily_interest_rate = (1 + 0.025)**(1/365) - 1
    total_daily_returns = price_returns + daily_interest_rate

    # Compounded value series
    value_series = initial_value * (1 + total_daily_returns).cumprod()
    value_series.name = "value"

    return value_series


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Sovereign Gold Bond tranche data into a Pandas DataFrame.")
    parser.add_argument("csv_path", type=str, nargs="?", default="sgb_data.csv", help="Path to the SGB CSV file (default: sgb_data.csv)")
    args = parser.parse_args()

    sgb_df = create_sgb_dataframe(args.csv_path)
    print("\nSovereign Gold Bond Tranche Data:\n")
    print(sgb_df)
