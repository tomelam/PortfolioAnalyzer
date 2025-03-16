import pandas as pd
import numpy as np
import argparse

def create_sgb_daily_returns(csv_path="sgb_data.csv"):
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

    # Resample daily and forward-fill missing days
    df_daily = df.resample('D').ffill()

    # Remove rows before first valid issue price
    first_valid_idx = df_daily['Issue Price (₹/gram)'].first_valid_index()
    df_daily = df_daily.loc[first_valid_idx:]

    # Calculate daily returns
    df_daily_returns = df_daily['Issue Price (₹/gram)'].pct_change()

    # Replace infinite values and initial NaNs with zero
    df_daily_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_daily_returns = df_daily_returns.fillna(0)

    return df_daily_returns.to_frame(name='Daily Returns')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Sovereign Gold Bond tranche data into a Pandas DataFrame.")
    parser.add_argument("csv_path", type=str, nargs="?", default="sgb_data.csv", help="Path to the SGB CSV file (default: sgb_data.csv)")
    args = parser.parse_args()

    sgb_df = create_sgb_dataframe(args.csv_path)
    print("\nSovereign Gold Bond Tranche Data:\n")
    print(sgb_df)
