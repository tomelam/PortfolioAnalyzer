import pandas as pd
import os

def load_gold_prices(csv_path="data/Gold Futures Historical Data.csv"):
    """
    Loads a gold-futures CSV file and renames its detected price column to "Adjusted Spot Price".

    Raises:
        RuntimeError: If the CSV can't be read or if no price column is found.
    """
    # 1. Basic file existence check
    if not os.path.isfile(csv_path):
        raise RuntimeError(f"Gold price file not found: {csv_path}")

    # 2. Attempt reading the CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load gold data from {csv_path}: {e}")

    # 3. Identify a column named something like "Price" or "Close"
    possible_price_cols = [col for col in df.columns if "price" in col.lower() or "close" in col.lower()]
    if not possible_price_cols:
        raise RuntimeError(
            f"No price column found in gold CSV file {csv_path}. "
            f"Columns present: {list(df.columns)}"
        )

    price_col = possible_price_cols[0]
    #print(f"Detected price column: {price_col}")

    # 4. Rename the detected price column to "Adjusted Spot Price"
    df.rename(columns={price_col: "Adjusted Spot Price"}, inplace=True)

    # 5. Convert "Date" to datetime if it exists; set it as the index, then sort
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
    else:
        raise RuntimeError(
            f"No 'Date' column found in {csv_path}; data can't be time-indexed."
        )

    # 6. Clean the Adjusted Spot Price column (remove commas, %), convert to float
    df["Adjusted Spot Price"] = (
        df["Adjusted Spot Price"]
        .astype(str)
        .str.replace(",", "")
        .str.replace("%", "")
        .astype(float)
    )

    # 7. Optional debug print
    #print(f"Loaded gold data from {csv_path}, DataFrame shape: {df.shape}")
    #print(df.head(3))

    return df

if __name__ == "__main__":
    df_gold = load_gold_data_from_csv()
    print("Historical Gold Data (first 5 rows):")
    print(df_gold.head())
