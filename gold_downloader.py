import pandas as pd

def load_gold_data_from_csv(csv_file_path="Gold Futures Historical Data.csv"):
    """
    Load historical gold price data from a manually downloaded CSV file.
    
    This function expects the CSV to include at least columns "Date" and "Price".
    It will print the CSV header and a preview of the raw data, then convert
    the Date column to datetime and clean the Price column (removing commas).
    
    Returns:
        A pandas DataFrame with a datetime index and a 'price' column.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        raise Exception(f"Error reading CSV file {csv_file_path}: {e}")
    
    # Identify the date column (e.g. "Date")
    possible_date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
    if not possible_date_cols:
        raise ValueError("No date column found in CSV file.")
    date_col = possible_date_cols[0]
    
    # Identify the price column (e.g. "Price")
    possible_price_cols = [col for col in df.columns if "price" in col.lower() or "close" in col.lower()]
    if not possible_price_cols:
        raise ValueError("No price column found in CSV file.")
    price_col = possible_price_cols[0]
    
    # Convert the date column to datetime and set as index.
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df.set_index(date_col, inplace=True)
    
    # Clean the price column: remove commas and whitespace, then convert to numeric.
    df[price_col] = df[price_col].astype(str).str.replace(',', '', regex=False).str.strip()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])
    
    # Keep only the price column and rename it to 'price'
    df = df[[price_col]].rename(columns={price_col: "price"})
    df.sort_index(inplace=True)
    
    return df

if __name__ == "__main__":
    df_gold = load_gold_data_from_csv()
    print("Historical Gold Data (first 5 rows):")
    print(df_gold.head())
