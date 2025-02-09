import yfinance as yf
import pandas as pd

def get_gold_adjusted_spot(start_date="2000-01-01", end_date=None):
    """
    Fetches front-month COMEX gold futures prices and estimates spot prices.

    Parameters:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Adjusted gold spot price series.
    """
    # print(f"get_gold_adjusted_spot() received start_date: {start_date}")
    if start_date is None:
        start_date = "2000-01=01"
    if end_date is None:
        from datetime import datetime
        end_date = datetime.today().strftime("%Y-%m-%d")  # Use today's date if not provided

    print(f"Fetching gold futures data from {start_date} to {end_date}.")

    # Use only front-month futures (GC=F)
    front_month = "GC=F"

    # Fetch historical front-month futures prices
    try:
        front_data = yf.Ticker(front_month).history(start=start_date, end=end_date)
    except Exception as e:
        print(f"Error fetching front-month gold futures data: {e}")
        return None

    # Ensure valid data was retrieved
    if front_data.empty:
        print(f"Error: No data found for {front_month}")
        return None

    # Convert index to naive datetime
    front_data.index = pd.to_datetime(front_data.index).tz_localize(None)

    '''
    print("Gold futures data retrieved successfully. Sample:")
    print(front_data.head())

    # Print the last few prices
    print("Last available gold futures data:")
    print(front_data.tail())

    # Check if the latest gold futures price is suspiciously low
    latest_price = front_data["Close"].iloc[-1]
    print(f"Latest available gold futures price: {latest_price}")
    '''
    
    # Store front-month prices
    gold_futures = pd.DataFrame(index=front_data.index)
    gold_futures["Front-Month"] = front_data["Close"]

    # Estimate next-month futures using rolling 20-day average premium (0.5% per month assumption)
    gold_futures["Next-Month"] = gold_futures["Front-Month"] * 1.005

    # Calculate estimated futures premium (contango/backwardation effect)
    gold_futures["Premium"] = gold_futures["Next-Month"] - gold_futures["Front-Month"]

    # Estimate spot price: Front-month price minus premium adjustment
    gold_futures["Adjusted Spot Price"] = gold_futures["Front-Month"] - (gold_futures["Premium"] / 2)

    return gold_futures

# Example usage
if __name__ == "__main__":
    gold_spot = get_gold_adjusted_spot(start_date="2013-01-01", end_date="2023-12-31")
    if gold_spot is not None:
        print(gold_spot.head())  # Print first few rows
