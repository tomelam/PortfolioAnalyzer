import pandas as pd

# Load the sheet explicitly
df = pd.read_excel(
    "Gold_price_averages_in_a range_of_currencies_since_1978.xlsx",
    sheet_name="Monthly_Avg",
    header=None,
    skiprows=7,  # skip non-data header lines
    usecols=[2, 9],  # column 2: date, column 9: INR price
    names=["Date", "Price"]
)

# Drop rows with missing data
df.dropna(inplace=True)

# Optional: convert to correct types and sort
df["Date"] = pd.to_datetime(df["Date"])
df["Price"] = pd.to_numeric(df["Price"])
df.sort_values("Date", inplace=True)

# Save to CSV
df.to_csv("gold_monthly_inr.csv", index=False)
print("✅ Saved to gold_monthly_inr.csv")
