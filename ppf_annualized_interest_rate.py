import pandas as pd
import numpy as np

def calculate_ppf_overall_annualized(csv_file):
    # 1) Load CSV
    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    # 2) We'll assume columns: ['date','rate'], where 'rate' is e.g. 8.0 for 8%
    # Confirm the earliest & latest date
    start_date = df['date'].iloc[0]
    end_date = df['date'].iloc[-1]

    # If you want to define your own final date, you could do so, but let's assume last row is end
    # If you have a separate 'end_date' beyond the last announcement, you'd handle that.

    # 3) Build intervals
    # We'll create an expanded DF with an 'end_date' column for each row
    # The last row's 'end_date' is your final date
    df.reset_index(drop=True, inplace=True)
    intervals = []
    for i in range(len(df) - 1):
        current_date = df.loc[i, 'date']
        current_rate = df.loc[i, 'rate']
        next_date = df.loc[i+1, 'date']
        # Interval length in years:
        delta_years = (next_date - current_date).days / 365.25
        intervals.append((delta_years, current_rate))

    # Add a final interval from the last row's date to 'end_date' (if they differ)
    final_start = df.loc[len(df)-1, 'date']
    final_rate = df.loc[len(df)-1, 'rate']
    if final_start < end_date:
        delta_years = (end_date - final_start).days / 365.25
        intervals.append((delta_years, final_rate))

    # 4) Compute the product of each interval's growth factor
    total_factor = 1.0
    for (yrs, annual_rate) in intervals:
        # factor_i = (1 + (annual_rate/100))^yrs
        total_factor *= (1.0 + annual_rate/100.0) ** yrs

    # 5) The total time from entire series
    total_years = (end_date - start_date).days / 365.25
    # 6) Solve for overall annualized return
    annualized_return = total_factor ** (1.0 / total_years) - 1.0

    return annualized_return, total_factor, total_years

# Example usage:
if __name__ == '__main__':
    csv_file = 'ppf_interest_rates.csv'  # Replace with your actual CSV
    ann_return, total_factor, yrs = calculate_ppf_overall_annualized(csv_file)
    print(f"Overall annualized return from {csv_file}: {ann_return*100:.4f}%")
    print(f"Final factor: {total_factor:.4f} over ~{yrs:.2f} years")
