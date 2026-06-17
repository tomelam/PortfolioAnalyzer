"""Load PPF interest rates and synthesize a daily-aligned CIV series.

The PPF (Public Provident Fund) is a long-running Government of India
savings scheme. Its declared annual interest rate changes at most
quarterly; balances accrue interest monthly with annual crediting.
This module isolates that machinery from the broader data_loader so
the rate CSV and synthetic CIV can be tested independently.
"""

from __future__ import annotations

import pandas as pd


def load_ppf_interest_rates(csv_file_path: str = "data/ppf_interest_rates.csv") -> pd.DataFrame:
    """Read a CSV of declared PPF rates indexed by effective date.

    The CSV must have ``date`` and ``rate`` columns (date is parsed as
    ``YYYY-MM-DD``; rate as float percent).
    """
    if csv_file_path is None:
        csv_file_path = "data/ppf_interest_rates.csv"

    try:
        ppf_data = pd.read_csv(csv_file_path)
    except Exception as e:
        raise FileNotFoundError(f"Failed to read PPF rates from {csv_file_path}: {e}") from e

    if "date" not in ppf_data.columns or "rate" not in ppf_data.columns:
        raise ValueError("CSV file must contain 'date' and 'rate' columns.")

    ppf_data["date"] = pd.to_datetime(ppf_data["date"], format="%Y-%m-%d", errors="coerce")
    ppf_data = ppf_data.dropna(subset=["date"]).set_index("date")
    ppf_data["rate"] = pd.to_numeric(ppf_data["rate"], errors="coerce")
    ppf_data = ppf_data.dropna(subset=["rate"]).sort_index()
    return ppf_data


def load_ppf_civ() -> pd.Series:
    """Return a daily PPF cumulative-investment-value Series.

    The underlying synthetic generator emits one row per month. We
    reindex to a daily frequency and forward-fill so the series can be
    aligned with daily mutual-fund NAVs without introducing NaNs (which
    would be rejected by ``asset_timeseries.from_civ``). The series is
    extended out to "today" carrying the most recent interest rate
    forward — PPF rates change quarterly at most, so this matches how
    PPF balances actually accrue between rate-change announcements.
    """
    from synthetic_civ import calculate_ppf_relative_civ

    monthly = calculate_ppf_relative_civ(load_ppf_interest_rates())["civ"]
    end = max(monthly.index.max(), pd.Timestamp.today().normalize())
    daily_index = pd.date_range(monthly.index.min(), end, freq="D")
    return monthly.reindex(daily_index).ffill().rename("PPF")
