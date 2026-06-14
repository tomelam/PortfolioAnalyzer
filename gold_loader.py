"""Load monthly gold spot prices from a CSV.

Returns a ``pd.Series`` named ``"price"`` indexed by date so the result
slots directly into ``main.py``'s ``nav_inputs`` dict alongside the
mutual-fund Series.
"""

from __future__ import annotations

import os

import pandas as pd


def load_gold_prices(csv_path: str = "data/gold_monthly_inr.csv") -> pd.Series:
    """Read a gold-price CSV and return a sorted Series of float prices.

    Raises:
        RuntimeError: if the file is missing, unreadable, or lacks a
            recognizable price column or a ``Date`` column.
    """
    if not os.path.isfile(csv_path):
        raise RuntimeError(f"Gold price file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load gold data from {csv_path}: {e}") from e

    price_cols = [c for c in df.columns if "price" in c.lower() or "close" in c.lower()]
    if not price_cols:
        raise RuntimeError(
            f"No price column found in gold CSV file {csv_path}. "
            f"Columns present: {list(df.columns)}"
        )

    if "Date" not in df.columns:
        raise RuntimeError(f"No 'Date' column found in {csv_path}; data can't be time-indexed.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.set_index("Date").sort_index()

    series = (
        df[price_cols[0]]
        .astype(str)
        .str.replace(",", "")
        .str.replace("%", "")
        .astype(float)
        .rename("price")
    )
    return series
