"""Load daily gold spot prices from a CSV.

Returns a ``pd.Series`` named ``"price"`` indexed by date so the result
slots directly into ``main.py``'s ``nav_inputs`` dict alongside the
mutual-fund Series.

Unit caveat: ``data/reference/gold_lbma_usd_daily.csv`` stores prices in **USD per
troy ounce** (the LBMA Gold Price PM fix, auto-refreshed via
``loaders.data_update``). USD is used unconverted: the analyzer reports only
normalized returns, and a time-varying USD/INR path would inject rupee/RBI
dynamics into gold's measured return (a constant unit cancels; a varying FX
path does not). For relative-returns analysis (the main.py path) the units
cancel in normalization anyway. For absolute valuation (the SGB engine — where
you multiply by ``units_grams``), call ``load_gold_prices_per_gram`` to get the
**USD-per-gram** series.
"""

from __future__ import annotations

import os

import pandas as pd

GRAMS_PER_TROY_OUNCE = 31.1034768

DEFAULT_GOLD_CSV = "data/reference/gold_lbma_usd_daily.csv"


def load_gold_prices(csv_path: str = DEFAULT_GOLD_CSV) -> pd.Series:
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

    parsed = pd.to_datetime(df["Date"], errors="coerce")
    if parsed.isna().any():  # fail fast: a dropped row would silently bend the series
        bad = df["Date"][parsed.isna()].tolist()
        raise RuntimeError(f"Unparseable date(s) in {csv_path}: {bad}")
    df["Date"] = parsed
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


def load_gold_prices_per_gram(csv_path: str = DEFAULT_GOLD_CSV) -> pd.Series:
    """Return the same series as :func:`load_gold_prices` but in **USD per gram**.

    Wraps the underlying CSV's per-troy-ounce values with a single
    documented division so callers doing absolute valuation (the
    SGB engine, P&L reports, position sizing) don't have to remember
    the magic constant.
    """
    return load_gold_prices(csv_path) / GRAMS_PER_TROY_OUNCE
