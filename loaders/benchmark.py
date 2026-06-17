"""Load benchmark/index time series from CSV.

The function auto-detects a single date column (any column with "date"
in its name) and a single value column by priority order
(rate > price > close > yield), strips commas, converts to float, and
returns a ``TimeseriesReturn``. Optionally enforces a staleness gate.

Extracted from ``data_loader.load_timeseries_csv``; that name is kept
there as a thin re-export for backward compatibility.
"""

from __future__ import annotations

import os

import pandas as pd

from timeseries.returns import TimeseriesReturn
from utils import dbg, info

try:
    from main import DEBUG
except ImportError:
    DEBUG = False


def load_timeseries_csv(
    file_path: str,
    date_format: str,
    max_delay_days: int | None = None,
    as_of: pd.Timestamp | None = None,
) -> TimeseriesReturn:
    """Load a benchmark CSV and return a TimeseriesReturn.

    Args:
        file_path: Path to the CSV file.
        date_format: ``strftime``-style format for the date column.
        max_delay_days: If set, raise RuntimeError when the latest date
            in the CSV is older than today minus this many days. Use
            None to disable the staleness gate.

    Returns:
        TimeseriesReturn wrapping a float-valued Series indexed by date.
    """
    df = pd.read_csv(file_path)
    if DEBUG:
        info(f"📂 Loading time‑series «{file_path}»")

    date_cols = [c for c in df.columns if "date" in c.lower()]
    if len(date_cols) != 1:
        raise ValueError(f"Expected exactly one date column containing 'date'; found: {date_cols}")
    date_column = date_cols[0]

    parsed_dates = pd.to_datetime(df[date_column], format=date_format, errors="coerce")
    bad_rows = df[parsed_dates.isna()]
    if len(bad_rows):
        info(f"❗ Found {len(bad_rows)} bad date(s) in '{date_column}'. Sample:")
        info(bad_rows.head(min(5, len(bad_rows))).to_string(index=False))
        raise ValueError(f"{file_path}: date parsing failed. Check format: {date_format}")

    df[date_column] = parsed_dates
    df = df.set_index(date_column).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Expected DatetimeIndex, got {type(df.index)}")
    if DEBUG:
        info(f"Date index now ranges from {df.index.min().date()} to {df.index.max().date()}")

    col_priority = ["rate", "price", "close", "yield"]
    candidates = [
        col
        for name in col_priority
        for col in df.columns
        if name in col.lower() and col != date_column
    ]
    if not candidates:
        raise ValueError(
            f"{file_path}: no column found containing 'rate', 'price', 'close', or 'yield'"
        )
    if len(candidates) > 1:
        raise ValueError(f"{file_path}: multiple candidate value columns: {candidates}")
    df = df.rename(columns={candidates[0]: "value"})

    try:
        df["value"] = df["value"].astype(str).str.replace(",", "").astype(float)
    except ValueError as e:
        bad_rows = df[
            ~df["value"]
            .astype(str)
            .str.replace(".", "", 1)
            .str.replace("-", "", 1)
            .str.isnumeric()
        ]
        info("❗ Could not convert 'value' column to float. Sample of bad rows:")
        info(bad_rows.head(5).to_string(index=False))
        raise ValueError(f"{file_path}: 'value' column contains non-numeric entries.") from e

    if max_delay_days is not None:
        last_date = df.index[-1]
        as_of = as_of if as_of is not None else pd.Timestamp.today().normalize()
        expected_latest = (as_of - pd.Timedelta(days=max_delay_days)).replace(day=1)

        dbg(f'Latest date in "{os.path.basename(file_path)}": {last_date.date()}')
        dbg(f"Required minimum date: {expected_latest.date()}")

        if last_date < expected_latest:
            raise RuntimeError(
                f"{file_path}: data is outdated (last date: {last_date.date()}) — "
                f"fetch a fresh copy"
            )

    return TimeseriesReturn(df["value"])
