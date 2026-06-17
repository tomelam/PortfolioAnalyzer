"""Load and align risk-free rate series.

``fetch_and_standardize_risk_free_rates`` reads a CSV (typically the
India 10-Year Bond Yield series) via the shared benchmark CSV loader,
converts percent to decimal, and surfaces a plain Series.
``align_dynamic_risk_free_rates`` aligns that series to portfolio
return dates with time-interpolation and edge-fill so Sharpe/Sortino
math has a value for every trading day.

Extracted from ``data_loader``; the original names are kept there as
thin re-exports for backward compatibility.
"""

from __future__ import annotations

import pandas as pd

from loaders.benchmark import load_timeseries_csv
from utils import dbg, info

try:
    from main import DEBUG
except ImportError:
    DEBUG = False


def fetch_and_standardize_risk_free_rates(
    file_path: str,
    date_format: str,
    max_allowed_delay_days: int | None,
) -> pd.Series:
    """Read a risk-free-rate CSV (values in percent) and return decimals."""
    dbg(
        f'📂 Loading risk‑free series "{file_path}" '
        f"(max staleness {max_allowed_delay_days} days)"
    )
    try:
        ts = load_timeseries_csv(file_path, date_format, max_delay_days=max_allowed_delay_days)
        ts.set_series(ts.value_series() / 100.0)  # percent → decimal
        return ts.value_series()
    except Exception as e:
        raise ValueError(f"Failed to load risk-free rate data: {e}") from e


def align_dynamic_risk_free_rates(
    portfolio_returns: pd.Series,
    risk_free_data: pd.Series,
) -> pd.Series:
    """Time-interpolate risk-free rates onto the portfolio's date index."""
    assert isinstance(risk_free_data.index, pd.DatetimeIndex), (
        "Risk-free rate data index must be a DatetimeIndex. "
        f"Got: {type(risk_free_data.index).__name__}"
    )

    overlap = portfolio_returns.index.intersection(risk_free_data.index)
    if DEBUG:
        info(f"Risk-free rate series shape: {risk_free_data.shape}")
        idx = risk_free_data.index
        start = pd.to_datetime(idx.min(), errors="coerce")
        end = pd.to_datetime(idx.max(), errors="coerce")
        info(
            f"Index range: {start.date() if pd.notna(start) else 'NaT'} → "
            f"{end.date() if pd.notna(end) else 'NaT'}"
        )
        info(f"Common dates between portfolio and risk-free: {len(overlap)}")
    if len(overlap) < 10:
        info("⚠️ Very few or no overlapping dates — risk-free may not be aligned.")

    aligned = risk_free_data.reindex(portfolio_returns.index)
    rate_series = aligned.infer_objects(copy=False).interpolate(method="time")
    return rate_series.ffill().bfill()
