"""
civ_to_returns.py

Simple utilities for converting a Cumulative Investment Value (CIV) series
(e.g., NAVs, PPF balances, bond accruals) into a return series.

Main functions:
- civ_to_returns(): Converts monthly or daily NAV/CIV values into returns.

Usage:
- Called after generating or loading CIV data.
- Prepares returns for timeseries-based performance analysis.

Notes:
- Assumes CIV input is a pandas Series with a datetime index.
- Returns are percent change per period, not annualized.
"""
import pandas as pd

def civ_to_returns(series: pd.Series, frequency: str = "monthly") -> pd.Series:
    """
    Convert a NAV/CIV series into a returns series.
    
    Args:
        series (pd.Series): The NAV/CIV series.
        frequency (str): 'monthly' (default) or 'daily'.
    
    Returns:
        pd.Series: Returns series.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"Expected pd.Series, got {type(series)}")

    series = series.sort_index()

    if frequency == "monthly":
        resampled = series.resample("ME").last()
    elif frequency == "daily":
        resampled = series
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")

    returns = resampled.pct_change().dropna()

    return returns
