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
