"""Variable-rate bond cumulative-gain series.

Only ``calculate_variable_bond_cumulative_gain`` is live (called by
``main.py`` via the SCSS path). Four other functions used to live here:

- ``calculate_bond_cumulative_gain`` — fixed-rate bond (uncalled).
- ``calculate_sgb_cumulative_gain``, ``calculate_merged_sgb_series``,
  ``calculate_realistic_sgb_series`` — pre-refactor SGB approximations,
  all superseded by ``sgb_holdings.sgb_holding_civ`` (the per-tranche
  IBJA-spot + coupon-accrual engine introduced 2026-06-16).

The dead code was deleted in cycle 5 of the post-v0.1 cleanup
(KANBAN: "Audit bond_calculators.py").
"""

import pandas as pd


def calculate_variable_bond_cumulative_gain(rate_df, portfolio_start_date):
    """Daily cumulative-gain series from a time-varying annual rate.

    Args:
        rate_df: DataFrame with a DatetimeIndex and a column named
            ``"interest"`` or ``"rate"`` carrying the annual interest rate
            in percent. Duplicate or NaN-indexed rows are dropped.
        portfolio_start_date: When to start compounding; clamped forward
            to the first available rate date if it precedes the rate series.

    Returns:
        A daily-indexed ``pd.Series`` of cumulative growth factors
        (start ≈ 1.0).
    """
    if "interest" not in rate_df.columns and "rate" in rate_df.columns:
        rate_df = rate_df.rename(columns={"rate": "interest"})

    rate_df = rate_df[rate_df.index.notna()]
    rate_df = rate_df[~rate_df.index.duplicated(keep="first")]

    portfolio_start_date = pd.to_datetime(portfolio_start_date)
    end_date = pd.Timestamp.today()
    effective_start_date = max(portfolio_start_date, rate_df.index.min())

    dates = pd.date_range(start=effective_start_date, end=end_date, freq="B")
    rate_series = rate_df["interest"].reindex(dates, method="ffill")

    # Annual percent → daily effective return.
    daily_rate = (1 + rate_series / 100) ** (1 / 252) - 1

    cum_series = (1 + daily_rate).cumprod()
    return cum_series.asfreq("B", method="ffill")
