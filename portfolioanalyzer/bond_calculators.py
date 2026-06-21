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
from dateutil.relativedelta import relativedelta


def term_locked_rate_series(rate, anchor_date, end_date, term_years, freq="B"):
    """Daily applicable-rate series under a term-locked rollover.

    Models how a fixed-term scheme (SCSS) actually behaves: the annual rate is
    **locked at account opening for the whole term** and re-looked-up only when
    the holding rolls over into a fresh term. Subsequent intra-term revisions to
    the published rate do **not** touch an open account — they are seen only at
    the next rollover boundary.

    Args:
        rate: ``pd.Series`` indexed by date carrying the published annual rate
            in percent (sparse — rows on rate *changes* only).
        anchor_date: The original (re)investment date; rollover boundaries are
            laid out as ``anchor + k·term``.
        end_date: Last date the holding is modelled through.
        term_years: Term length in years (e.g. 5 for SCSS).
        freq: Output calendar frequency (business days by default).

    Returns:
        ``pd.Series`` over ``[anchor_date, end_date]`` at ``freq``, giving the
        locked annual rate in effect on each day — a step function that changes
        only at rollover boundaries.
    """
    anchor = pd.Timestamp(anchor_date)
    end = pd.Timestamp(end_date)

    rate = rate[rate.index.notna()].sort_index()
    rate = rate[~rate.index.duplicated(keep="first")]

    # Rollover boundaries anchored at the opening date. Use months so a
    # fractional term (rare) still lands on a sensible date.
    term_months = int(round(float(term_years) * 12))
    boundaries = []
    k = 0
    while True:
        boundary = anchor + relativedelta(months=term_months * k)
        if boundary > end:
            break
        boundaries.append(boundary)
        k += 1

    boundary_idx = pd.DatetimeIndex(boundaries)
    # Locked rate per term = the published rate as-of each boundary. A boundary
    # predating the rate history (anchor before the first published rate) gets
    # the earliest available rate.
    locked = rate.reindex(boundary_idx, method="ffill")
    if not rate.empty:
        locked = locked.fillna(rate.iloc[0])

    daily = pd.date_range(anchor, end, freq=freq)
    return locked.reindex(daily, method="ffill")


def calculate_variable_bond_cumulative_gain(
    rate_df, portfolio_start_date, *, term_years=None, anchor_date=None
):
    """Daily cumulative-gain series from a time-varying annual rate.

    Two valuation modes:

    - ``term_years is None`` (default, generic): the rate is applied
      **continuously** — every published revision flows onto the balance on its
      effective date. This is the legacy behaviour, kept for back-compat.
    - ``term_years`` set (SCSS): **term-locked rollover** — the rate is locked
      per ``term_years``-year term and re-looked-up only at each rollover
      boundary (``term_locked_rate_series``). Reinvestment is implicit: the
      holding keeps compounding past each maturity into a fresh term, which is
      required for the series to characterise the asset (rather than decaying to
      idle cash) over a backtest window.

    Args:
        rate_df: DataFrame with a DatetimeIndex and a column named
            ``"interest"`` or ``"rate"`` carrying the annual interest rate
            in percent. Duplicate or NaN-indexed rows are dropped.
        portfolio_start_date: When to start compounding; clamped forward
            to the first available rate date if it precedes the rate series.
        term_years: Term length in years for the lock; ``None`` → continuous.
        anchor_date: First-term anchor (the holding's open date) for the lock;
            ``None`` → the effective start. Ignored when ``term_years is None``.

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

    if term_years is None:
        dates = pd.date_range(start=effective_start_date, end=end_date, freq="B")
        rate_series = rate_df["interest"].reindex(dates, method="ffill")
    else:
        anchor = (
            pd.to_datetime(anchor_date) if anchor_date is not None else effective_start_date
        )
        rate_series = term_locked_rate_series(
            rate_df["interest"], anchor, end_date, term_years, freq="B"
        )
        # Can't compound before the first published rate.
        rate_series = rate_series[rate_series.index >= rate_df.index.min()]

    # Annual percent → daily effective return.
    daily_rate = (1 + rate_series / 100) ** (1 / 252) - 1

    cum_series = (1 + daily_rate).cumprod()
    return cum_series.asfreq("B", method="ffill")
