"""Pin the daily-frequency contract of combined_civ_series.

Some assets (gold, PPF) are sampled at a coarser cadence than mutual
fund NAVs (daily). The previous ``pd.concat(join="inner")`` collapsed
the portfolio CIV to the *intersection* of dates — effectively monthly
when gold or PPF was present. Downstream code then applied a
``sqrt(252)`` daily annualization to a monthly series, inflating
volatility/Sharpe ten-fold or more.

The contract: ``combined_civ_series`` produces a *daily* CIV by
forward-filling each asset's series onto the common business-day
calendar before weighting. This preserves the assets' natural
information content (gold doesn't move daily) while keeping the index
at a frequency consistent with how downstream metrics annualize.
"""

from __future__ import annotations

import pandas as pd

from timeseries.asset import from_civ
from timeseries.portfolio import PortfolioTimeseries


def test_combined_civ_is_daily_even_when_one_asset_is_monthly() -> None:
    """A monthly-sampled gold asset must not collapse the portfolio CIV
    to monthly granularity."""
    # Daily MF: 60 business days, +0.05%/day
    mf_dates = pd.bdate_range("2024-01-01", periods=60)
    mf = pd.Series([100.0 * (1.0005**i) for i in range(60)], index=mf_dates, name="value")

    # Monthly gold: 3 month-end observations within that window
    gold_dates = pd.DatetimeIndex(["2024-01-31", "2024-02-29", "2024-03-29"])
    gold = pd.Series([5000.0, 5100.0, 5200.0], index=gold_dates, name="value")

    assets = {"mf": from_civ(mf), "gold": from_civ(gold)}
    portfolio = PortfolioTimeseries(assets=assets, weights={"mf": 0.7, "gold": 0.3})

    civ = portfolio.combined_civ_series().series
    # Must be daily, not monthly. Overlap window is gold's first date
    # (2024-01-31) to the MF's last business day (~2024-03-22) → ~38 bdays.
    assert len(civ) >= 30, f"Expected daily granularity (~30+ points), got {len(civ)}"
    # Index should be business-day frequency.
    assert civ.index.freq == "B" or civ.index.freqstr == "B"


def test_daily_returns_dont_collapse_to_monthly_jumps() -> None:
    """With a daily MF and a monthly gold, daily returns should be small
    and frequent — not concentrated on the 3 gold-rebase dates."""
    mf_dates = pd.bdate_range("2024-01-01", periods=60)
    mf = pd.Series([100.0 * (1.0005**i) for i in range(60)], index=mf_dates, name="value")

    gold_dates = pd.DatetimeIndex(["2024-01-31", "2024-02-29", "2024-03-29"])
    gold = pd.Series([5000.0, 5100.0, 5200.0], index=gold_dates, name="value")

    assets = {"mf": from_civ(mf), "gold": from_civ(gold)}
    portfolio = PortfolioTimeseries(assets=assets, weights={"mf": 0.7, "gold": 0.3})

    civ = portfolio.combined_civ_series().series
    daily_returns = civ.pct_change().dropna()

    # MF moves +0.05%/day with weight 0.7 → portfolio should move
    # ~0.035% on non-gold-rebase days, larger on gold-rebase days.
    # The MAX daily return must not be a huge jump (>5%): that would
    # indicate the series was inner-joined to monthly and pct_change
    # picked up monthly moves as if daily.
    assert daily_returns.abs().max() < 0.05, (
        f"Daily returns contain monthly-scale jumps "
        f"(max abs return {daily_returns.abs().max():.4f}); "
        "the CIV was likely inner-joined to monthly dates."
    )
