"""Tests for ``PortfolioTimeseries.effective_window``.

The portfolio's effective window is bounded by:
- start: the latest ``index.min()`` across assets (last asset to come online), and
- end:   the earliest ``index.max()`` across assets (first asset to go dark).

The window-setting asset on each end is surfaced so ``main.py`` can print a
banner explaining why the displayed window differs from "today". Without
this surface, a portfolio containing one defunct fund silently plots up to
that fund's last NAV — useful behavior, but only if the user is told.
"""

from __future__ import annotations

import pandas as pd

from asset_timeseries import from_civ
from portfolio_timeseries import PortfolioTimeseries


def _series(values: list[float], start: str) -> pd.Series:
    return pd.Series(
        values,
        index=pd.bdate_range(start, periods=len(values)),
        name="value",
    )


def test_effective_window_clips_end_at_defunct_asset() -> None:
    """A defunct fund (last NAV well in the past) sets the portfolio end."""
    live = _series([100.0 + i for i in range(500)], "2024-01-01")  # extends ~2 yrs
    defunct = _series([100.0 + i for i in range(50)], "2024-01-01")  # 50 bdays only

    portfolio = PortfolioTimeseries(
        assets={"live_fund": from_civ(live), "defunct_fund": from_civ(defunct)},
        weights={"live_fund": 0.5, "defunct_fund": 0.5},
    )
    window = portfolio.effective_window()

    assert window["end"] == defunct.index.max()
    assert window["end_limited_by"] == "defunct_fund"


def test_effective_window_starts_at_latest_inception() -> None:
    """Symmetric: portfolio can't start before its youngest asset's launch."""
    veteran = _series([100.0 + i for i in range(300)], "2020-01-01")
    rookie = _series([100.0 + i for i in range(100)], "2024-06-03")

    portfolio = PortfolioTimeseries(
        assets={"veteran": from_civ(veteran), "rookie": from_civ(rookie)},
        weights={"veteran": 0.5, "rookie": 0.5},
    )
    window = portfolio.effective_window()

    assert window["start"] == rookie.index.min()
    assert window["start_limited_by"] == "rookie"


def test_effective_window_with_all_fresh_assets_unchanged() -> None:
    """All-fresh portfolios are not artificially shortened: end matches the
    fund with the earliest last-NAV in the set (yesterday-vs-today noise),
    not some legacy hard-coded cutoff."""
    today_ish = _series([100.0 + i for i in range(252)], "2024-01-01")
    one_day_short = _series([100.0 + i for i in range(251)], "2024-01-01")

    portfolio = PortfolioTimeseries(
        assets={"a": from_civ(today_ish), "b": from_civ(one_day_short)},
        weights={"a": 0.5, "b": 0.5},
    )
    window = portfolio.effective_window()

    assert window["end"] == one_day_short.index.max()
    assert window["end_limited_by"] == "b"
    assert window["start"] == today_ish.index.min()


def test_effective_window_matches_combined_civ_series_bounds() -> None:
    """``effective_window`` must agree with the bounds the actual portfolio
    CIV gets computed on; otherwise the banner would lie about the data
    the metrics were derived from."""
    a = _series([100.0 + i for i in range(200)], "2023-01-02")
    b = _series([100.0 + i for i in range(120)], "2023-06-01")

    portfolio = PortfolioTimeseries(
        assets={"a": from_civ(a), "b": from_civ(b)},
        weights={"a": 0.5, "b": 0.5},
    )
    window = portfolio.effective_window()
    civ = portfolio.combined_civ_series().series

    assert civ.index.min() == window["start"]
    assert civ.index.max() == window["end"]
