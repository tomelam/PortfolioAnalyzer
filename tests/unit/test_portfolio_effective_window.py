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
import pytest

from timeseries.asset import from_civ
from timeseries.portfolio import PortfolioTimeseries, from_multiple_nav_series


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


# --- guard branches --------------------------------------------------------

def test_constructor_rejects_empty_assets() -> None:
    """The empty-portfolio case is forbidden at construction, which is why
    the downstream methods don't need their own empty-assets guards."""
    with pytest.raises(ValueError, match="at least one asset"):
        PortfolioTimeseries(assets={}, weights={})


def test_combined_civ_series_non_overlapping_assets_is_empty() -> None:
    """Assets whose date windows don't overlap (latest start > earliest end)
    yield an empty, validly-named portfolio CIV rather than crashing the
    TimeseriesCIV constructor (regression: the empty return used an unnamed
    Series, which TimeseriesCIV rejects)."""
    early = _series([100.0 + i for i in range(60)], "2020-01-01")
    late = _series([100.0 + i for i in range(60)], "2024-01-01")
    portfolio = PortfolioTimeseries(
        assets={"early": from_civ(early), "late": from_civ(late)},
        weights={"early": 0.5, "late": 0.5},
    )
    civ = portfolio.combined_civ_series()
    assert civ.series.empty
    assert civ.series.name == "value"


def test_from_multiple_nav_series_skips_none_and_non_series() -> None:
    """None entries and non-Series values are dropped, not turned into assets."""
    good = _series([100.0, 101.0, 102.0], "2024-01-01")
    portfolio = from_multiple_nav_series(
        {"good": good, "missing": None, "garbage": [1, 2, 3]},
        weights={"good": 1.0},
    )
    assert list(portfolio.assets) == ["good"]
