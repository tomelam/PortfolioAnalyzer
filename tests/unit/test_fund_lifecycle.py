"""Tests for the fund lifecycle helpers.

For every fund the portfolio analyzer fetches, we know:
- The earliest NAV date (the fund's inauguration / launch date).
- The most recent NAV date (when it last reported).

A fund whose latest NAV is older than a configurable threshold (default
30 days) is treated as **defunct** — it's no longer reporting, so its
contribution to the portfolio is frozen and the user should be warned
rather than silently included.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from portfolioanalyzer.fund_lifecycle import build_assets_meta, fund_dates, write_assets_csv


def _navs(dates: list[str], values: list[float] | None = None) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="date")
    return pd.DataFrame({"nav": values or [100.0] * len(dates)}, index=idx)


def test_fund_dates_inauguration_is_earliest_nav() -> None:
    df = _navs(["2010-04-15", "2010-04-16", "2026-06-15"])
    meta = fund_dates(df, as_of=dt.date(2026, 6, 17))
    assert meta["inauguration"] == dt.date(2010, 4, 15)


def test_fund_dates_last_nav_is_latest() -> None:
    df = _navs(["2010-04-15", "2026-06-15"])
    meta = fund_dates(df, as_of=dt.date(2026, 6, 17))
    assert meta["last_nav"] == dt.date(2026, 6, 15)


def test_fund_dates_status_live_when_recent() -> None:
    df = _navs(["2010-04-15", "2026-06-15"])
    meta = fund_dates(df, as_of=dt.date(2026, 6, 17), defunct_threshold_days=30)
    assert meta["status"] == "LIVE"
    assert meta["days_since_last_nav"] == 2


def test_fund_dates_status_defunct_when_too_stale() -> None:
    df = _navs(["2010-04-15", "2024-01-01"])
    meta = fund_dates(df, as_of=dt.date(2026, 6, 17), defunct_threshold_days=30)
    assert meta["status"] == "DEFUNCT"
    assert meta["days_since_last_nav"] > 30


def test_fund_dates_handles_empty_df() -> None:
    df = pd.DataFrame({"nav": []}, index=pd.DatetimeIndex([], name="date"))
    meta = fund_dates(df, as_of=dt.date(2026, 6, 17))
    assert meta["status"] == "N/A"
    assert meta["inauguration"] is None
    assert meta["last_nav"] is None


def test_write_assets_csv_one_row_per_asset(tmp_path: Path) -> None:
    out = tmp_path / "port.assets.csv"
    rows = [
        {"name": "Fund A", "type": "Fund", "allocation": 0.30,
         "inauguration": dt.date(2010, 4, 15),
         "last_nav": dt.date(2026, 6, 15),
         "status": "LIVE"},
        {"name": "Defunct Fund", "type": "Fund", "allocation": 0.20,
         "inauguration": dt.date(2008, 1, 1),
         "last_nav": dt.date(2023, 5, 1),
         "status": "DEFUNCT"},
        {"name": "PPF", "type": "Ppf", "allocation": 0.50,
         "inauguration": None, "last_nav": None, "status": "N/A"},
    ]
    write_assets_csv(rows, str(out))
    lines = out.read_text().splitlines()
    assert lines[0] == "asset_type,asset_name,allocation,inauguration_date,last_nav_date,status"
    assert len(lines) == 4
    # Inspect rows literally — pandas read_csv would auto-NaN the "N/A".
    fields = [line.split(",") for line in lines[1:]]
    assert fields[0][5] == "LIVE"
    assert fields[1][5] == "DEFUNCT"
    assert fields[2][5] == "N/A"
    # PPF row has empty date cells.
    assert fields[2][3] == ""
    assert fields[2][4] == ""


def test_build_assets_meta_combines_portfolio_and_navs(tmp_path: Path) -> None:
    """Integration: a portfolio dict + a map of fund-name → NAV DataFrame
    yields the asset-meta list the CSV writer expects."""
    portfolio = {
        "funds": [
            {"name": "Fund A", "allocation": 0.40},
            {"name": "Fund B", "allocation": 0.30},
        ],
        "ppf": {"name": "PPF", "allocation": 0.10},
        "sgb": [
            {"tranche_id": "2020-21-VII", "units_grams": 6, "allocation": 0.20},
        ],
    }
    navs = {
        "Fund A": _navs(["2015-01-01", "2026-06-15"]),
        "Fund B": _navs(["2018-03-15", "2023-12-01"]),  # defunct: stale
    }
    rows = build_assets_meta(portfolio, navs, as_of=dt.date(2026, 6, 17))

    by_name = {r["name"]: r for r in rows}
    assert by_name["Fund A"]["status"] == "LIVE"
    assert by_name["Fund B"]["status"] == "DEFUNCT"
    assert by_name["PPF"]["status"] == "N/A"
    assert by_name["2020-21-VII"]["type"] == "SGB"
    assert by_name["2020-21-VII"]["allocation"] == pytest.approx(0.20)
