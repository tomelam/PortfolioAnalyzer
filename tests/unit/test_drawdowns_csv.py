"""Tests for the per-drawdown CSV writer.

The single-row portfolio metrics CSV reports only the *worst* drawdown.
Users who want to see every recovered drawdown (depth, dates, duration)
get them via the sibling ``<portfolio>.drawdowns.csv`` produced alongside.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from drawdowns_csv import write_drawdowns_csv


def _sample_drawdowns() -> list[dict]:
    """Three drawdowns of varying shapes — matches the record schema
    emitted by ``TimeseriesCIV.max_drawdowns``."""
    return [
        {
            "start": pd.Timestamp("2020-02-19"),
            "end": pd.Timestamp("2020-06-12"),
            "trough": pd.Timestamp("2020-03-23"),
            "drawdown": 0.2761,
            "drawdown_days": 33,
            "recovery_days": 81,
        },
        {
            "start": pd.Timestamp("2021-11-16"),
            "end": pd.Timestamp("2023-06-12"),
            "trough": pd.Timestamp("2022-06-15"),
            "drawdown": 0.1923,
            "drawdown_days": 211,
            "recovery_days": 362,
        },
        # An unrecovered final drawdown — recovery_days=None.
        {
            "start": pd.Timestamp("2026-01-06"),
            "end": pd.Timestamp("2026-05-06"),
            "trough": pd.Timestamp("2026-04-15"),
            "drawdown": 0.10,
            "drawdown_days": 99,
            "recovery_days": None,
        },
    ]


def test_writes_header_and_one_row_per_drawdown(tmp_path: Path) -> None:
    out = tmp_path / "port-1.drawdowns.csv"
    write_drawdowns_csv(_sample_drawdowns(), str(out))
    rows = out.read_text().splitlines()
    # 1 header + 3 data rows
    assert len(rows) == 4
    assert rows[0].startswith("start_date,trough_date,recovery_date,depth_pct")


def test_depth_pct_is_negative_percent_form(tmp_path: Path) -> None:
    """User expectation: depth_pct mirrors the human-readable stdout
    report (e.g. ``-19.23%`` for a 19.23% drawdown)."""
    out = tmp_path / "x.drawdowns.csv"
    write_drawdowns_csv(_sample_drawdowns(), str(out))
    df = pd.read_csv(out)
    # The 0.1923 drawdown is the second row.
    assert df.iloc[1]["depth_pct"] == -19.23


def test_unrecovered_drawdown_has_blank_recovery_date(tmp_path: Path) -> None:
    out = tmp_path / "x.drawdowns.csv"
    write_drawdowns_csv(_sample_drawdowns(), str(out))
    rows = out.read_text().splitlines()
    # Third data row is the unrecovered one; recovery_date column should be empty.
    last = rows[-1].split(",")
    assert last[2] == ""  # recovery_date column blank
    assert last[5] == ""  # recovery_days column blank


def test_empty_drawdowns_writes_just_header(tmp_path: Path) -> None:
    out = tmp_path / "x.drawdowns.csv"
    write_drawdowns_csv([], str(out))
    assert out.read_text().splitlines() == [
        "start_date,trough_date,recovery_date,depth_pct,drawdown_days,recovery_days"
    ]


def test_dates_are_iso(tmp_path: Path) -> None:
    out = tmp_path / "x.drawdowns.csv"
    dd = [{
        "start": dt.date(2020, 2, 19),
        "end": dt.date(2020, 6, 12),
        "trough": dt.date(2020, 3, 23),
        "drawdown": 0.10,
        "drawdown_days": 33,
        "recovery_days": 81,
    }]
    write_drawdowns_csv(dd, str(out))
    rows = out.read_text().splitlines()
    cols = rows[1].split(",")
    assert cols[0] == "2020-02-19"
    assert cols[1] == "2020-03-23"
    assert cols[2] == "2020-06-12"
