"""Fund inauguration / defunct-status helpers.

Every NAV DataFrame fetched from mfapi.in has a date index. The earliest
date is the fund's inauguration (when it was launched); the latest date
is when it last reported a NAV. If the latest is older than a threshold
(default 30 days), the fund is treated as **defunct** — it stopped
reporting, so its contribution to the portfolio is frozen and the user
should be warned rather than silently aggregated with live data.

This module surfaces both pieces of information so:

- the per-portfolio metrics CSV can emit a sibling ``<portfolio>.assets.csv``
  with one row per asset (name, type, allocation, inauguration, status), and
- the plot's allocation table can show "Inaugurated" and "Closed" columns.
"""

from __future__ import annotations

import csv
import datetime as dt

import pandas as pd

_DEFUNCT_DEFAULT_DAYS = 30


def fund_dates(
    nav_df: pd.DataFrame,
    as_of: dt.date | None = None,
    defunct_threshold_days: int = _DEFUNCT_DEFAULT_DAYS,
) -> dict:
    """Return inauguration / last-NAV / status for one fund's NAV history.

    Args:
        nav_df: DataFrame indexed by date with at least one NAV column.
        as_of: Evaluation date for the staleness check. Defaults to
            ``dt.date.today()`` if not provided.
        defunct_threshold_days: A fund whose latest NAV is older than this
            is flagged DEFUNCT.

    Returns:
        ``{"inauguration": date|None, "last_nav": date|None,
           "days_since_last_nav": int|None, "status": "LIVE"|"DEFUNCT"|"N/A"}``
    """
    as_of = as_of or dt.date.today()
    if nav_df.empty:
        return {
            "inauguration": None,
            "last_nav": None,
            "days_since_last_nav": None,
            "status": "N/A",
        }
    inaug = nav_df.index.min().date()
    last = nav_df.index.max().date()
    days_since = (as_of - last).days
    status = "DEFUNCT" if days_since > defunct_threshold_days else "LIVE"
    return {
        "inauguration": inaug,
        "last_nav": last,
        "days_since_last_nav": days_since,
        "status": status,
    }


def build_assets_meta(
    portfolio_dict: dict,
    nav_by_name: dict[str, pd.DataFrame],
    as_of: dt.date | None = None,
    defunct_threshold_days: int = _DEFUNCT_DEFAULT_DAYS,
) -> list[dict]:
    """Build a uniform list of asset-metadata rows for the portfolio.

    For mutual funds, inauguration / last_nav / status come from the
    NAV DataFrames. For non-fund assets (PPF, gold, SCSS, SGB tranches)
    those fields are blank ("N/A") because there's no NAV history to
    introspect.
    """
    rows: list[dict] = []

    for fund in portfolio_dict.get("funds", []):
        name = fund.get("name", "Unknown Fund")
        df = nav_by_name.get(name, pd.DataFrame())
        lifecycle = fund_dates(df, as_of=as_of, defunct_threshold_days=defunct_threshold_days)
        rows.append({
            "name": name,
            "type": "Fund",
            "allocation": fund.get("allocation", 0.0),
            **lifecycle,
        })

    for key in ("ppf", "gold", "scss"):
        if key in portfolio_dict:
            entry = portfolio_dict[key]
            rows.append({
                "name": entry.get("name", key.upper()),
                "type": key.capitalize(),
                "allocation": entry.get("allocation", 0.0),
                "inauguration": None,
                "last_nav": None,
                "days_since_last_nav": None,
                "status": "N/A",
            })

    for entry in portfolio_dict.get("sgb", []):
        rows.append({
            "name": entry["tranche_id"],
            "type": "SGB",
            "allocation": entry.get("allocation", 0.0),
            "inauguration": None,
            "last_nav": None,
            "days_since_last_nav": None,
            "status": "N/A",
        })

    return rows


def _fmt_date(d) -> str:
    if d is None:
        return ""
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    return str(d)


def write_assets_csv(rows: list[dict], path: str) -> None:
    """Write one row per asset to ``path``."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["asset_type", "asset_name", "allocation",
             "inauguration_date", "last_nav_date", "status"]
        )
        for r in rows:
            w.writerow([
                r.get("type", ""),
                r.get("name", ""),
                f"{r.get('allocation', 0.0):.4f}",
                _fmt_date(r.get("inauguration")),
                _fmt_date(r.get("last_nav")),
                r.get("status", ""),
            ])
