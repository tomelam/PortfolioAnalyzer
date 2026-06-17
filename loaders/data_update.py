"""Auto-update benchmark / risk-free data from stable, no-auth sources.

PortfolioAnalyzer reads benchmark and risk-free series from CSVs in
``data/``. Left alone those files go stale (the staleness gate then blocks
runs). This module refreshes them from upstream feeds so the data stays
current without manual investing.com downloads.

Design
------
Each :class:`DataSource` maps one upstream feed to a local CSV that the
pipeline's ``load_timeseries_csv`` already knows how to read. A fetcher
returns a date-indexed ``value`` DataFrame; :func:`write_normalized_csv`
writes it with a ``date``-named column and a loader-recognized value column
(``rate``/``price``). :func:`update_all` refreshes every source, writes a
last-fetched stamp, and isolates failures so one dead feed can't abort the
rest. The whole thing is cron-able via the ``portfolio-analyzer-update``
console script (see :mod:`data_update_cli`).

Sources
-------
- ``risk_free_fred`` → FRED ``INDIRLTLT01STM`` (India 10Y government bond
  rate, monthly). Stable, no-auth, machine-readable CSV — fully verifiable.
- ``benchmark_nifty_tri`` → NIFTY 50 TRI from niftyindices.com. The parser
  is unit-tested, but niftyindices is anti-scraping and often unreachable
  from non-browser / CI contexts; **the live fetch must be verified in an
  environment that can reach the host** before relying on it.
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import requests

DATA_DIR = "data"
STAMP_FILE = os.path.join(DATA_DIR, ".last_fetched.json")

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
# niftyindices historical-data CSV export (NIFTY 50 TRI). Subject to change
# and anti-scraping; kept here as the documented upstream.
NIFTY_TRI_URL = (
    "https://www.niftyindices.com/IndexConstituent/"
    "ind_close_all_TRI.csv"
)

# FRED and most feeds are happy with the default ``requests`` User-Agent
# (matching loaders.mutual_fund); we only override it where a host demands
# a browser-like agent (niftyindices). Some hosts/proxies actively reject
# *custom* UAs, so we don't invent one.
# niftyindices rejects non-browser agents; a browser-like UA is required and
# still may not be enough from a datacenter IP.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://www.niftyindices.com/",
}


# --- pure parsers ----------------------------------------------------------

def parse_fred_csv(text: str) -> pd.DataFrame:
    """Parse a FRED ``fredgraph`` CSV into a date-indexed ``value`` frame.

    FRED CSVs are ``observation_date,<SERIES_ID>`` with ISO dates and ``.``
    for missing observations. Missing/NaN rows are dropped.
    """
    df = pd.read_csv(io.StringIO(text))
    if df.shape[1] < 2:
        raise ValueError("FRED CSV must have at least two columns")
    date_col, val_col = df.columns[0], df.columns[1]
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "value": pd.to_numeric(df[val_col], errors="coerce"),  # '.' -> NaN
        }
    ).dropna()
    out = out.set_index("date").sort_index()
    if out.empty:
        raise ValueError("FRED CSV parsed to zero rows")
    return out


def parse_niftyindices_tri_csv(text: str) -> pd.DataFrame:
    """Parse a niftyindices TRI history CSV into a date-indexed ``value`` frame.

    niftyindices column names vary across endpoints; we auto-detect a date
    column and the total-returns/closing value column, strip comma
    thousands separators, and parse day-first dates.
    """
    df = pd.read_csv(io.StringIO(text))
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if not date_cols:
        raise ValueError(f"no date column in niftyindices CSV: {list(df.columns)}")
    val_priority = [
        "total returns index",
        "closing index value",
        "closing",
        "close",
        "index value",
        "price",
    ]
    val_col = next(
        (c for key in val_priority for c in df.columns if key in c.lower()),
        None,
    )
    if val_col is None:
        raise ValueError(f"no value column in niftyindices CSV: {list(df.columns)}")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_cols[0]], dayfirst=True, errors="coerce"),
            "value": pd.to_numeric(
                df[val_col].astype(str).str.replace(",", "", regex=False), errors="coerce"
            ),
        }
    ).dropna()
    out = out.set_index("date").sort_index()
    if out.empty:
        raise ValueError("niftyindices CSV parsed to zero rows")
    return out


# --- network fetchers ------------------------------------------------------

def _get(url: str, *, session=None, timeout: int = 30, headers: dict | None = None) -> str:
    # headers=None → requests' default User-Agent (works for FRED & mfapi-style
    # feeds). Only niftyindices passes explicit browser headers.
    resp = (session or requests).get(url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    return resp.text


def fetch_fred_series(series_id: str, *, session=None) -> pd.DataFrame:
    """Fetch a FRED series as a date-indexed ``value`` frame."""
    return parse_fred_csv(_get(FRED_CSV_URL.format(series_id=series_id), session=session))


def fetch_niftyindices_tri(url: str = NIFTY_TRI_URL, *, session=None) -> pd.DataFrame:
    """Fetch the NIFTY 50 TRI history from niftyindices.com.

    Sends browser-like headers; may still be blocked from non-browser IPs.
    """
    return parse_niftyindices_tri_csv(_get(url, session=session, headers=_BROWSER_HEADERS))


# --- sources + writer ------------------------------------------------------

@dataclass
class DataSource:
    """One refreshable feed → one local CSV the pipeline reads."""

    name: str
    target_path: str
    fetch: Callable[..., pd.DataFrame]
    out_date_col: str  # must contain "date" for load_timeseries_csv
    out_value_col: str  # one of: rate / price / close / yield
    out_date_format: str  # strftime; must match the config's date_format
    description: str


def write_normalized_csv(source: DataSource, df: pd.DataFrame) -> None:
    """Write a date-indexed ``value`` frame to ``source.target_path`` in the
    loader-compatible schema (``out_date_col``, ``out_value_col``)."""
    out = df.reset_index()
    out.columns = [source.out_date_col, source.out_value_col]
    out[source.out_date_col] = pd.to_datetime(out[source.out_date_col]).dt.strftime(
        source.out_date_format
    )
    parent = os.path.dirname(source.target_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    out.to_csv(source.target_path, index=False)


REGISTRY: dict[str, DataSource] = {
    "risk_free_fred": DataSource(
        name="risk_free_fred",
        target_path=os.path.join(DATA_DIR, "INDIRLTLT01STM.csv"),
        fetch=lambda session=None: fetch_fred_series("INDIRLTLT01STM", session=session),
        out_date_col="observation_date",
        out_value_col="rate",
        out_date_format="%Y-%m-%d",
        description="FRED India 10Y government bond rate (monthly); risk-free proxy.",
    ),
    "benchmark_nifty_tri": DataSource(
        name="benchmark_nifty_tri",
        target_path=os.path.join(DATA_DIR, "NIFTY Total Returns Historical Data.csv"),
        fetch=lambda session=None: fetch_niftyindices_tri(session=session),
        out_date_col="Date",
        out_value_col="Price",
        out_date_format="%m/%d/%Y",
        description="NIFTY 50 TRI from niftyindices.com (live fetch unverified in CI).",
    ),
}


# --- stamps ----------------------------------------------------------------

def _read_stamps() -> dict:
    try:
        with open(STAMP_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_stamp(name: str, last_data_date: str, rows: int) -> None:
    stamps = _read_stamps()
    stamps[name] = {
        "fetched_at": pd.Timestamp.utcnow().isoformat(),
        "last_date": last_data_date,
        "rows": rows,
    }
    parent = os.path.dirname(STAMP_FILE)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(STAMP_FILE, "w", encoding="utf-8") as f:
        json.dump(stamps, f, indent=2, sort_keys=True)


# --- update orchestration --------------------------------------------------

def update_source(name: str, *, session=None) -> dict:
    """Fetch one source, write its CSV, and stamp it. Raises on fetch error."""
    src = REGISTRY[name]
    df = src.fetch(session=session)
    write_normalized_csv(src, df)
    last_date = df.index.max().date().isoformat()
    _write_stamp(name, last_date, len(df))
    return {"name": name, "ok": True, "rows": len(df), "last_date": last_date}


def update_all(*, session=None) -> list[dict]:
    """Refresh every registered source. A failing source is reported but
    never aborts the others."""
    results = []
    for name in REGISTRY:
        try:
            results.append(update_source(name, session=session))
        except Exception as e:  # noqa: BLE001 — one dead feed must not kill the rest
            results.append({"name": name, "ok": False, "error": str(e)})
    return results
