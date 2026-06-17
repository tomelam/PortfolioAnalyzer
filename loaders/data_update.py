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

import datetime as dt
import io
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import requests

from loaders.benchmark import load_timeseries_csv

DATA_DIR = "data"
STAMP_FILE = os.path.join(DATA_DIR, ".last_fetched.json")

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# niftyindices serves its Total-Returns-Index history from a JSON POST API
# behind an ASP.NET session + ARR-affinity cookie wall. A naive GET returns
# an anti-scrape HTML page; the working recipe is: (1) GET the historical-data
# page with a browser User-Agent to mint the session cookies, then (2) POST
# the date range to getTotalReturnIndexString on the *same* session. The
# response is a ``{"d": "<json-array-string>"}`` envelope.
NIFTY_HIST_PAGE = "https://www.niftyindices.com/reports/historical-data"
NIFTY_TRI_ENDPOINT = "https://www.niftyindices.com/Backpage.aspx/getTotalReturnIndexString"
# FRED and most feeds are happy with the default ``requests`` User-Agent
# (matching loaders.mutual_fund); we only override it where a host demands a
# browser-like agent. Some hosts/proxies actively reject *custom* UAs, so we
# don't invent one — we present a real browser string only to niftyindices.
_NIFTY_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_NIFTY_POST_HEADERS = {
    "User-Agent": _NIFTY_UA,
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "X-Requested-With": "XMLHttpRequest",
    "Content-Type": "application/json; charset=UTF-8",
    "Referer": NIFTY_HIST_PAGE,
    "Origin": "https://www.niftyindices.com",
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


def parse_niftyindices_tri_json(text: str) -> pd.DataFrame:
    """Parse the niftyindices TRI JSON response into a date-indexed frame.

    The endpoint returns ``{"d": "<json-array-string>"}`` where each record
    has ``Date`` ("12 Jun 2026") and ``TotalReturnsIndex`` (string number).
    Returns a DataFrame indexed by a ``date`` DatetimeIndex with a single
    float ``value`` column.
    """
    envelope = json.loads(text)
    if "d" not in envelope:
        raise ValueError("niftyindices response missing 'd' envelope key")
    records = json.loads(envelope["d"])
    if not records:
        raise ValueError("niftyindices response contained zero records")
    df = pd.DataFrame(records)
    if "Date" not in df.columns or "TotalReturnsIndex" not in df.columns:
        raise ValueError(f"unexpected niftyindices columns: {list(df.columns)}")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"], format="%d %b %Y", errors="coerce"),
            "value": pd.to_numeric(
                df["TotalReturnsIndex"].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            ),
        }
    ).dropna()
    out = out.set_index("date").sort_index()
    if out.empty:
        raise ValueError("niftyindices TRI parsed to zero rows")
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


def fetch_niftyindices_tri(
    *,
    index_name: str = "NIFTY 50",
    start: str = "01-Jan-2007",
    end: str | None = None,
    session=None,
    retries: int = 4,
    timeout: int = 30,
    backoff: float = 1.5,
) -> pd.DataFrame:
    """Fetch the NIFTY 50 Total-Returns-Index history from niftyindices.com.

    Defeats the anti-scrape wall by priming an ASP.NET session on the
    historical-data page, then POSTing the date range to the TRI endpoint on
    the same session (a single request returns the full range — no chunking).

    Args:
        index_name: niftyindices index name (default "NIFTY 50").
        start: inclusive start, ``DD-Mon-YYYY`` (default covers full history).
        end: inclusive end, ``DD-Mon-YYYY``; defaults to today.
        session: optional pre-built requests Session (for tests / reuse).
        retries: attempts for the POST before giving up.
    """
    end = end or dt.date.today().strftime("%d-%b-%Y")
    # cinfo is a (single-quoted) JSON string embedded inside the JSON body;
    # built by concatenation to avoid brace-escaping noise.
    cinfo = (
        "{'name':'" + index_name + "','startDate':'" + start
        + "','endDate':'" + end + "','indexName':'" + index_name + "'}"
    )
    body = json.dumps({"cinfo": cinfo})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            # niftyindices is flaky; retry the WHOLE flow (cookie-prime + POST)
            # on a fresh session so a hung prime can't strand the attempt.
            sess = session or requests.Session()
            sess.headers.setdefault("User-Agent", _NIFTY_UA)
            sess.get(NIFTY_HIST_PAGE, timeout=timeout)  # mint session cookies
            resp = sess.post(
                NIFTY_TRI_ENDPOINT, data=body, headers=_NIFTY_POST_HEADERS, timeout=timeout
            )
            resp.raise_for_status()
            return parse_niftyindices_tri_json(resp.text)
        except Exception as e:  # noqa: BLE001 — retry transient/anti-scrape failures
            last_error = e
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(
        f"niftyindices TRI fetch failed after {retries} attempts: {last_error}"
    ) from last_error


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


# --- staleness-driven refresh (the on-run "force refresh when stale" path) -

def source_for_path(path: str) -> DataSource | None:
    """Return the registered source that writes ``path`` (basename match)."""
    base = os.path.basename(path)
    return next(
        (s for s in REGISTRY.values() if os.path.basename(s.target_path) == base), None
    )


def local_last_date(source: DataSource) -> pd.Timestamp | None:
    """Latest date currently on disk for ``source``'s file, or None if the
    file is missing/unreadable."""
    try:
        ts = load_timeseries_csv(source.target_path, source.out_date_format, max_delay_days=None)
        return ts.index.max()
    except Exception:  # noqa: BLE001 — missing/garbled file ⇒ treat as "no data"
        return None


def refresh_path_if_stale(
    path: str, max_age_days: int, *, session=None, today=None
) -> dict:
    """Force-refresh the source feeding ``path`` if its local data is older
    than ``max_age_days``. Returns a status dict with a user-facing
    ``message`` (empty when nothing to say).

    Statuses: ``fresh`` (already current), ``refreshed`` (pulled new data),
    ``failed`` (upstream unavailable — warn and keep existing data, an
    early-warning rather than a hard block), ``no_source`` (``path`` has no
    registered upstream, so it can't be auto-updated).
    """
    today = (
        pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.today().normalize()
    )
    src = source_for_path(path)
    if src is None:
        return {"status": "no_source", "message": ""}
    last = local_last_date(src)
    cutoff = today - pd.Timedelta(days=max_age_days)
    if last is not None and last >= cutoff:
        return {"status": "fresh", "last_date": last.date().isoformat(), "message": ""}
    try:
        res = update_source(src.name, session=session)
        return {
            "status": "refreshed",
            "last_date": res["last_date"],
            "message": (
                f"🔄 Auto-updated {src.name}: {res['rows']} rows "
                f"through {res['last_date']}."
            ),
        }
    except Exception as e:  # noqa: BLE001 — upstream down ⇒ early-warning, not a crash
        loc = f"last {last.date()}" if last is not None else "no local copy"
        return {
            "status": "failed",
            "last_date": last.date().isoformat() if last is not None else None,
            "message": (
                f"⚠️  Could not auto-update {src.name} ({type(e).__name__}); "
                f"proceeding with existing data ({loc})."
            ),
        }
