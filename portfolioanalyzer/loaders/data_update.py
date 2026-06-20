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
(``rate``/``price``). :func:`ensure_reference_data_fresh` is the on-run path
``main.py`` calls: it refreshes any reference source that has fallen behind its
publication cadence (business day for NIFTY, month for FRED) and reports a
per-source status the freshness gate blocks on. :func:`update_all` is the
unconditional refresh behind the optional ``portfolio-analyzer-update`` console
script (see :mod:`data_update_cli`); both stamp ``data/.last_fetched.json`` and
isolate failures so one dead feed can't abort the rest. No cron — the analyzer
refreshes on demand.

Sources
-------
- ``risk_free_fred`` → FRED ``INDIRLTLT01STM`` (India 10Y government bond
  rate, monthly). Stable, no-auth, machine-readable CSV — fully verifiable.
- ``benchmark_nifty_tri`` → NIFTY 50 TRI from niftyindices.com. Fetched via a
  stealth Chromium browser (optional ``browser`` extra), since niftyindices
  aggressively blocks non-browser clients and holds blocks against the source
  IP. The parser is unit-tested; **the live browser fetch must be verified in
  an environment that can reach the host** before relying on it.
- ``gold_lbma`` → LBMA Gold Price PM fix (USD per troy ounce, daily) from the
  LBMA's own price feed (``prices.lbma.org.uk``). Clean public JSON, no auth /
  key / browser — like FRED. This replaced the dead World Gold Council "gold
  price averages since 1978" dataset, which was discontinued in March 2025 when
  ICE Benchmark Administration pulled the historical LBMA Gold Price from
  third-party redistributors. LBMA still publishes the canonical London
  benchmark directly here; USD is used unconverted (the tool reports only
  normalized returns, and a time-varying USD/INR path would inject rupee/RBI
  dynamics into gold's measured return).
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import io
import json
import os
import random
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import requests

from portfolioanalyzer.loaders.benchmark import load_timeseries_csv

DATA_DIR = "data"
STAMP_FILE = os.path.join(DATA_DIR, ".last_fetched.json")

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# The LBMA Gold Price (the canonical London auction benchmark) is published
# directly on the LBMA's own site as a plain JSON array. We use the afternoon
# (PM) fix — the conventional end-of-day reference for gold valuation — in USD
# per troy ounce. No auth / key / browser needed (clean public JSON, like FRED).
LBMA_GOLD_PM_URL = "https://prices.lbma.org.uk/json/gold_pm.json"

# niftyindices serves its Total-Returns-Index history from a JSON POST API
# behind an ASP.NET session + ARR-affinity cookie wall, and aggressively
# blocks non-browser clients (a naive GET returns an anti-scrape HTML page).
# Because a block is held against the source IP, we never hit it with raw
# ``requests``; the sole path is a stealth browser that navigates the
# historical-data page (minting cookies / clearing any JS challenge) and then
# POSTs the date range to getTotalReturnIndexString from inside the page. The
# response is a ``{"d": "<json-array-string>"}`` envelope.
NIFTY_HIST_PAGE = "https://www.niftyindices.com/reports/historical-data"
NIFTY_TRI_ENDPOINT = "https://www.niftyindices.com/Backpage.aspx/getTotalReturnIndexString"
# Real browser User-Agent presented by the stealth Chromium context.
_NIFTY_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


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


def parse_lbma_gold_json(text: str) -> pd.DataFrame:
    """Parse the LBMA Gold Price JSON into a date-indexed USD/oz ``value`` frame.

    The feed is a JSON array of records ``{"d": "YYYY-MM-DD", "v": [USD, GBP,
    EUR]}``. We take the USD column (``v[0]``) — the LBMA Gold Price in **USD per
    troy ounce**. Records with a missing/None USD value (a handful of the oldest
    rows carry ``None`` for some currencies) are dropped.

    Raises on a non-JSON body — the LBMA endpoint, like any web feed, can return
    an HTML error / rate-limit page instead of data; detecting that here makes a
    bad response a *failed fetch* (→ stale → block) rather than a silently
    truncated series.
    """
    try:
        records = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LBMA gold response was not JSON (got {text[:80]!r})"
        ) from e
    if not isinstance(records, list) or not records:
        raise ValueError("LBMA gold response contained zero records")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [r.get("d") for r in records], errors="coerce"
            ),
            "value": pd.to_numeric(
                [
                    (r.get("v")[0] if isinstance(r.get("v"), list) and r.get("v") else None)
                    for r in records
                ],
                errors="coerce",
            ),
        }
    ).dropna()
    out = out.set_index("date").sort_index()
    if out.empty:
        raise ValueError("LBMA gold JSON parsed to zero rows")
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


def fetch_lbma_gold(*, session=None) -> pd.DataFrame:
    """Fetch the LBMA Gold Price (PM fix, USD/oz) as a date-indexed ``value``
    frame. Clean public JSON over plain ``requests`` (default UA works); raises
    on HTTP error or a non-JSON body (see :func:`parse_lbma_gold_json`)."""
    return parse_lbma_gold_json(_get(LBMA_GOLD_PM_URL, session=session))


def _niftyindices_body(index_name: str, start: str, end: str) -> str:
    """Build the JSON POST body for the niftyindices TRI endpoint.

    ``cinfo`` is a (single-quoted) JSON string embedded inside the JSON body;
    built by concatenation to avoid brace-escaping noise.
    """
    cinfo = (
        "{'name':'" + index_name + "','startDate':'" + start
        + "','endDate':'" + end + "','indexName':'" + index_name + "'}"
    )
    return json.dumps({"cinfo": cinfo})


def _playwright_available() -> bool:
    """True if the optional ``browser`` extra (playwright + stealth) is installed."""
    return (
        importlib.util.find_spec("playwright") is not None
        and importlib.util.find_spec("playwright_stealth") is not None
    )


def _fetch_niftyindices_browser(
    index_name: str, start: str, end: str, *, timeout: int = 30, headless: bool = True
) -> pd.DataFrame:  # pragma: no cover - requires a real browser + network
    """Drive a stealth Chromium to fetch the TRI, mirroring the proven
    ``mysore-spa-intelligence-engine`` scraper.

    Presents an authentic fingerprint (stealth hides ``navigator.webdriver``;
    real UA, ``en-IN`` locale, ``Asia/Kolkata`` timezone, 1920×1080 viewport)
    and makes a *single* human-paced hit: navigate the historical-data page to
    mint session cookies / clear any JS challenge, pause briefly, then issue
    the TRI POST from inside the page so the same-origin request carries the
    cookies automatically.
    """
    from playwright.sync_api import sync_playwright
    from playwright_stealth import Stealth

    body = _niftyindices_body(index_name, start, end)
    with Stealth().use_sync(sync_playwright()) as p:
        browser = p.chromium.launch(headless=headless)
        try:
            context = browser.new_context(
                user_agent=_NIFTY_UA,
                viewport={"width": 1920, "height": 1080},
                locale="en-IN",
                timezone_id="Asia/Kolkata",
            )
            page = context.new_page()
            page.goto(
                NIFTY_HIST_PAGE, wait_until="domcontentloaded", timeout=timeout * 1000
            )
            page.wait_for_timeout(random.randint(1500, 3500))  # human-ish pause
            text = page.evaluate(
                """async ([url, body]) => {
                    const r = await fetch(url, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json; charset=UTF-8',
                            'X-Requested-With': 'XMLHttpRequest',
                            'Accept': 'application/json, text/javascript, */*; q=0.01',
                        },
                        body,
                    });
                    if (!r.ok) throw new Error('TRI POST HTTP ' + r.status);
                    return await r.text();
                }""",
                [NIFTY_TRI_ENDPOINT, body],
            )
        finally:
            browser.close()
    return parse_niftyindices_tri_json(text)


def fetch_niftyindices_tri(
    *,
    index_name: str = "NIFTY 50",
    start: str = "01-Jan-2007",
    end: str | None = None,
    timeout: int = 30,
    headless: bool = True,
) -> pd.DataFrame:
    """Fetch the NIFTY 50 Total-Returns-Index history from niftyindices.com.

    niftyindices is aggressively anti-scrape and holds blocks against the
    source IP, so the **only** fetch path is a real stealth browser (see
    :func:`_fetch_niftyindices_browser`) — no raw ``requests`` path, which
    would risk flagging the IP. A single browser hit returns the full date
    range (no chunking, no retries: bursts are what trip the block, so the
    real "retry" is the next scheduled run via ``portfolio-analyzer-update``).

    Requires the optional ``browser`` extra::

        pip install '.[browser]' && playwright install chromium

    Args:
        index_name: niftyindices index name (default "NIFTY 50").
        start: inclusive start, ``DD-Mon-YYYY`` (default covers full history).
        end: inclusive end, ``DD-Mon-YYYY``; defaults to today.
        timeout: per-navigation timeout, seconds.
        headless: run Chromium headless (set False to watch / debug).

    Raises:
        RuntimeError: if the optional ``browser`` extra is not installed.
    """
    if not _playwright_available():
        raise RuntimeError(
            "niftyindices TRI fetch requires the optional 'browser' extra — install "
            "with: pip install '.[browser]' && playwright install chromium"
        )
    end = end or dt.date.today().strftime("%d-%b-%Y")
    return _fetch_niftyindices_browser(
        index_name, start, end, timeout=timeout, headless=headless
    )


# --- sources + writer ------------------------------------------------------

@dataclass
class DataSource:
    """One refreshable reference feed → one local CSV the pipeline reads."""

    name: str
    target_path: str
    fetch: Callable[..., pd.DataFrame]
    out_date_col: str  # must contain "date" for load_timeseries_csv
    out_value_col: str  # one of: rate / price / close / yield
    out_date_format: str  # strftime; must match the config's date_format
    # Publication cadence the freshness gate measures "behind" against:
    # "business_day" (the feed updates every trading day) or "month".
    cadence: str = "business_day"
    # True for hosts that punish repeated hits (niftyindices) — every attempt
    # is stamped and a second attempt the same day is suppressed.
    day_gated: bool = False
    # Human-readable metrics this source bears on, named in the stale warning.
    affects: str = ""
    # Short label for provenance / messages.
    label: str = ""


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
        target_path=os.path.join(DATA_DIR, "reference", "INDIRLTLT01STM.csv"),
        fetch=lambda session=None: fetch_fred_series("INDIRLTLT01STM", session=session),
        out_date_col="observation_date",
        out_value_col="rate",
        out_date_format="%Y-%m-%d",
        # FRED India 10Y government bond rate (monthly); risk-free proxy.
        cadence="month",
        day_gated=False,  # clean public CSV, no ban risk: refresh whenever behind
        affects="Sharpe, Sortino, alpha",
        label="FRED India 10Y (risk-free)",
    ),
    "benchmark_nifty_tri": DataSource(
        name="benchmark_nifty_tri",
        target_path=os.path.join(DATA_DIR, "reference", "NIFTY Total Returns Historical Data.csv"),
        fetch=lambda session=None: fetch_niftyindices_tri(),
        out_date_col="Date",
        out_value_col="Price",
        out_date_format="%m/%d/%Y",
        # NIFTY 50 TRI from niftyindices.com via stealth browser
        # (needs 'browser' extra; live fetch unverified in CI).
        cadence="business_day",
        day_gated=True,  # anti-scrape host: at most one attempt per day
        affects="alpha, beta",
        label="NIFTY 50 TRI (benchmark)",
    ),
    "gold_lbma": DataSource(
        name="gold_lbma",
        target_path=os.path.join(DATA_DIR, "reference", "gold_lbma_usd_daily.csv"),
        fetch=lambda session=None: fetch_lbma_gold(session=session),
        out_date_col="Date",
        # "Close" so the gold loader's price-column detection (which matches
        # "price"/"close") picks it up unchanged.
        out_value_col="Close",
        out_date_format="%Y-%m-%d",
        # LBMA Gold Price PM fix, USD/troy-ounce, published every London
        # business day. Clean public JSON ⇒ no ban risk, refresh whenever behind.
        cadence="business_day",
        day_gated=False,
        affects="gold and SGB valuation",
        label="LBMA Gold Price PM (USD/oz)",
    ),
}


# --- stamps ----------------------------------------------------------------

def _read_stamps() -> dict:
    try:
        with open(STAMP_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def read_stamp(name: str) -> dict:
    """Provenance for one source: ``last_date`` / ``fetched_at`` / ``rows`` /
    ``attempted_at`` (whatever has been recorded), or ``{}`` if never stamped."""
    return _read_stamps().get(name, {})


def _now_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()


def _update_stamp(name: str, **fields) -> None:
    """Merge ``fields`` into the stamp entry for ``name`` (preserving the rest)."""
    stamps = _read_stamps()
    entry = stamps.get(name, {})
    entry.update(fields)
    stamps[name] = entry
    parent = os.path.dirname(STAMP_FILE)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(STAMP_FILE, "w", encoding="utf-8") as f:
        json.dump(stamps, f, indent=2, sort_keys=True)


def _is_today(iso: str | None, today: pd.Timestamp) -> bool:
    """True if ISO timestamp ``iso`` falls on ``today`` (date comparison)."""
    if not iso:
        return False
    try:
        return pd.Timestamp(iso).date() == today.date()
    except (ValueError, TypeError):
        return False


# --- update orchestration --------------------------------------------------

def update_source(name: str, *, session=None) -> dict:
    """Fetch one source, write its CSV, and stamp it. Raises on fetch error.

    The stamp records both ``attempted_at`` and ``fetched_at`` (a successful
    update is also an attempt) plus ``last_date`` / ``rows``.
    """
    src = REGISTRY[name]
    df = src.fetch(session=session)
    write_normalized_csv(src, df)
    last_date = df.index.max().date().isoformat()
    now = _now_iso()
    _update_stamp(name, attempted_at=now, fetched_at=now, last_date=last_date, rows=len(df))
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


# --- cadence-driven freshness (the on-run "refresh if behind" path) --------

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


def cadence_frontier(cadence: str, today) -> pd.Timestamp:
    """The most recent date the feed *should* have published by ``today``.

    Local data on or after this date is current by the feed's own cadence;
    older than it is "behind". No magic-number tolerance — the frontier is
    the cadence boundary itself.

    - ``business_day``: the most recent trading day on or before ``today``
      (rolls a weekend back to Friday).
    - ``month``: the first of ``today``'s month (monthly feeds stamp the 1st).
    """
    today = pd.Timestamp(today).normalize()
    if cadence == "business_day":
        return pd.offsets.BDay().rollback(today).normalize()
    if cadence == "month":
        return today.replace(day=1)
    raise ValueError(f"unknown cadence: {cadence!r}")


def manual_staleness_warning(
    label: str,
    last_date,
    cadence: str,
    affects: str,
    refresh_hint: str,
    *,
    today=None,
) -> str | None:
    """Warn (never block) when a hand-maintained, feedless data file is behind.

    A generic utility for sources with no upstream feed to auto-refresh (see
    ARCHITECTURE.md → *Data freshness as a correctness invariant*: feedless
    sources can only warn). Gold is no longer one of these — it auto-refreshes
    from the LBMA feed and blocks like the other reference sources. This
    surfaces a file that has fallen behind its own publication cadence so
    the user knows to refresh it — rather than letting carried-forward values
    silently shorten or flatten the affected metrics. Returns one warning line,
    or ``None`` when the source is current.

    Args:
        label: human name of the source (and its file).
        last_date: latest dated row in the file (``None`` if none parsed).
        cadence: ``"month"`` or ``"business_day"`` — how often it should update.
        affects: metrics that degrade when it's stale (named in the warning).
        refresh_hint: how to refresh (a file path / doc pointer).
        today: override for testing.
    """
    if last_date is None:
        return f"⚠️  {label}: no dated rows found — {affects} may be wrong. {refresh_hint}"
    frontier = cadence_frontier(cadence, today or pd.Timestamp.today().normalize())
    if pd.Timestamp(last_date).normalize() < frontier:
        return (
            f"⚠️  {label} is stale (latest {pd.Timestamp(last_date).date()}, expected "
            f"≥ {frontier.date()}); {affects} use carried-forward values. {refresh_hint}"
        )
    return None


def assess_freshness(source: DataSource, *, today=None) -> dict:
    """Read local data + stamp and report where ``source`` stands today.

    ``current`` is the certification the block gate cares about: a source is
    current if its local data meets the cadence frontier *or* a successful
    fetch already happened today (the latter covers exchange holidays / feed
    lag, where a fetch returns the latest the source offers even though the
    calendar frontier is technically ahead of it).

    The ``fetched_today`` shortcut is honoured only when the file on disk
    actually reflects that fetch — i.e. its real last date reaches the stamp's
    recorded ``last_date``. A reorg/restore can revert the CSV to an older copy
    while the stamp keeps its success record; trusting the stamp alone then
    certifies a stale file as current. ``stamp_honoured`` is False in that
    desync, so the source falls back to its on-disk cadence assessment and is
    re-fetched.
    """
    today = (
        pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.today().normalize()
    )
    last = local_last_date(source)
    frontier = cadence_frontier(source.cadence, today)
    behind = last is None or last < frontier
    stamp = read_stamp(source.name)
    fetched_today = _is_today(stamp.get("fetched_at"), today)
    attempted_today = _is_today(stamp.get("attempted_at"), today)

    # File-vs-stamp consistency: the stamp's success claim is only credible if
    # the file reaches the date the stamp says was written.
    stamp_last = stamp.get("last_date")
    stamp_last_ts = pd.Timestamp(stamp_last).normalize() if stamp_last else None
    stamp_honoured = stamp_last_ts is None or (last is not None and last >= stamp_last_ts)
    return {
        "last_date": last,
        "frontier": frontier,
        "behind": behind,
        "fetched_today": fetched_today,
        "attempted_today": attempted_today,
        "stamp_honoured": stamp_honoured,
        "current": (not behind) or (fetched_today and stamp_honoured),
    }


def ensure_source_current(name: str, *, session=None, today=None) -> dict:
    """Refresh ``name`` if it is behind its cadence, honouring the day-gate.

    Returns a status dict with a user-facing ``message`` (empty when nothing
    to say), plus ``affects`` (the metrics this source bears on) for the
    caller's stale warning. Statuses:

    - ``current``: already on cadence, or already fetched today — no fetch.
    - ``refreshed``: was behind, fetched fresh data this run.
    - ``stale``: behind and could not be certified current — either the fetch
      failed, or a day-gated source already attempted (and failed) today so a
      retry is suppressed. The block gate stops the run on this unless
      ``--allow-stale``.
    """
    today = (
        pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.today().normalize()
    )
    src = REGISTRY[name]
    a = assess_freshness(src, today=today)
    last_iso = a["last_date"].date().isoformat() if a["last_date"] is not None else None

    if a["current"]:
        return {"name": name, "status": "current", "last_date": last_iso, "message": "",
                "affects": src.affects}

    # Behind and not yet certified current today. Day-gated hosts get at most
    # one attempt per day: if today's attempt already happened (and left us
    # not-current, i.e. it failed), don't poke the host again.
    if src.day_gated and a["attempted_today"]:
        return {
            "name": name, "status": "stale", "last_date": last_iso,
            "message": (
                f"⚠️  {src.label} is behind and already attempted today; not "
                f"retrying ({src.name} is contacted at most once per day)."
            ),
            "affects": src.affects,
        }

    # Record the attempt *before* fetching so a day-gated host can't be hit
    # twice in a day even if the fetch raises.
    _update_stamp(name, attempted_at=_now_iso())
    try:
        res = update_source(name, session=session)
        return {
            "name": name, "status": "refreshed", "last_date": res["last_date"],
            "message": (
                f"🔄 Refreshed {src.label}: {res['rows']} rows through {res['last_date']}."
            ),
            "affects": src.affects,
        }
    except Exception as e:  # noqa: BLE001 — upstream down ⇒ stale, gate decides the rest
        loc = f"last {last_iso}" if last_iso else "no local copy"
        return {
            "name": name, "status": "stale", "last_date": last_iso,
            "message": (
                f"⚠️  Could not refresh {src.label} ({type(e).__name__}); "
                f"existing data is stale ({loc})."
            ),
            "affects": src.affects,
        }


def reference_provenance(paths) -> list[dict]:
    """Per-source provenance for the reference feeds behind ``paths``, read from
    the stamp file: ``name`` / ``label`` / ``last_date`` / ``fetched_at`` /
    ``attempted_at`` (missing values are ``None``). Unregistered paths are
    skipped. This is what rides into the run's stderr log and the PNG metadata.
    """
    out = []
    for path in paths:
        src = source_for_path(path)
        if src is None:
            continue
        st = read_stamp(src.name)
        out.append(
            {
                "name": src.name,
                "label": src.label,
                "last_date": st.get("last_date"),
                "fetched_at": st.get("fetched_at"),
                "attempted_at": st.get("attempted_at"),
            }
        )
    return out


def ensure_reference_data_fresh(paths, *, session=None, today=None) -> list[dict]:
    """Run :func:`ensure_source_current` for every ``path`` that maps to a
    registered reference source. Unregistered paths (e.g. a legacy manual CSV)
    are silently skipped — they have no upstream to certify against."""
    results = []
    for path in paths:
        src = source_for_path(path)
        if src is None:
            continue
        results.append(ensure_source_current(src.name, session=session, today=today))
    return results
