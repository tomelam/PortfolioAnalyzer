"""Ingest SGB premature/maturity **redemption prices** from RBI press releases.

Background (Phase B2 of the gold/SGB thread). An SGB is an INR instrument whose
contractual redemption value is set by the RBI, not by LBMA gold: ~3 business
days before each redemption date RBI publishes a press release fixing the
redemption price (₹ per unit; 1 unit = 1 gram of 999 gold) as the simple average
of the IBJA closing gold price over the preceding three business days. This
module ingests those announcements into ``data/funds/sgb_redemptions.csv`` — the
data layer the (deferred) co-equal "redemption view" valuation will read.

Source path (established by ``scripts/probe_rbi_sgb_redemption.py`` — the B2
Step-0 fail-loud gate). The mobile directory ``BS_SwarnaBharat.aspx`` is
CAPTCHA-walled, so we route around it over plain ``requests``:

- **Enumerate** redemption press releases via RBI's open search endpoint
  (``SearchResults.aspx?search=...``), which lists each announcement linked by
  ``?prid=<N>``. The full history is reached by walking the ASP.NET pagination
  (14 hits/page, ``hdnPageNo`` postback) across two queries — *premature* (PRE)
  and *final* (MAT) redemption, which have distinct titles.
- **Parse** each ``BS_PressReleaseDisplay.aspx?prid=<N>`` press release for the
  redemption date(s), the ₹/unit price, and the tranche reference(s).

Decomposed like ``loaders/scss.py`` so the parsers are unit-testable offline:

- ``parse_search_prids`` — pure: one search page's HTML → list of redemption PRIDs
- ``enumerate_redemption_prids`` — network: walk all pages of both searches → PRIDs
- ``parse_redemption_pr`` — pure: one PR's HTML → list of redemption rows
- ``fetch_search_html`` / ``fetch_pr_html`` — network: URL → HTML text
- ``update_sgb_redemptions`` — composition: enumerate → fetch → parse → upsert
  the CSV (preserving existing rows), reusing ``data_update``'s stamp machinery.

This is **event-cadence** ingest (redemptions happen on irregular scheduled
dates), so it is *not* a ``data_update`` ``DataSource`` — that machinery is for
single-value daily/monthly time series with a calendar frontier, whereas this
is a growing multi-column table with no frontier to measure "behind" against.
It reuses ``data_update``'s stamp helpers for provenance only.
"""

from __future__ import annotations

import os
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from portfolioanalyzer.loaders import data_update
from portfolioanalyzer.utils import info

# Open RBI endpoints (no CAPTCHA, plain requests — see module docstring).
# Two searches are needed: "premature redemption" lists the pre-maturity (PRE)
# announcements; "final redemption" lists the at-maturity (MAT) ones — they have
# distinct titles, so a single query misses one kind. Both are unioned by PRID.
RBI_SEARCH_URL = (
    "https://rbi.org.in/Scripts/SearchResults.aspx"
    "?search=premature%20redemption%20sovereign%20gold%20bond"
)
RBI_FINAL_SEARCH_URL = (
    "https://rbi.org.in/Scripts/SearchResults.aspx"
    "?search=final%20redemption%20sovereign%20gold%20bond"
)
RBI_SEARCH_URLS = (RBI_SEARCH_URL, RBI_FINAL_SEARCH_URL)
RBI_PR_URL = "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid={prid}"

# RBI's search returns 14 results per page; older redemptions are reached by
# walking the ASP.NET postback pagination (see ``enumerate_redemption_prids``).
_SEARCH_PAGE_SIZE = 14

# Where the ingested table lives, and the stamp key (reusing data_update's stamp
# file for unified provenance) for this event-cadence source.
REDEMPTIONS_CSV = os.path.join(data_update.DATA_DIR, "funds", "sgb_redemptions.csv")
STAMP_NAME = "sgb_redemptions"

# CSV schema (the data model the deferred redemption-view valuation reads).
COLUMNS = [
    "tranche_id",
    "redemption_date",
    "kind",  # PRE (premature) | MAT (maturity)
    "inr_per_gram",  # ₹ per unit; 1 unit = 1 gram of 999 gold
    "source_prid_or_url",
    "tier",  # provenance tier: T1 = fetched direct from the RBI press release
]

_ROMAN = r"[IVXLC]+"

_MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)"
)
# One or more redemption dates as RBI phrases them: a single "Month DD, YYYY",
# or a co-incident pair "Month D and D, YYYY" / "Month D and Month D, YYYY"
# (two tranches redeemed in the same week at one price). The trailing year is
# shared. ``_parse_due_dates`` turns the captured string into date objects.
_DATE = (
    rf"{_MONTHS}\s+\d{{1,2}}"
    rf"(?:\s*(?:and|&)\s*(?:{_MONTHS}\s+)?\d{{1,2}})*"
    rf",?\s*\d{{4}}"
)
# The robust, era-spanning anchor: "...(falling) due on <date(s)> ... shall be
# ₹|Rs.|INR <amount>/-". ``[^.]*?`` keeps the date and price within one sentence
# so the header "due on <date>" (which has no price) is skipped in favour of the
# operative "...shall be ₹<amount>" sentence. Handles both the modern "₹14,774"
# and the older "Rs. 5119" / "₹4804" forms.
_DUE_PRICE_RE = re.compile(
    rf"(?:falling\s+)?due\s+on\s+({_DATE})[^.]*?"
    rf"shall\s+be\s*(?:₹|Rs\.?|INR)\s?([\d,]+)"
)


# --- pure parsers ----------------------------------------------------------

def parse_search_prids(html: str) -> list[str]:
    """Pure: RBI search-results HTML → ordered, de-duplicated list of PRIDs
    whose anchor text is an SGB redemption announcement.

    The search page links each press release as ``...?prid=<N>``; we keep only
    anchors whose text names a Sovereign Gold Bond redemption (the search is
    already scoped to that, but we filter defensively so an unrelated result
    can't slip a bogus PRID into the ingest).
    """
    soup = BeautifulSoup(html, "html.parser")
    prids: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        m = re.search(r"prid=(\d+)", a["href"], re.I)
        if not m:
            continue
        prid = m.group(1)
        text = a.get_text(" ", strip=True).lower()
        if "gold bond" not in text and "sgb" not in text:
            continue
        if "redemption" not in text:
            continue
        if prid not in seen:
            seen.add(prid)
            prids.append(prid)
    return prids


def _normalize_fiscal_year(year_token: str) -> str:
    """Normalize an RBI 'SGB year' token to the ``data/funds/sgb_tranches.csv``
    fiscal-year form. ``'2020-21'`` passes through; a bare ``'2015'`` (used for
    the inaugural tranches) becomes ``'2015-16'``."""
    year_token = year_token.strip()
    if "-" in year_token:
        return year_token
    y = int(year_token)
    return f"{y}-{(y + 1) % 100:02d}"


def _extract_tranche_ids(text: str) -> list[str]:
    """Pull every ``<fiscal_year>-<series>`` tranche id referenced in a PR body.

    Handles both RBI phrasings seen in the wild:
      - ``"Series I of SGB 2015"``         (older releases)
      - ``"SGB 2020-21 Series-III"``       (recent releases)
    A single release can name several tranches (when multiple tranches share a
    redemption date); all share the one gold-based redemption price.
    """
    pairs: list[tuple[str, str]] = []
    # Pattern A: "SGB 2020-21 Series-III" / "SGB 2015, Series I"
    for yr, ser in re.findall(rf"SGB\s*(20\d\d(?:-\d\d)?)\s*,?\s*Series[\s-]+({_ROMAN})", text):
        pairs.append((yr, ser))
    # Pattern B: "Series I of SGB 2015"
    for ser, yr in re.findall(rf"Series[\s-]+({_ROMAN})\s+of\s+SGB\s*(20\d\d(?:-\d\d)?)", text):
        pairs.append((yr, ser))
    # Pattern C: "SGB 2016 (I)" / "SGB 2016-I" / "SGB 2016 - I" — the older
    # series-less form, where the roman is bound to the year by a parenthesis or
    # a dash (NOT a bare space, which would risk swallowing stray romans like the
    # "I" in "In terms of…").
    for yr, ser in re.findall(rf"SGB\s*(20\d\d(?:-\d\d)?)\s*(?:\(\s*|-\s*|\s-\s*)({_ROMAN})\b", text):
        pairs.append((yr, ser))
    out: list[str] = []
    seen: set[str] = set()
    for yr, ser in pairs:
        tid = f"{_normalize_fiscal_year(yr)}-{ser}"
        if tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out


def _parse_due_dates(date_str: str) -> list:
    """Turn a captured ``_DATE`` string into a list of ``datetime.date``.

    Handles ``"May 30, 2022"``, ``"August 5 and 8, 2021"`` (shared month+year),
    and ``"November 04 and November 06, 2023"`` (explicit second month). The
    4-digit year is shared across all days; a day without its own month inherits
    the previous one. Returns ``[]`` if nothing parses.
    """
    year_m = re.search(r"\d{4}", date_str)
    if not year_m:
        return []
    year = year_m.group(0)
    body = re.sub(r"\d{4}", "", date_str)  # strip the year; days are 1–2 digits
    out = []
    last_month = None
    for mon, day in re.findall(rf"(?:({_MONTHS})\s+)?(\d{{1,2}})", body):
        mon = mon or last_month
        if not mon:
            continue
        last_month = mon
        d = pd.to_datetime(f"{mon} {day} {year}", errors="coerce")
        if pd.notna(d):
            out.append(d.date())
    return out


def parse_redemption_pr(html: str) -> list[dict]:
    """Pure: one redemption press release's HTML → list of redemption rows.

    Returns one dict per tranche named in the release, each with
    ``tranche_id``, ``redemption_date`` (ISO ``YYYY-MM-DD``), ``kind``
    (``PRE``/``MAT``), and ``inr_per_gram`` (int). Provenance columns
    (``source_prid_or_url``, ``tier``) are attached by the caller, which knows
    the PRID. Returns ``[]`` if the page isn't a parseable redemption *price*
    notice — a CAPTCHA interstitial, an unexpected layout, or a "scheme
    calendar" announcement that carries no price — so the caller treats an empty
    parse as "nothing to ingest from this PRID", never a crash.

    The single ``_DUE_PRICE_RE`` anchor captures the redemption date(s) and the
    ₹/unit price together (1 unit = 1 gram of 999 gold). ``kind`` is ``PRE`` when
    the body mentions *premature* redemption, else ``MAT`` (the final at-maturity
    redemption). When one release covers several tranches:

    - **one date, N tranches** — all share that date and price (co-redeemed);
    - **N dates, N tranches** — paired in listed order (two tranches whose
      windows fall in the same week at one price);
    - **mismatched counts** — ambiguous; the release is skipped with a logged
      note rather than guessing a date↔tranche mapping (no silent bad rows).
    """
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))

    m = _DUE_PRICE_RE.search(text)
    if not m:
        return []
    dates = _parse_due_dates(m.group(1))
    tranche_ids = _extract_tranche_ids(text)
    if not dates or not tranche_ids:
        return []

    inr_per_gram = int(m.group(2).replace(",", ""))
    kind = "PRE" if re.search(r"premature", text, re.I) else "MAT"

    if len(dates) == 1:
        pairs = [(tid, dates[0]) for tid in tranche_ids]
    elif len(dates) == len(tranche_ids):
        pairs = list(zip(tranche_ids, dates))
    else:
        info(
            f"sgb_redemptions: skipping ambiguous multi-date release "
            f"(dates={m.group(1)!r}, tranches={tranche_ids}); add manually if needed"
        )
        return []

    return [
        {
            "tranche_id": tid,
            "redemption_date": d.isoformat(),
            "kind": kind,
            "inr_per_gram": inr_per_gram,
        }
        for tid, d in pairs
    ]


# --- network fetchers ------------------------------------------------------

def fetch_search_html(url: str = RBI_SEARCH_URL, *, session=None) -> str:
    """Fetch the RBI redemption search-results page (plain requests)."""
    resp = (session or requests).get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def fetch_pr_html(prid: str, *, session=None) -> str:
    """Fetch one RBI press release by PRID (plain requests)."""
    resp = (session or requests).get(RBI_PR_URL.format(prid=prid), timeout=30)
    resp.raise_for_status()
    return resp.text


def _form_fields(html: str) -> dict:
    """Extract the ASP.NET form's hidden/visible ``<input>`` name→value pairs
    (``__VIEWSTATE``, ``__EVENTVALIDATION``, ``hdnPageNo``, the search box, …).
    These must be echoed back on each pagination postback."""
    soup = BeautifulSoup(html, "html.parser")
    return {i.get("name"): i.get("value", "") for i in soup.find_all("input") if i.get("name")}


def _total_records(html: str) -> int:
    """The ``N Records`` count RBI prints above the result list (0 if absent)."""
    m = re.search(r"([\d,]+)\s+Records", re.sub(r"<[^>]+>", " ", html))
    return int(m.group(1).replace(",", "")) if m else 0


def enumerate_redemption_prids(*, session=None, search_urls=RBI_SEARCH_URLS, sleep: float = 0.25) -> list[str]:
    """Network: walk every page of each RBI redemption search and return the
    union of redemption PRIDs (newest first within each search).

    RBI's ``SearchResults.aspx`` shows 14 hits per page and pages via an ASP.NET
    ``__doPostBack('btnUpdate','')`` that reads ``hdnPageNo``. We replay that:
    GET page 1, then POST page 2…N echoing the form's ``__VIEWSTATE`` /
    ``__EVENTVALIDATION`` (regenerated each response, so they are re-read every
    hop) with ``hdnPageNo`` bumped. A persistent ``Session`` carries cookies.
    Both the premature- and final-redemption searches are walked so PRE and MAT
    announcements are covered (see ``RBI_SEARCH_URLS``).
    """
    sess = session or requests.Session()
    out: list[str] = []
    seen: set[str] = set()
    for url in search_urls:
        html = fetch_search_html(url, session=sess)
        pages = max(1, -(-_total_records(html) // _SEARCH_PAGE_SIZE))  # ceil-div
        fields = _form_fields(html)
        for prid in parse_search_prids(html):
            if prid not in seen:
                seen.add(prid)
                out.append(prid)
        for page in range(2, pages + 1):
            data = dict(fields)
            data["__EVENTTARGET"] = "btnUpdate"
            data["__EVENTARGUMENT"] = ""
            data["hdnPageNo"] = str(page)
            resp = sess.post(url, data=data, timeout=30)
            resp.raise_for_status()
            html = resp.text
            fields = _form_fields(html)  # refresh viewstate for the next hop
            for prid in parse_search_prids(html):
                if prid not in seen:
                    seen.add(prid)
                    out.append(prid)
            if sleep:
                time.sleep(sleep)
    return out


# --- CSV read / merge / write ----------------------------------------------

def read_redemptions(path: str = REDEMPTIONS_CSV) -> pd.DataFrame:
    """Read the redemptions table (empty, correctly-typed frame if absent)."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype=str).reindex(columns=COLUMNS)
    return df


def lookup_redemption(tranche_id: str, redemption_date: str | None = None,
                      *, path: str = REDEMPTIONS_CSV) -> dict:
    """Return one redemption row for ``tranche_id`` as a plain dict.

    Used by the valuation layer to price an early-redeemed SGB holding: the row's
    ``inr_per_gram`` is the contractual cash value (₹/unit) on ``redemption_date``.

    ``redemption_date`` (ISO ``YYYY-MM-DD``) selects a specific redemption when a
    tranche has more than one on record. When omitted, the lookup succeeds only if
    the tranche has **exactly one** redemption row; 0 or ≥2 candidates are an error
    (fail-loud, no silent pick) so a portfolio that says "redeemed" without a date
    can't quietly mark to the wrong window.

    Raises ``ValueError`` on no match or an ambiguous (date-less, multi-row) lookup.
    ``inr_per_gram`` is returned as ``int``; other columns pass through as strings.
    """
    df = read_redemptions(path)
    cand = df[df["tranche_id"] == tranche_id]
    if redemption_date is not None:
        cand = cand[cand["redemption_date"] == redemption_date]
    if cand.empty:
        when = f" on {redemption_date}" if redemption_date else ""
        raise ValueError(
            f"no SGB redemption price on record for tranche {tranche_id!r}{when} "
            f"in {os.path.basename(path)}; run "
            f"`python -m portfolioanalyzer.loaders.sgb_redemptions` to refresh, or "
            f"check the tranche_id / redemption_date"
        )
    if len(cand) > 1:
        dates = ", ".join(sorted(cand["redemption_date"]))
        raise ValueError(
            f"tranche {tranche_id!r} has {len(cand)} redemption rows ({dates}); "
            f"set redemption_date on the [[sgb]] entry to pick one"
        )
    row = cand.iloc[0].to_dict()
    row["inr_per_gram"] = int(row["inr_per_gram"])
    return row


def merge_redemptions(existing: pd.DataFrame, incoming: list[dict]) -> pd.DataFrame:
    """Upsert ``incoming`` rows into ``existing`` keyed on
    ``(tranche_id, redemption_date)``.

    Incoming rows (freshly fetched from RBI, tier T1) win on conflict; existing
    rows for keys not in the incoming batch are preserved (so a partial fetch
    never drops previously-seeded history). The result is sorted by
    redemption_date then tranche_id and de-duplicated on the key.
    """
    inc = pd.DataFrame(incoming, columns=COLUMNS) if incoming else pd.DataFrame(columns=COLUMNS)
    combined = pd.concat([existing, inc], ignore_index=True)
    combined = combined.dropna(subset=["tranche_id", "redemption_date"])
    # keep="last" → the appended incoming row supersedes an existing duplicate.
    combined = combined.drop_duplicates(subset=["tranche_id", "redemption_date"], keep="last")
    combined = combined.sort_values(["redemption_date", "tranche_id"]).reset_index(drop=True)
    return combined[COLUMNS]


def write_redemptions(df: pd.DataFrame, path: str = REDEMPTIONS_CSV) -> None:
    """Write the redemptions table to ``path`` (creating parent dirs)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(path, index=False)


# --- composition -----------------------------------------------------------

def update_sgb_redemptions(
    *,
    path: str = REDEMPTIONS_CSV,
    session=None,
    replay_from: str | None = None,
    limit: int | None = None,
) -> dict:
    """Enumerate RBI redemption press releases, parse each, and upsert the CSV.

    Network by default: ``enumerate_redemption_prids`` walks *every* page of both
    the premature- and final-redemption searches (full history, PRE + MAT), then
    each PR is fetched and parsed. The upsert preserves existing rows, so a re-run
    is the backfill mechanism — it adds newly-published redemptions and any older
    ones not seen before. With ``replay_from`` set, HTML is read from local files
    (``<dir>/rbi_sgb_redemption_search.html`` and
    ``<dir>/rbi_sgb_redemption_pr_<prid>.html``) instead of RBI — the offline
    path used by tests, which uses the single saved search page (no pagination).
    ``limit`` caps how many PRIDs are fetched (newest first, as the search lists
    them).

    Returns a summary dict: ``prids`` seen, ``rows`` ingested this run, and the
    resulting ``total`` row count. Stamps ``data/.last_fetched.json`` under
    ``sgb_redemptions`` for unified provenance with the other reference feeds.
    """
    def _read(name: str) -> str:
        with open(os.path.join(replay_from, name), encoding="utf-8") as f:
            return f.read()

    if replay_from:
        # Offline: a single saved search page (no live pagination/postback).
        prids = parse_search_prids(_read("rbi_sgb_redemption_search.html"))
    else:
        # Network: walk every page of both searches so older PRE redemptions and
        # the at-maturity (MAT) ones are reached, not just the recent first page.
        prids = enumerate_redemption_prids(session=session)
    if limit is not None:
        prids = prids[:limit]

    incoming: list[dict] = []
    for prid in prids:
        try:
            pr_html = _read(f"rbi_sgb_redemption_pr_{prid}.html") if replay_from else fetch_pr_html(prid, session=session)
        except FileNotFoundError:
            continue  # replay fixtures only cover a subset of PRIDs
        for row in parse_redemption_pr(pr_html):
            row["source_prid_or_url"] = RBI_PR_URL.format(prid=prid)
            row["tier"] = "T1"  # fetched direct from the RBI press release
            incoming.append(row)

    merged = merge_redemptions(read_redemptions(path), incoming)
    write_redemptions(merged, path)

    now = pd.Timestamp.utcnow().isoformat()
    last_date = merged["redemption_date"].max() if not merged.empty else None
    data_update._update_stamp(
        STAMP_NAME, attempted_at=now, fetched_at=now, last_date=last_date, rows=int(len(merged))
    )
    return {
        "prids": len(prids),
        "rows": len(incoming),
        "total": int(len(merged)),
        "last_date": last_date,
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI
    """``python -m portfolioanalyzer.loaders.sgb_redemptions`` — one-shot ingest."""
    res = update_sgb_redemptions()
    info(
        f"SGB redemptions: scanned {res['prids']} press release(s), ingested "
        f"{res['rows']} row(s); {res['total']} total through {res['last_date']}."
    )
    return 0 if res["total"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
