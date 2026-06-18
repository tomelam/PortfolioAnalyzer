"""Collect Value Research Online (VRO) published trailing returns.

Used to validate our own metrics against an external authority (see the
VRO-parity wire test). VRO sits behind a Cloudflare JS challenge, so — exactly
like the niftyindices scraper in ``loaders.data_update`` — the only fetch path
is a stealth Chromium browser (the optional ``browser`` extra). Once the
challenge clears, VRO's React app talks to same-origin JSON APIs; we issue the
``peer-comparison-returns`` request from *inside* the page so the session
cookies ride along automatically.

That endpoint returns the fund alongside its peers for one period; the fund's
own **annualised** trailing return (percent) is the ``returns`` entry whose
``plan_id`` matches the requested ``fund_id``:

    GET /api/funds/peer-comparison-returns/?fund_id=15841&period=5Y
    -> {"returns": [{"plan_id": "15841", "returns": "14.08"}, ...peers], ...}

The parse step is a pure function (offline-unit-tested against a fixture); only
the fetch needs a browser + network.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import DateOffset

import metrics

# Real desktop-Chrome User-Agent presented by the stealth context (mirrors the
# niftyindices fetcher's authentic-fingerprint approach).
VRO_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Periods VRO exposes that are directly comparable to an annualised CAGR.
VRO_PERIODS: tuple[str, ...] = ("1Y", "3Y", "5Y")

_VRO_FUNDS_CSV = Path(__file__).resolve().parent.parent / "data" / "vro_funds.csv"


@dataclass(frozen=True)
class VROFund:
    """One row of the mfapi ↔ VRO fund mapping (``data/vro_funds.csv``)."""

    mfapi_code: str
    vro_plan_id: str
    vro_slug: str
    isin: str
    name: str

    @property
    def mfapi_url(self) -> str:
        return f"https://api.mfapi.in/mf/{self.mfapi_code}"


def load_vro_fund_map(path: Path | None = None) -> list[VROFund]:
    """Load the mfapi ↔ VRO fund mapping from ``data/vro_funds.csv``."""
    path = path or _VRO_FUNDS_CSV
    with path.open(newline="") as f:
        return [
            VROFund(
                mfapi_code=row["mfapi_code"].strip(),
                vro_plan_id=row["vro_plan_id"].strip(),
                vro_slug=row["vro_slug"].strip(),
                isin=row["isin"].strip(),
                name=row["name"].strip(),
            )
            for row in csv.DictReader(f)
        ]


def vro_page_url(plan_id: str, slug: str) -> str:
    return f"https://www.valueresearchonline.com/funds/{plan_id}/{slug}/"


def vro_returns_api(plan_id: str, period: str) -> str:
    return (
        "https://www.valueresearchonline.com/api/funds/peer-comparison-returns/"
        f"?fund_id={plan_id}&period={period}"
    )


def parse_peer_comparison_returns(json_text: str, plan_id: str) -> float:
    """Extract a fund's own annualised trailing return (percent) from a
    ``peer-comparison-returns`` payload.

    Args:
        json_text: Raw JSON body from the endpoint.
        plan_id: VRO plan id of the fund of interest.

    Returns:
        The annualised return as a percent (e.g. ``14.08``).

    Raises:
        ValueError: If the payload is malformed or the plan id is absent.
    """
    try:
        payload = json.loads(json_text)
        rows = payload["returns"]
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"malformed peer-comparison-returns payload: {e}") from e

    for row in rows:
        if str(row.get("plan_id")) == str(plan_id):
            return float(row["returns"])
    raise ValueError(f"plan_id {plan_id} not found in peer-comparison returns")


def trailing_cagr_pct(
    nav: pd.Series, years: int, as_of: pd.Timestamp | None = None
) -> float:
    """Our point-to-point trailing CAGR (percent), the methodology that matches
    VRO's published trailing returns.

    Reconciliation against VRO (2026-06-18) showed VRO uses point-to-point
    *daily* NAVs — latest NAV to exactly ``years`` prior — not month-end
    sampling (month-end ran ~0.5–0.8pp low). So this trims the NAV series to
    ``[anchor - years, anchor]`` and takes the plain endpoint CAGR; it agrees
    with VRO to ≤0.25pp on the funds tested.

    Args:
        nav: Daily NAV series indexed by date.
        years: Trailing window length in years.
        as_of: Anchor date; defaults to the latest NAV date.
    """
    anchor = as_of if as_of is not None else nav.index.max()
    start = anchor - DateOffset(years=years)
    window = nav[(nav.index >= start) & (nav.index <= anchor)]
    if len(window) < 2:
        raise ValueError(f"too few NAVs in the trailing {years}Y window to compute CAGR")
    return metrics.cagr(window) * 100.0


def browser_extra_available() -> bool:
    """True if the optional ``browser`` extra (playwright + stealth) is installed."""
    return (
        importlib.util.find_spec("playwright") is not None
        and importlib.util.find_spec("playwright_stealth") is not None
    )


def fetch_vro_trailing_returns(
    plan_id: str,
    slug: str,
    periods: tuple[str, ...] = VRO_PERIODS,
    *,
    timeout: int = 45,
    headless: bool = True,
) -> dict[str, float]:  # pragma: no cover - requires a real browser + network
    """Drive a stealth Chromium to collect VRO's annualised trailing returns.

    Navigates the fund page once to clear the Cloudflare challenge / mint
    session cookies, pauses briefly, then fetches ``peer-comparison-returns``
    for each period from inside the page (same-origin, cookies carried).

    Returns:
        ``{period: annualised_return_pct}`` for each requested period.
    """
    from playwright.sync_api import sync_playwright
    from playwright_stealth import Stealth

    out: dict[str, float] = {}
    with Stealth().use_sync(sync_playwright()) as p:
        browser = p.chromium.launch(headless=headless)
        try:
            context = browser.new_context(
                user_agent=VRO_UA,
                viewport={"width": 1920, "height": 1080},
                locale="en-IN",
                timezone_id="Asia/Kolkata",
            )
            page = context.new_page()
            page.goto(
                vro_page_url(plan_id, slug),
                wait_until="domcontentloaded",
                timeout=timeout * 1000,
            )
            page.wait_for_timeout(random.randint(3000, 5000))  # let Cloudflare settle
            for period in periods:
                text = page.evaluate(
                    """async (url) => {
                        const r = await fetch(url, {
                            headers: {
                                'X-Requested-With': 'XMLHttpRequest',
                                'Accept': 'application/json, text/javascript, */*; q=0.01',
                            },
                        });
                        if (!r.ok) throw new Error('VRO returns HTTP ' + r.status);
                        return await r.text();
                    }""",
                    vro_returns_api(plan_id, period),
                )
                out[period] = parse_peer_comparison_returns(text, plan_id)
        finally:
            browser.close()
    return out
