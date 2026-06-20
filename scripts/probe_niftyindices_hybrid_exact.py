#!/usr/bin/env python3
"""Thread-5 probe: do the 3 hybrid/debt benchmarks answer the niftyindices
historical endpoints when queried by their EXACT registered names?

Prior probes (`probe_niftyindices_benchmarks.py`, `probe_niftyindices_indexmaster.py`)
only test-fetched names they could *find* in the page's autocomplete/dropdowns —
and the hybrid/debt indices never appear there. They therefore never actually
POSTed the exact registered names to the HIST endpoint. This script closes that
gap: for each of the three benchmarks it POSTs both the TRI endpoint
(getTotalReturnIndexString) and the historical-price endpoint
(getHistoricaldatatabletoString) with the verbatim name, in one stealth session.

If even the exact name returns empty on both endpoints, the earlier
"endpoint-coverage gap" conclusion is confirmed and we must source these indices
elsewhere (NSE factsheets). If HIST answers, we have a live daily series and can
wire it in. One stealth session, human-paced. Throwaway diagnostics (kept +
documented per repo convention).
"""
from __future__ import annotations

import datetime as dt
import json
import random
import sys

from portfolioanalyzer.loaders.data_update import _NIFTY_UA, NIFTY_HIST_PAGE, NIFTY_TRI_ENDPOINT

HIST_ENDPOINT = "https://www.niftyindices.com/Backpage.aspx/getHistoricaldatatabletoString"

# Exact registered names as published on the index product pages / VRO benchmark
# column. These are the strings we POST verbatim.
NAMES = [
    "NIFTY 50 Hybrid Composite Debt 65:35 Index",   # HDFC Balanced Advantage
    "NIFTY 50 Hybrid Composite Debt 15:85 Index",   # HDFC Hybrid Debt
    "NIFTY Corporate Bond Index A-II",              # ICICI Corporate Bond
    # A couple of plausible spelling variants to rule out a naming mismatch:
    "NIFTY 50 Hybrid Composite Debt 65:35",
    "NIFTY Corporate Bond Index A- II",
    # Control: a known-good equity TRI index, proves the session/endpoint work.
    "NIFTY 100",
]

START = (dt.date.today() - dt.timedelta(days=200)).strftime("%d-%b-%Y")
END = dt.date.today().strftime("%d-%b-%Y")

FETCH_JS = """async ([url, body]) => {
    try {
        const r = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json; charset=UTF-8',
                'X-Requested-With': 'XMLHttpRequest',
                'Accept': 'application/json, text/javascript, */*; q=0.01',
            },
            body,
        });
        const text = await r.text();
        return { ok: r.ok, status: r.status, text };
    } catch (e) { return { ok: false, status: -1, text: String(e) }; }
}"""


def _body(name: str) -> str:
    cinfo = (
        "{'name':'" + name + "','startDate':'" + START
        + "','endDate':'" + END + "','indexName':'" + name + "'}"
    )
    return json.dumps({"cinfo": cinfo})


def _records(text: str):
    try:
        env = json.loads(text)
        inner = env.get("d") if isinstance(env, dict) else None
        if not inner:
            return []
        recs = json.loads(inner)
        return recs if isinstance(recs, list) else None
    except Exception:
        return None


def main() -> int:
    from playwright.sync_api import sync_playwright
    from playwright_stealth import Stealth

    rows = []
    with Stealth().use_sync(sync_playwright()) as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                user_agent=_NIFTY_UA,
                viewport={"width": 1920, "height": 1080},
                locale="en-IN",
                timezone_id="Asia/Kolkata",
            )
            page = context.new_page()
            page.goto(NIFTY_HIST_PAGE, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(random.randint(1500, 3500))

            for name in NAMES:
                for label, endpoint in (
                    ("TRI", NIFTY_TRI_ENDPOINT),
                    ("HIST", HIST_ENDPOINT),
                ):
                    page.wait_for_timeout(random.randint(1200, 2600))
                    res = page.evaluate(FETCH_JS, [endpoint, _body(name)])
                    recs = _records(res["text"])
                    n = len(recs) if isinstance(recs, list) else None
                    cols = list(recs[0].keys()) if recs else None
                    sample = recs[0] if recs else (res["text"][:160] if not recs else None)
                    hit = bool(recs)
                    print(
                        f"[{'HIT ' if hit else 'miss'}] {label:4s} status={res['status']:>4} "
                        f"rows={str(n):>5}  name={name!r}"
                    )
                    if hit:
                        print(f"         cols={cols}  first={sample}")
                    elif res["status"] != 200:
                        print(f"         body={sample!r}")
                    rows.append((name, label, res["status"], n))
        finally:
            browser.close()

    print("\n=== summary ===")
    for name in NAMES:
        wins = [r for r in rows if r[0] == name and r[3]]
        if wins:
            _, label, _, n = wins[0]
            print(f"  HIT  {name!r} via {label} ({n} rows)")
        else:
            print(f"  MISS {name!r} (no endpoint answered)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
