#!/usr/bin/env python3
"""One-session probe: confirm the exact TRI-endpoint spelling for the equity
benchmarks used by the fund catalog (data/fund_catalog.csv).

The niftyindices live-watch master (LiveIndicesWatch_new.json) lists indices by
*abbreviated* names ("NIFTY SMLCAP 250", "LARGEMID250"); the TRI endpoint
(getTotalReturnIndexString) may want the full display spelling instead. This
script POSTs each candidate (abbreviated and full, for the ambiguous ones) in a
SINGLE stealth session — one navigate, then human-paced in-page POSTs — to avoid
the burst pattern that trips niftyindices' block. Records which spelling returns
rows so the catalog's `niftyindices_name` column is actually fetchable.
Throwaway diagnostics (kept + documented).
"""
from __future__ import annotations

import datetime as dt
import json
import random
import sys

from loaders.data_update import _NIFTY_UA, NIFTY_HIST_PAGE, NIFTY_TRI_ENDPOINT

CANDIDATES = [
    "NIFTY 100",          # control (known good)
    "NIFTY 50",
    "NIFTY 500",
    "NIFTY SMALLCAP 250",
    "NIFTY SMLCAP 250",
    "NIFTY MIDCAP 150",
    "NIFTY LARGEMIDCAP 250",
    "NIFTY LARGEMID250",
    "NIFTY IT",
    "NIFTY PHARMA",
]

START = (dt.date.today() - dt.timedelta(days=90)).strftime("%d-%b-%Y")
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
        return { ok: r.ok, status: r.status, text: await r.text() };
    } catch (e) { return { ok: false, status: -1, text: String(e) }; }
}"""


def _body(name: str) -> str:
    cinfo = (
        "{'name':'" + name + "','startDate':'" + START
        + "','endDate':'" + END + "','indexName':'" + name + "'}"
    )
    return json.dumps({"cinfo": cinfo})


def _rows(text: str):
    try:
        env = json.loads(text)
        inner = env.get("d") if isinstance(env, dict) else None
        return json.loads(inner) if inner else []
    except Exception:
        return None


def main() -> int:
    from playwright.sync_api import sync_playwright
    from playwright_stealth import Stealth

    out = []
    with Stealth().use_sync(sync_playwright()) as p:
        browser = p.chromium.launch(headless=True)
        try:
            ctx = browser.new_context(
                user_agent=_NIFTY_UA, viewport={"width": 1920, "height": 1080},
                locale="en-IN", timezone_id="Asia/Kolkata",
            )
            page = ctx.new_page()
            page.goto(NIFTY_HIST_PAGE, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(random.randint(1500, 3500))
            for name in CANDIDATES:
                page.wait_for_timeout(random.randint(1200, 2600))
                res = page.evaluate(FETCH_JS, [NIFTY_TRI_ENDPOINT, _body(name)])
                recs = _rows(res["text"])
                n = len(recs) if isinstance(recs, list) else None
                print(f"[{'HIT ' if recs else 'miss'}] rows={str(n):>4}  {name!r}")
                out.append((name, n))
        finally:
            browser.close()

    print("\n=== fetchable spellings ===")
    for name, n in out:
        if n:
            print(f"  OK  {name!r} ({n} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
