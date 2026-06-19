#!/usr/bin/env python3
"""One-off probe/discovery: which VRO-fund benchmarks are fetchable from niftyindices?

Beta/Alpha parity needs each fund's *stated benchmark* TRI series. Four of the
five port-1 benchmarks are NIFTY-family indices (the fifth, Russell 3000 Growth,
is deliberately out of scope — see KANBAN). niftyindices' equity TRI endpoint
serves NIFTY 100 fine, but the tiered/hybrid debt indices returned empty for the
obvious name spellings — so this script first DISCOVERS the exact registered
index names from the historical-data page, then matches our three debt/hybrid
benchmarks against that list and test-fetches the matches.

All work happens in ONE stealth browser session (avoid the burst pattern that
trips niftyindices' block). Throwaway diagnostics; not wired into the app.
"""
from __future__ import annotations

import datetime as dt
import json
import random
import re
import sys

from loaders.data_update import (
    _NIFTY_UA,
    NIFTY_HIST_PAGE,
    NIFTY_TRI_ENDPOINT,
)

HIST_ENDPOINT = "https://www.niftyindices.com/Backpage.aspx/getHistoricaldatatabletoString"

# Substrings we want to locate in the registered-name list (lowercased match).
WANTED = {
    "ICICI Bluechip": ["nifty 100"],
    "HDFC Balanced Advantage": ["hybrid composite debt 65:35", "65:35"],
    "HDFC Hybrid Debt": ["hybrid composite debt 15:85", "15:85"],
    "ICICI Corporate Bond": ["corporate bond", "a-ii"],
}

START = (dt.date.today() - dt.timedelta(days=120)).strftime("%d-%b-%Y")
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
        const text = await r.text();   // FULL text, no truncation
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
    """Return parsed record list from a {"d":"[...]"} envelope, or None."""
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

            # 1) Discover registered index names. Try <option> values/text across
            #    every <select>, plus any JS array of names embedded in the page.
            names = page.evaluate(
                """() => {
                    const out = new Set();
                    document.querySelectorAll('select option').forEach(o => {
                        const v = (o.value || '').trim();
                        const t = (o.textContent || '').trim();
                        if (v) out.add(v);
                        if (t) out.add(t);
                    });
                    return Array.from(out);
                }"""
            )
            names = sorted({n for n in names if n and len(n) < 120})
            print(f"discovered {len(names)} option strings from page dropdowns\n")

            # Surface NIFTY-ish names so we can eyeball debt/hybrid spellings.
            niftyish = [n for n in names if re.search(r"nifty|hybrid|debt|bond", n, re.I)]
            print("--- NIFTY/debt/hybrid/bond option strings ---")
            for n in niftyish:
                print(f"    {n!r}")
            print()

            # 2) For each wanted benchmark, find candidate registered names.
            matches = {}
            for fund, subs in WANTED.items():
                cands = [
                    n for n in names
                    if any(s in n.lower() for s in subs)
                ]
                matches[fund] = cands
                print(f"[match] {fund:24s} -> {cands or 'NONE'}")
            print()

            # 3) Test-fetch each candidate (TRI first, then HIST) in this session.
            results = []
            for fund, cands in matches.items():
                done = False
                for name in cands:
                    if done:
                        break
                    for label, endpoint in (("TRI", NIFTY_TRI_ENDPOINT), ("HIST", HIST_ENDPOINT)):
                        page.wait_for_timeout(random.randint(1200, 2600))
                        res = page.evaluate(FETCH_JS, [endpoint, _body(name)])
                        recs = _records(res["text"])
                        n = len(recs) if isinstance(recs, list) else None
                        cols = list(recs[0].keys()) if recs else None
                        hit = bool(recs)
                        print(
                            f"[{'HIT ' if hit else 'miss'}] {fund:24s} {label:4s} "
                            f"name={name!r} status={res['status']} rows={n} cols={cols}"
                        )
                        results.append((fund, name, label, n))
                        if hit:
                            done = True
                            break
        finally:
            browser.close()

    print("\n=== summary ===")
    for fund in WANTED:
        wins = [r for r in results if r[0] == fund and r[3]]
        if wins:
            f, name, label, n = wins[0]
            print(f"  {fund:24s} OK via {label} as {name!r} ({n} rows)")
        else:
            print(f"  {fund:24s} NOT FETCHABLE (no candidate name answered)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
