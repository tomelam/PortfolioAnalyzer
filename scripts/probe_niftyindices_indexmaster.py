#!/usr/bin/env python3
"""Discovery: find niftyindices' index-name master + the right endpoint for the
tiered/hybrid debt benchmarks.

The historical-data page loads its index list dynamically (autocomplete), so the
static <option> tags don't carry the names. This script captures the page's
network responses and the autocomplete results, then greps them for the exact
registered names of our three debt/hybrid benchmarks. One stealth session.
Throwaway diagnostics.
"""
from __future__ import annotations

import random
import re
import sys

from portfolioanalyzer.loaders.data_update import _NIFTY_UA, NIFTY_HIST_PAGE

NEEDLES = re.compile(r"hybrid|composite debt|corporate bond|nifty 100|15:85|65:35|a-ii", re.I)


def main() -> int:
    from playwright.sync_api import sync_playwright
    from playwright_stealth import Stealth

    captured: list[tuple[str, str]] = []

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

            def on_response(resp):
                url = resp.url
                try:
                    if resp.request.resource_type in ("image", "font", "stylesheet", "media"):
                        return
                    body = resp.text()
                except Exception:
                    return
                if NEEDLES.search(body) or "getIndex" in url or "autocomplete" in url.lower():
                    captured.append((url, body[:4000]))

            page.on("response", on_response)
            page.goto(NIFTY_HIST_PAGE, wait_until="networkidle", timeout=45000)
            page.wait_for_timeout(random.randint(2000, 3500))

            # Try to trigger the index-search autocomplete by typing into any
            # text input that looks like the index picker.
            for sel in ("input#txtindexname", "input[type=text]", "input"):
                try:
                    box = page.query_selector(sel)
                    if box:
                        box.click()
                        box.type("NIFTY 50 Hybrid", delay=120)
                        page.wait_for_timeout(2500)
                        break
                except Exception:
                    continue
            page.wait_for_timeout(2000)
        finally:
            browser.close()

    print(f"captured {len(captured)} candidate responses\n")
    seen_names: set[str] = set()
    for url, body in captured:
        print(f"--- {url}")
        # Pull quoted strings that look like index names.
        for m in re.findall(r'"([^"]*(?:NIFTY|Nifty)[^"]{0,60})"', body):
            if NEEDLES.search(m) or "hybrid" in m.lower() or "bond" in m.lower():
                seen_names.add(m.strip())
        print(f"    {body[:300]!r}\n")

    print("=== candidate registered names (hybrid/debt/bond/100) ===")
    for n in sorted(seen_names):
        print(f"    {n!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
