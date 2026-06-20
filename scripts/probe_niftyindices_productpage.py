#!/usr/bin/env python3
"""Thread-5 probe: the hybrid/debt index PRODUCT pages render a performance
chart — what data endpoint feeds it, and does that endpoint expose a series we
can pull?

The Backpage historical/TRI endpoints don't serve these indices (proven by
`probe_niftyindices_hybrid_exact.py`). But each index has a product page
(/indices/multi-asset/hybrid-indices/...) with a returns chart. This script
navigates the 65:35 product page in one stealth session and captures every
Backpage.aspx / data XHR it fires, so we can see whether any of them carries a
usable historical series (and with what request shape). Throwaway diagnostics.
"""
from __future__ import annotations

import random
import sys

from portfolioanalyzer.loaders.data_update import _NIFTY_UA

PRODUCT_URL = (
    "https://www.niftyindices.com/indices/multi-asset/hybrid-indices/"
    "nifty-50-hybrid-composite-debt-65-35-index"
)


def main() -> int:
    from playwright.sync_api import sync_playwright
    from playwright_stealth import Stealth

    captured: list[tuple[str, str, str]] = []

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
                rt = resp.request.resource_type
                if rt in ("image", "font", "stylesheet", "media", "script"):
                    return
                if "Backpage.aspx" in url or "api" in url.lower() or "/data" in url.lower():
                    try:
                        body = resp.text()
                    except Exception:
                        body = "<no-body>"
                    post = resp.request.post_data or ""
                    captured.append((url, post[:300], body[:500]))

            page.on("response", on_response)
            page.goto(PRODUCT_URL, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(random.randint(3000, 5000))
            # Click any performance/returns period toggles to provoke chart XHRs.
            for txt in ("1Y", "3Y", "5Y", "Since Inception", "Total Returns"):
                try:
                    el = page.get_by_text(txt, exact=False).first
                    if el and el.is_visible():
                        el.click(timeout=2000)
                        page.wait_for_timeout(1500)
                except Exception:
                    continue
        finally:
            browser.close()

    print(f"captured {len(captured)} data XHRs\n")
    for url, post, body in captured:
        print(f"--- {url}")
        if post:
            print(f"    POST: {post!r}")
        print(f"    BODY: {body!r}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
