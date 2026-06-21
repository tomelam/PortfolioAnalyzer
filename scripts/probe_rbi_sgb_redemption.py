#!/usr/bin/env python3
"""Discovery spike — does a fetchable, machine-readable RBI source of SGB
*redemption* prices (₹/gram) exist? (Phase B2, Step 0 — the fail-loud gate.)

This is a **throwaway diagnostic**, not imported by the app or CI. It records
*how* the B2 Step-0 source decision was established so it can be re-verified.

Conclusion (established 2026-06-21, run from a residential IP):

  PASS. RBI publishes each premature/maturity redemption price as an ordinary
  press release, reachable over plain ``requests`` (no stealth browser):

    1. ENUMERATION — the press-release SEARCH endpoint
         https://rbi.org.in/Scripts/SearchResults.aspx?search=<terms>
       and the plural listing https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx
       both return HTTP 200 with NO CAPTCHA, and list "(Premature) redemption …
       Sovereign Gold Bond" titles each linked by ``?prid=<N>``. This replaces
       the CAPTCHA-walled mobile directory the plan flagged
       (``m.rbi.org.in/Scripts/BS_SwarnaBharat.aspx`` — confirmed CAPTCHA below).

    2. DETAIL/PARSE — each press release
         https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=<N>
       fetches cleanly and carries a parseable title (tranche reference + the
       redemption date) plus the ₹/unit price (1 SGB unit = 1 gram of 999 gold).
       Example: prid 52638 → "Redemption Price for premature redemption due on
       November 30, 2021 (Series I of SGB 2015)" → ₹4808 per unit.

  So B2 ingest can be a plain-requests fetcher: SearchResults → collect SGB
  redemption PRIDs → fetch each PR → regex the date + ₹/unit. No IBJA paid API
  and no stealth browser required. (IBJA ``SGBPdf`` PDFs remain a fallback only.)

Run: ``python scripts/probe_rbi_sgb_redemption.py``
"""

from __future__ import annotations

import re
import sys
import urllib.request

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
_SEARCH = (
    "https://rbi.org.in/Scripts/SearchResults.aspx"
    "?search=premature%20redemption%20sovereign%20gold%20bond"
)
_DIRECTORY = "https://m.rbi.org.in/Scripts/BS_SwarnaBharat.aspx"
_PR = "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid="
_SAMPLE_REDEMPTION_PRID = "52638"  # Nov 29 2021 PR, SGB 2015 Series I redemption


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (trusted host)
        return resp.read().decode("utf-8", errors="replace")


def _strip(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def main() -> int:
    ok = True

    # 1. Directory page is expected to be CAPTCHA-walled (route around it).
    dir_html = _get(_DIRECTORY)
    walled = bool(re.search(r"captcha", dir_html, re.I))
    print(f"[directory ] BS_SwarnaBharat.aspx  captcha={walled}  "
          f"(expected True — we do NOT depend on this page)")

    # 2. Enumeration: search endpoint must be open and list SGB redemption PRIDs.
    search_html = _get(_SEARCH)
    if re.search(r"captcha", search_html, re.I):
        print("[enumerate ] SearchResults.aspx is CAPTCHA-walled — FAIL")
        ok = False
        prids: list[str] = []
    else:
        prids = sorted(set(re.findall(r"prid=(\d+)", search_html)))
        hits = len(re.findall(r"premature redemption", search_html, re.I))
        print(f"[enumerate ] SearchResults.aspx  HTTP-OK  "
              f"'premature redemption' hits={hits}  distinct prids={len(prids)}")
        if not prids:
            ok = False

    # 3. Detail/parse: a known redemption PR must yield date + ₹/unit price.
    pr_html = _get(_PR + _SAMPLE_REDEMPTION_PRID)
    if re.search(r"captcha", pr_html, re.I):
        print(f"[detail    ] prid={_SAMPLE_REDEMPTION_PRID} CAPTCHA-walled — FAIL")
        ok = False
    else:
        text = _strip(pr_html)
        title = re.search(r"(Premature redemption[^.]+?\))", text)
        price = re.findall(r"(?:₹|Rs\.?)\s?[0-9][0-9,]{2,}", text)
        print(f"[detail    ] prid={_SAMPLE_REDEMPTION_PRID}  "
              f"title={title.group(1) if title else '(?)'!r}")
        print(f"[detail    ] parsed ₹/unit figures: {price[:5]}")
        if not (title and price):
            ok = False

    print("\nSTEP 0:", "PASS — buildable on plain requests" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
