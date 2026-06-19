#!/usr/bin/env python3
"""Discovery spike: map Value Research Online's (VRO) per-fund API surface.

The VRO-parity work validates our metrics against VRO's published figures, but
only *trailing returns* were ever wired up (the ``peer-comparison-returns``
endpoint). VRO's fund pages also publish a risk-ratio family — Standard
Deviation, Sharpe, Sortino, Beta, Alpha, Mean — served by *some* same-origin
JSON API whose URL/shape was never mapped (a prior pass only established it is
NOT in the overview API). This script finds it empirically.

It reuses the exact stealth pattern in ``loaders.vro.fetch_vro_metrics``
(``Stealth().use_sync(sync_playwright())``, the real-Chrome UA, en-IN /
Asia/Kolkata, 1920x1080), but instead of fetching one known URL it records
*every* ``/api/`` request the page makes — on first load and after nudging the
page toward its Risk section — then dumps each URL with a JSON preview so the
risk-ratio endpoint can be identified by eye.

Run with the ``browser`` extra installed::

    ./venv/bin/python scripts/vro_discover_endpoints.py
    ./venv/bin/python scripts/vro_discover_endpoints.py --plan-id 15841 \\
        --slug icici-prudential-bluechip-fund-direct-plan --headed
    ./venv/bin/python scripts/vro_discover_endpoints.py --dump-dir outputs/vro_api

Nothing here is imported by the test suite; it is an investigative tool. Once
the risk-ratio endpoint is identified, capture its body as
``tests/fixtures/vro_risk_ratios.json`` for the offline parser tests.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Allow running as a plain script from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loaders.vro import (  # noqa: E402
    VRO_UA,
    browser_extra_available,
    vro_page_url,
)

# Default probe target: ICICI Prudential Bluechip (the fund the returns parity
# was first reconciled against).
DEFAULT_PLAN_ID = "15841"
DEFAULT_SLUG = "icici-prudential-bluechip-fund-direct-plan"

# Risk-ratio names we are hunting for in any captured payload, so a hit can be
# flagged automatically rather than only by eye.
RISK_HINTS = (
    "std",
    "sharpe",
    "sortino",
    "beta",
    "alpha",
    "mean",
    "deviation",
    "risk",
)


def _preview(text: str, limit: int = 600) -> str:
    """A compact one-screen preview of a (possibly large) response body."""
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + f"… (+{len(text) - limit} chars)"


def _looks_riskish(url: str, body: str) -> bool:
    blob = (url + " " + body[:4000]).lower()
    return sum(hint in blob for hint in RISK_HINTS) >= 2


def discover(
    plan_id: str,
    slug: str,
    *,
    headless: bool,
    dump_dir: Path | None,
    probe_risk_peers: list[str] | None = None,
) -> None:
    from playwright.sync_api import sync_playwright
    from playwright_stealth import Stealth

    # (url, body, content_type) for every /api/ response we observe.
    hits: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    def _on_response(resp) -> None:
        url = resp.url
        if "/api/" not in url or url in seen:
            return
        seen.add(url)
        ctype = (resp.headers or {}).get("content-type", "")
        try:
            body = resp.text()
        except Exception as e:  # body may be unavailable for some responses
            body = f"<unreadable: {e}>"
        hits.append((url, body, ctype))

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
            # The response listener (which calls resp.text() synchronously) is
            # only needed for full discovery; on the fast risk probe it adds a
            # potential sync-handler deadlock for no benefit.
            if probe_risk_peers is None:
                page.on("response", _on_response)

            url = vro_page_url(plan_id, slug)
            print(f"navigating: {url}", file=sys.stderr)
            page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            # Risk probe needs a longer settle (the same-origin fetch must wait
            # for the Cloudflare clearance to mint cookies); full discovery is
            # fine with a short pause.
            settle = 8000 if probe_risk_peers is not None else random.randint(3000, 5000)
            page.wait_for_timeout(settle)

            # Probe the risk-ratios fragment endpoint discovered in the page
            # bundle: GET /funds/risk-ratios-tab-data/ returns an HTML fragment
            # (fund + 4 peers). Fetch it same-origin so cookies/CF clearance ride
            # along, and dump it for fixture capture + parser design. This is a
            # fast path: it skips the scroll/click/HTML-dump exploration below
            # (which can tangle with ad-network iframes on this page).
            if probe_risk_peers is not None:
                # The risk endpoint keys on the fund's SHORT NAME (e.g.
                # "ICICI Pru Large Cap Dir") held in the page's #fund_name hidden
                # input — NOT the numeric plan_id. Peers come from #peer-fund-
                # search1..4. Read them straight from the cleared page.
                names = page.evaluate(
                    """() => {
                        const v = id => (document.getElementById(id) || {}).value || '';
                        return {
                            fund: v('fund_name'),
                            peers: [1,2,3,4].map(i => v('peer-fund-search' + i)).filter(Boolean),
                            lang: v('curr_lang') || 'en',
                        };
                    }"""
                )
                print(f"  page names: {names}", file=sys.stderr)
                from urllib.parse import quote

                params = [f"fund_name={quote(names['fund'])}"]
                params += [
                    f"fund_name{i}={quote(pn)}" for i, pn in enumerate(names["peers"], start=1)
                ]
                params.append(f"lang={names['lang'] or 'en'}")
                risk_url = (
                    "https://www.valueresearchonline.com/funds/risk-ratios-tab-data/?"
                    + "&".join(params)
                )
                # The dynamic /funds/ route holds an in-page XHR behind a CF
                # challenge headless, so capture it as a top-level navigation in
                # the already-cleared session (the CF cookie carries). Fall back
                # to a time-boxed in-page fetch if navigation yields nothing.
                # Navigate with wait_until='commit' (return as soon as the
                # response starts; don't wait for the CF challenge JS), then poll
                # the content until the cf_clearance cookie auto-clears the
                # interstitial and the real fragment renders.
                frag = ""
                try:
                    page.goto(risk_url, wait_until="commit", timeout=30_000)
                except Exception as e:
                    print(f"goto(commit): {e}", file=sys.stderr)
                for attempt in range(8):
                    page.wait_for_timeout(2000)
                    frag = page.content()
                    cleared = "Just a moment" not in frag and "challenge" not in frag.lower()
                    print(
                        f"  poll {attempt}: {len(frag)} chars, cleared={cleared}",
                        file=sys.stderr,
                    )
                    if cleared and len(frag) > 500:
                        break
                # CF is now cleared for this path; the route only returns real
                # data to an XHR (it served a "data unavailable" stub to the bare
                # navigation). We are on the same origin, so an in-page fetch with
                # the XHR header now both passes CF *and* satisfies the route.
                xhr = page.evaluate(
                    """async (url) => {
                        const c = new AbortController();
                        const t = setTimeout(() => c.abort(), 20000);
                        try {
                            const r = await fetch(url, {
                                headers: {'X-Requested-With': 'XMLHttpRequest'},
                                signal: c.signal,
                            });
                            return 'HTTP ' + r.status + '\\n' + (await r.text());
                        } catch (e) { return 'FETCH_ERROR: ' + e; }
                        finally { clearTimeout(t); }
                    }""",
                    risk_url,
                )
                print(f"  XHR result: {len(xhr)} chars", file=sys.stderr)
                if "FETCH_ERROR" not in xhr and "underlying data is unavailable" not in xhr:
                    frag = xhr
                print(f"\n=== risk-ratios-tab-data fragment ({len(frag)} chars) ===", file=sys.stderr)
                print(f"URL: {risk_url}", file=sys.stderr)
                if dump_dir:
                    dump_dir.mkdir(parents=True, exist_ok=True)
                    (dump_dir / "risk_fragment.html").write_text(frag)
                    print(f"written to {dump_dir / 'risk_fragment.html'}", file=sys.stderr)
                return  # fast path: skip the scroll/click exploration below

            # Nudge the page toward lazy-loaded sections: walk the whole page in
            # small steps (intersection-observer sections load on dwell), then
            # try to reveal/click anything labelled "Risk".
            for _ in range(40):
                page.mouse.wheel(0, 1200)
                page.wait_for_timeout(400)
            for sel in (
                "a:has-text('Risk')",
                "button:has-text('Risk')",
                "[href*='risk']",
                "text=/risk measures/i",
                "text=/risk ratio/i",
                "text=/sharpe/i",
            ):
                try:
                    loc = page.locator(sel).first
                    if loc.count() > 0:
                        loc.scroll_into_view_if_needed(timeout=2000)
                        loc.click(timeout=2000)
                        page.wait_for_timeout(1500)
                except Exception:
                    pass
            page.wait_for_timeout(2000)

            # Dump rendered HTML + enumerate API paths and nav links referenced
            # anywhere in the DOM, so the risk endpoint can be found from the
            # bundle rather than guessed.
            html = page.content()
            if dump_dir:
                dump_dir.mkdir(parents=True, exist_ok=True)
                (dump_dir / "page.html").write_text(html)
            import re as _re

            api_paths = sorted(set(_re.findall(r"/api/funds/[A-Za-z0-9_\-/]+", html)))
            print("\n--- /api/funds/ paths referenced in the DOM/HTML ---", file=sys.stderr)
            for ap in api_paths:
                print(f"  {ap}", file=sys.stderr)
            for term in ("sharpe", "sortino", "standard deviation", "alpha", "beta"):
                print(f"  HTML mentions {term!r}: {term in html.lower()}", file=sys.stderr)

            nav = page.eval_on_selector_all(
                "a[href]",
                "els => els.map(e => [e.textContent.trim(), e.getAttribute('href')])"
                ".filter(([t,h]) => h && h.includes('"
                + plan_id
                + "'))",
            )
            print("\n--- in-page links mentioning this fund id ---", file=sys.stderr)
            for text, href in nav:
                print(f"  {text[:40]!r:44} -> {href}", file=sys.stderr)
        finally:
            browser.close()

    print(f"\nCaptured {len(hits)} distinct /api/ responses.\n", file=sys.stderr)
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    for i, (url, body, ctype) in enumerate(hits):
        flag = "  <-- RISK-RATIO CANDIDATE" if _looks_riskish(url, body) else ""
        print("=" * 100)
        print(f"[{i}] {url}{flag}")
        print(f"     content-type: {ctype}")
        print("-" * 100)
        print(_preview(body))
        print()
        if dump_dir:
            # Mechanically extracted artifact: full body, named by index, with
            # the source URL recorded inside for provenance.
            (dump_dir / f"api_{i:02d}.json").write_text(
                json.dumps({"_source_url": url, "_content_type": ctype, "body": body}, indent=2)
            )

    if dump_dir:
        print(f"\nFull bodies written to {dump_dir}/", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-id", default=DEFAULT_PLAN_ID)
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--headed", action="store_true", help="show the browser window")
    parser.add_argument(
        "--dump-dir",
        type=Path,
        help="write each full /api/ response body here (for fixture capture)",
    )
    parser.add_argument(
        "--probe-risk-peers",
        help="comma-separated peer plan_ids; fetch the risk-ratios HTML fragment "
        "for this fund + these peers",
    )
    args = parser.parse_args()

    if not browser_extra_available():
        sys.exit(
            "The [browser] extra is required:\n"
            "  pip install '.[browser]' && playwright install chromium"
        )

    peers = (
        [p.strip() for p in args.probe_risk_peers.split(",") if p.strip()]
        if args.probe_risk_peers
        else None
    )
    discover(
        args.plan_id,
        args.slug,
        headless=not args.headed,
        dump_dir=args.dump_dir,
        probe_risk_peers=peers,
    )


if __name__ == "__main__":
    main()
