"""VRO-parity wire test: our trailing CAGR vs VRO's published figures.

A live (network + browser) check that our metrics reproduce Value Research
Online's published annualised trailing returns. VRO is Cloudflare-protected, so
this needs the optional ``browser`` extra (playwright + stealth); without it the
test skips. Marked ``network`` (skipped by default) and ``vro`` (run the set
with ``pytest -m vro``).

Methodology (reconciled 2026-06-18): VRO uses point-to-point **daily** NAVs —
latest NAV to exactly N years prior — so we compare against
``loaders.vro.trailing_cagr_pct``, NOT the month-end ``--metrics-method
monthly`` (which ran ~0.5–0.8pp low). Observed agreement was ≤0.25pp; the
assertion tolerance is a deliberately loose 1.0pp, leaving room for NAV-date
drift across funds and run days. Tighten once it has proven stable.
"""

from __future__ import annotations

import pytest

from loaders.mutual_fund import fetch_navs
from loaders.vro import (
    browser_extra_available,
    fetch_vro_trailing_returns,
    load_vro_fund_map,
    trailing_cagr_pct,
)

# Starting tolerance in percentage points (loose by design — see module docstring).
TOL_PP = 1.0
PERIODS = (("3Y", 3), ("5Y", 5))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.network,
    pytest.mark.vro,
    pytest.mark.skipif(
        not browser_extra_available(),
        reason="VRO parity needs the [browser] extra (playwright + playwright-stealth)",
    ),
]


@pytest.mark.parametrize("fund", load_vro_fund_map(), ids=lambda f: f.mfapi_code)
def test_our_trailing_cagr_matches_vro(fund) -> None:
    vro = fetch_vro_trailing_returns(
        fund.vro_plan_id, fund.vro_slug, periods=tuple(p for p, _ in PERIODS)
    )
    nav = fetch_navs(fund.mfapi_url)["nav"]
    for period, years in PERIODS:
        ours = trailing_cagr_pct(nav, years)
        assert abs(ours - vro[period]) < TOL_PP, (
            f"{fund.name} {period}: ours={ours:.2f}% vs VRO={vro[period]:.2f}% "
            f"(Δ{ours - vro[period]:+.2f}pp ≥ {TOL_PP}pp tolerance)"
        )
