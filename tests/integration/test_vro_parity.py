"""VRO-parity wire test: our metrics vs VRO's published figures.

A live (network + browser) check that our numbers reproduce Value Research
Online's. VRO is Cloudflare-protected, so this needs the optional ``browser``
extra (playwright + stealth); without it the test skips. Marked ``network``
(skipped by default) and ``vro`` (run the set with ``pytest -m vro``).

Two metric families, collected in one stealth session via
``loaders.vro.fetch_vro_metrics``:

* **Trailing returns** (reconciled 2026-06-18): VRO uses point-to-point *daily*
  NAVs, so we compare against ``trailing_cagr_pct`` (not the month-end
  ``monthly`` method). Agreement was ≤0.25pp; tolerance 0.5pp.

* **Risk ratios** (Mean / Std Dev / Sharpe / Sortino): VRO computes these on a
  trailing-3Y, *monthly* basis, mirrored by ``trailing_risk_ratios``
  (``metrics.*`` with ``periods_per_year=12``). Mean and Std Dev are
  risk-free-independent and the firmest targets; Sharpe / Sortino additionally
  depend on VRO's assumed risk-free (``VRO_RISK_FREE_ANNUAL``, back-solved from a
  captured fragment), so they get a looser tolerance. **Beta / Alpha** are
  published by VRO and fetched, but asserting ours needs each fund's *stated*
  benchmark TRI, which the repo does not ship (only NIFTY TRI) — so they are
  collected for the record but not yet asserted (see KANBAN).

Tolerances are starting points calibrated from the first live reconciliation;
adjust here if a methodology detail shifts them.
"""

from __future__ import annotations

import pytest

from loaders.mutual_fund import fetch_navs
from loaders.vro import (
    browser_extra_available,
    fetch_vro_metrics,
    load_vro_fund_map,
    trailing_cagr_pct,
    trailing_risk_ratios,
)

# Trailing-return tolerance (percentage points); see module docstring.
TOL_RET_PP = 0.5
RETURN_PERIODS = (("3Y", 3), ("5Y", 5))

# Risk-ratio tolerances. Mean/Std Dev are pure return-series stats (pp); Sharpe/
# Sortino additionally carry the risk-free assumption, hence looser (ratio units).
TOL_MEAN_PP = 1.0
TOL_STD_PP = 1.0
TOL_RATIO = 0.2

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
def test_our_metrics_match_vro(fund) -> None:
    vro = fetch_vro_metrics(
        fund.vro_plan_id, fund.vro_slug, periods=tuple(p for p, _ in RETURN_PERIODS)
    )
    nav = fetch_navs(fund.mfapi_url)["nav"]

    # --- trailing returns -------------------------------------------------
    for period, years in RETURN_PERIODS:
        ours = trailing_cagr_pct(nav, years)
        assert abs(ours - vro.returns[period]) < TOL_RET_PP, (
            f"{fund.name} {period}: ours={ours:.2f}% vs VRO={vro.returns[period]:.2f}% "
            f"(Δ{ours - vro.returns[period]:+.2f}pp ≥ {TOL_RET_PP}pp)"
        )

    # --- risk ratios (trailing 3Y, monthly) -------------------------------
    ours_risk = trailing_risk_ratios(nav, years=3)
    for key, tol in (
        ("mean", TOL_MEAN_PP),
        ("std_dev", TOL_STD_PP),
        ("sharpe", TOL_RATIO),
        ("sortino", TOL_RATIO),
    ):
        assert abs(ours_risk[key] - vro.risk[key]) < tol, (
            f"{fund.name} {key}: ours={ours_risk[key]:.2f} vs VRO={vro.risk[key]:.2f} "
            f"(Δ{ours_risk[key] - vro.risk[key]:+.2f} ≥ {tol})"
        )

    # Beta/Alpha are collected when VRO publishes them (only for funds with a
    # comparable benchmark — a US-equity FoF / some debt funds have neither), but
    # not asserted: parity needs each fund's stated benchmark TRI, not shipped yet.
