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
  captured fragment), so they get a looser tolerance.

* **Beta / Alpha** (CAPM, vs the fund's *stated* benchmark) are asserted only for
  funds whose benchmark TRI is sourceable from niftyindices — currently just
  ICICI Bluechip (NIFTY 100); see ``VROFund.benchmark_index`` and the probe under
  ``scripts/probe_niftyindices_*.py``. The benchmark TRI is fetched live (one
  niftyindices hit, only for those funds) and fed to ``trailing_risk_ratios`` as
  ``benchmark_nav``. The tiered/hybrid debt benchmarks aren't on niftyindices'
  feed and the US FoF's Russell 3000 Growth is out of scope, so those funds'
  Beta/Alpha are still only collected, not asserted (see KANBAN).

Tolerances are starting points calibrated from the first live reconciliation;
adjust here if a methodology detail shifts them.
"""

from __future__ import annotations

import pytest

from portfolioanalyzer.loaders.data_update import NiftyEndpointMoved
from portfolioanalyzer.loaders.mutual_fund import fetch_navs
from portfolioanalyzer.loaders.vro import (
    browser_extra_available,
    fetch_benchmark_tri,
    fetch_vro_metrics,
    load_vro_fund_map,
    trailing_cagr_pct,
    trailing_risk_ratios,
)

# Trailing-return tolerance (percentage points); see module docstring.
TOL_RET_PP = 0.5
RETURN_PERIODS = (("3Y", 3), ("5Y", 5))

# Risk-ratio tolerances, calibrated from the live snapshot across all 5 funds
# (2026-06-19, month-end-anchored). Observed worst |Δ|: Mean 0.00, Std Dev 0.00,
# Sharpe 0.03 — so Mean/Std/Sharpe are tightened well below the old 1.0/1.0/0.2;
# the residual headroom covers month-boundary window slides (when VRO's monthly
# risk update and our last-complete-month anchor land in different months).
#
# Sortino is held at 0.20, NOT tightened: Franklin (the high-volatility FoF) sits
# at Δ0.14 because VRO's downside-deviation differs from every standard formula
# there (its Sharpe matches exactly, so it's a downside-dev convention quirk, not
# a risk-free mismatch). The other four funds agree to ≤0.03. 0.20 leaves ~0.06
# for run-day noise on top of that systematic, stable gap.
TOL_MEAN_PP = 0.4
TOL_STD_PP = 0.4
TOL_SHARPE = 0.05   # observed worst |delta| 0.02 after the 2026-09-07 risk-free correction
# Sortino is compared RELATIVELY, not absolutely. Mean, Std Dev and Sharpe all
# reconcile to <=0.02 (2026-09-07), so our returns series and annualisation are
# right and the entire residual sits in the downside-deviation denominator, which
# VRO computes non-standardly (established in earlier reconciliations). The size
# of that residual scales inversely with fund volatility: measured 2026-09-07 it
# is 0.0% for HDFC BAF, 5% for HDFC Hybrid Debt, 7.7% for Franklin and 13.5% for
# the corporate bond fund -- whose Sortino is large precisely because its downside
# deviation is tiny. A fixed absolute band therefore says "pass" for volatile
# funds and "fail" for stable ones at the same underlying disagreement.
# 20% covers the observed spread with headroom; a real regression in the numerator
# would move Sharpe first, and Sharpe is now held to 0.05.
TOL_SORTINO_REL = 0.20

# Beta / Alpha tolerances (only ICICI Bluechip, vs NIFTY 100 TRI). Calibrated
# from the first live reconciliation (2026-06-18 NAV): Beta Δ−0.011 (ours 0.909
# vs VRO 0.92), Alpha Δ−0.010pp (ours 3.320 vs VRO 3.33) — agreement as tight as
# the returns parity. VRO publishes both to 2 decimals (so ~0.005 is pure
# rounding); the headroom over the observed Δ covers that plus month-boundary
# window slides (cf. the risk-ratio note above). Beta unitless; Alpha in pp.
TOL_BETA = 0.05
TOL_ALPHA_PP = 0.40

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
        ("sharpe", TOL_SHARPE),
    ):
        assert abs(ours_risk[key] - vro.risk[key]) < tol, (
            f"{fund.name} {key}: ours={ours_risk[key]:.2f} vs VRO={vro.risk[key]:.2f} "
            f"(Δ{ours_risk[key] - vro.risk[key]:+.2f} ≥ {tol})"
        )

    # Sortino: relative, for the reason given at TOL_SORTINO_REL.
    rel = abs(ours_risk["sortino"] - vro.risk["sortino"]) / max(abs(vro.risk["sortino"]), 1e-9)
    assert rel < TOL_SORTINO_REL, (
        f"{fund.name} sortino: ours={ours_risk['sortino']:.2f} vs "
        f"VRO={vro.risk['sortino']:.2f} ({rel*100:.1f}% ≥ {TOL_SORTINO_REL*100:.0f}%)"
    )

    # --- CAPM Beta / Alpha (vs the stated benchmark TRI) ------------------
    # Asserted only where the benchmark is sourceable from niftyindices AND VRO
    # publishes the figure. Fetch the benchmark TRI once, here, for those funds.
    if fund.benchmark_index:
        try:
            bench_tri = fetch_benchmark_tri(fund.benchmark_index)
        except NiftyEndpointMoved as e:
            # niftyindices put the TRI endpoint behind a login on 2026-09-07, so
            # no benchmark series is obtainable. Skip only the CAPM assertions:
            # everything above -- returns, mean, std dev, Sharpe, Sortino -- has
            # already run and still guards this fund.
            pytest.skip(f"benchmark TRI unavailable, so Beta/Alpha cannot be checked: {e}")
        ours_capm = trailing_risk_ratios(nav, years=3, benchmark_nav=bench_tri)
        for key, tol in (("beta", TOL_BETA), ("alpha", TOL_ALPHA_PP)):
            if key not in vro.risk:
                continue  # VRO didn't publish it for this fund
            assert abs(ours_capm[key] - vro.risk[key]) < tol, (
                f"{fund.name} {key}: ours={ours_capm[key]:.2f} vs "
                f"VRO={vro.risk[key]:.2f} (Δ{ours_capm[key] - vro.risk[key]:+.2f} ≥ {tol})"
            )
