#!/usr/bin/env python3
"""Collect Value Research Online (VRO) published metrics for the mapped funds,
alongside our own matched-methodology figures, and print + optionally snapshot.

This is the data-collection utility behind the VRO-parity wire test
(``tests/integration/test_vro_parity.py``). For each fund in
``data/funds/vro_funds.csv`` it shows VRO's published values, ours, and the gap — so
the agreement is auditable by eye — for two families:

* trailing returns (point-to-point CAGR), and
* risk ratios (Mean / Std Dev / Sharpe / Sortino, trailing 3Y monthly; plus
  Beta / Alpha, which VRO publishes but we don't yet compute — no benchmark TRI).

Run from anywhere with the ``browser`` extra installed::

    ./venv/bin/python scripts/fetch_vro_metrics.py
    ./venv/bin/python scripts/fetch_vro_metrics.py --json outputs/vro_metrics.json
    ./venv/bin/python scripts/fetch_vro_metrics.py --periods 1Y,3Y,5Y,10Y
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a plain script from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolioanalyzer.loaders.mutual_fund import fetch_navs  # noqa: E402
from portfolioanalyzer.loaders.vro import (  # noqa: E402
    VRO_PERIODS,
    VRO_RISK_FREE_ANNUAL,
    VRO_RISK_PERIOD_YEARS,
    browser_extra_available,
    fetch_benchmark_tri,
    fetch_vro_metrics,
    load_vro_fund_map,
    trailing_cagr_pct,
    trailing_risk_ratios,
)


def _years(period: str) -> int | None:
    """Window length for a year-period tag (``3Y`` -> 3); None for VRO-only
    tags like YTD / 1m / 3m where a point-to-point CAGR comparison doesn't apply."""
    p = period.strip().upper()
    return int(p[:-1]) if p.endswith("Y") and p[:-1].isdigit() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write a JSON snapshot to this path")
    parser.add_argument(
        "--periods",
        default=",".join(VRO_PERIODS),
        help="comma-separated periods (default: 1Y,3Y,5Y)",
    )
    args = parser.parse_args()

    if not browser_extra_available():
        sys.exit(
            "The [browser] extra is required for VRO:\n"
            "  pip install '.[browser]' && playwright install chromium"
        )

    periods = tuple(p.strip() for p in args.periods.split(",") if p.strip())
    snapshot: list[dict] = []

    for fund in load_vro_fund_map():
        vro = fetch_vro_metrics(fund.vro_plan_id, fund.vro_slug, periods=periods)
        nav = fetch_navs(fund.mfapi_url)["nav"]
        anchor = nav.index.max()
        print(
            f"\n{fund.name}  "
            f"(mfapi {fund.mfapi_code} / VRO {fund.vro_plan_id}, latest NAV {anchor.date()})"
        )

        # --- trailing returns ---
        return_rows: dict[str, dict] = {}
        for period in periods:
            vro_val = vro.returns.get(period)
            years = _years(period)
            if vro_val is None:
                continue
            if years is None:
                print(f"  ret {period:>4}: VRO={vro_val:6.2f}%  (VRO-only)")
                return_rows[period] = {"vro": vro_val}
                continue
            ours = trailing_cagr_pct(nav, years)
            delta = ours - vro_val
            print(f"  ret {period:>4}: VRO={vro_val:6.2f}%  ours={ours:6.2f}%  Δ{delta:+.2f}pp")
            return_rows[period] = {"vro": vro_val, "ours": round(ours, 2), "delta_pp": round(delta, 2)}

        # --- risk ratios (trailing 3Y monthly) ---
        # Beta/Alpha need the fund's benchmark TRI; fetch it only where sourceable
        # (currently just NIFTY 100 → ICICI Bluechip). Without it, those two stay
        # VRO-only in the snapshot.
        bench_tri = fetch_benchmark_tri(fund.benchmark_index) if fund.benchmark_index else None
        ours_risk = trailing_risk_ratios(
            nav, years=VRO_RISK_PERIOD_YEARS, benchmark_nav=bench_tri
        )
        risk_rows: dict[str, dict] = {}
        for key in ("mean", "std_dev", "sharpe", "sortino", "beta", "alpha"):
            vro_val = vro.risk.get(key)
            our_val = ours_risk.get(key)
            if vro_val is None:
                continue
            if our_val is None:  # VRO publishes it but we can't source the benchmark
                print(f"  {key:>8}: VRO={vro_val:7.2f}  (VRO-only)")
                risk_rows[key] = {"vro": vro_val}
                continue
            delta = our_val - vro_val
            print(f"  {key:>8}: VRO={vro_val:7.2f}  ours={our_val:7.2f}  Δ{delta:+.2f}")
            risk_rows[key] = {"vro": vro_val, "ours": round(our_val, 2), "delta": round(delta, 2)}

        snapshot.append(
            {
                "mfapi_code": fund.mfapi_code,
                "vro_plan_id": fund.vro_plan_id,
                "name": fund.name,
                "latest_nav_date": str(anchor.date()),
                "returns": return_rows,
                "risk": risk_rows,
            }
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "basis": {
                        "returns": "point-to-point daily NAV CAGR",
                        "risk": f"trailing {VRO_RISK_PERIOD_YEARS}Y monthly; "
                        f"risk_free_annual={VRO_RISK_FREE_ANNUAL}",
                    },
                    "funds": snapshot,
                },
                indent=2,
            )
        )
        print(f"\nSnapshot written to {args.json}")


if __name__ == "__main__":
    main()
