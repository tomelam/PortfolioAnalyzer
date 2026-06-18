#!/usr/bin/env python3
"""Collect Value Research Online (VRO) published trailing returns for the mapped
funds, alongside our own point-to-point CAGR, and print + optionally snapshot
them.

This is the data-collection utility behind the VRO-parity wire test
(``tests/integration/test_vro_parity.py``). For each fund in
``data/vro_funds.csv`` it shows, per period, VRO's published annualised return,
our ``trailing_cagr_pct``, and the gap — so the agreement is auditable by eye.

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

from loaders.mutual_fund import fetch_navs  # noqa: E402
from loaders.vro import (  # noqa: E402
    VRO_PERIODS,
    browser_extra_available,
    fetch_vro_trailing_returns,
    load_vro_fund_map,
    trailing_cagr_pct,
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
        vro = fetch_vro_trailing_returns(fund.vro_plan_id, fund.vro_slug, periods=periods)
        nav = fetch_navs(fund.mfapi_url)["nav"]
        anchor = nav.index.max()
        print(
            f"\n{fund.name}  "
            f"(mfapi {fund.mfapi_code} / VRO {fund.vro_plan_id}, latest NAV {anchor.date()})"
        )
        rows: dict[str, dict] = {}
        for period in periods:
            vro_val = vro.get(period)
            years = _years(period)
            if years is None:
                print(f"  {period:>4}: VRO={vro_val:6.2f}%  (VRO-only)")
                rows[period] = {"vro": vro_val}
                continue
            ours = trailing_cagr_pct(nav, years)
            delta = ours - vro_val
            print(f"  {period:>4}: VRO={vro_val:6.2f}%  ours={ours:6.2f}%  Δ{delta:+.2f}pp")
            rows[period] = {"vro": vro_val, "ours": round(ours, 2), "delta_pp": round(delta, 2)}
        snapshot.append(
            {
                "mfapi_code": fund.mfapi_code,
                "vro_plan_id": fund.vro_plan_id,
                "name": fund.name,
                "latest_nav_date": str(anchor.date()),
                "returns": rows,
            }
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(snapshot, indent=2))
        print(f"\nSnapshot written to {args.json}")


if __name__ == "__main__":
    main()
