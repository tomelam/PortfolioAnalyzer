#!/usr/bin/env python3
"""Read-only spot-check: SGB coupon load is a small fraction of capital.

Established the regression fixed in Part B1. Part A repointed the SGB engine's
gold input to the LBMA **USD** series while ``sgb_holding_civ`` summed the
rupee coupon into the same number — so each ₹63 coupon was added as if it were
$63, making each semiannual coupon ~47% of capital (≈520% coupon load for a
2020 tranche by 2026) instead of the contractual ~1.25%-of-nominal.

B1 converts each coupon to USD at its payment date's FRED ``DEXINUS`` USD/INR
rate before summing. After the fix each coupon is back to well under 1% of
current capital. This script prints the per-coupon load so the fix can be
re-verified by eye; it is not run by CI.

    venv/bin/python scripts/sgb_coupon_load_check.py
"""
from __future__ import annotations

import pandas as pd

from portfolioanalyzer.loaders.benchmark import load_timeseries_csv
from portfolioanalyzer.loaders.gold import load_gold_prices_per_gram
from portfolioanalyzer.sgb_holdings import (
    coupon_amount_per_payment,
    coupon_schedule,
    sgb_holding_civ,
)
from portfolioanalyzer.sgb_tranches import lookup_tranche

GOLD_CSV = "data/reference/gold_lbma_usd_daily.csv"
FX_CSV = "data/reference/DEXINUS.csv"


def main(tranche_id: str = "2020-21-VII") -> None:
    gold = load_gold_prices_per_gram(GOLD_CSV)
    fx = load_timeseries_csv(FX_CSV, "%Y-%m-%d", max_delay_days=None).value_series()

    t = lookup_tranche(tranche_id)
    civ = sgb_holding_civ(tranche_id, units_grams=1.0, gold_prices=gold, fx_inr_per_usd=fx)

    per_inr = coupon_amount_per_payment(
        1.0, float(t["issue_price_offline"]), float(t["coupon_rate_pct"])
    )
    paid = [d for d in coupon_schedule(t["issue_date"], t["maturity_date"])
            if pd.Timestamp(d) <= civ.index.max()]
    end_capital = 1.0 * gold.asof(civ.index.max())
    total_income_usd = civ.iloc[-1] - end_capital

    print(f"tranche {tranche_id}  issue {t['issue_date']}")
    print(f"per-coupon: ₹{per_inr:.2f} INR  |  coupons paid: {len(paid)}")
    print(f"total coupon income: ${total_income_usd:.2f} USD")
    print(f"end capital: ${end_capital:.2f}/gram")
    print(f"coupon income / capital: {100 * total_income_usd / end_capital:.1f}%")
    print(f"per coupon ≈ {100 * (total_income_usd / len(paid)) / end_capital:.2f}% of capital")


if __name__ == "__main__":
    main()
