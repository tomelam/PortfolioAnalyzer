"""Tests for the per-tranche SGB valuation engine.

The CIV (cumulative investment value) of one SGB holding at date t is:

    CIV(t) = units_grams × gold_price_per_gram(t)         # mark-to-market
           + Σ(coupons paid on or before t)               # accumulated income

Coupons are 2.5% p.a. on the nominal (offline) issue price × units,
paid semi-annually on the 6mo / 12mo / ... anniversaries of the issue
date through maturity (16 coupons over 8 years).

Anchors: ground-truth values are derived from the user's actual RBI
certificates (FY 2019-20 Series IX × 12 g, FY 2020-21 Series VII × 6 g).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from sgb_holdings import (
    coupon_amount_per_payment,
    coupon_schedule,
    sgb_holding_civ,
)

# ---------------------------------------------------------------------------
# Coupon math (closed-form, no time-series dependency)
# ---------------------------------------------------------------------------


def test_coupon_schedule_emits_16_dates_over_8_years() -> None:
    """A standard SGB has 16 semi-annual coupons over its 8-year life."""
    issue = dt.date(2020, 10, 20)  # FY 2020-21 Series VII
    maturity = dt.date(2028, 10, 20)
    dates = coupon_schedule(issue, maturity)
    assert len(dates) == 16
    # First coupon is 6 months after issue, not on issue date itself.
    assert dates[0] == dt.date(2021, 4, 20)
    # Final coupon coincides with maturity (paid alongside principal).
    assert dates[-1] == maturity


def test_coupon_schedule_handles_month_end_rollover() -> None:
    """Issue on the 31st rolls back to the 30th in 30-day months."""
    issue = dt.date(2020, 3, 31)
    maturity = dt.date(2028, 3, 31)
    dates = coupon_schedule(issue, maturity)
    # 6 months from 31 Mar → 30 Sep (Sep has 30 days, not 31).
    assert dates[0] == dt.date(2020, 9, 30)


def test_coupon_amount_for_2019_20_IX_holding_12g() -> None:
    """User's SBI cert: 12 g of 2019-20-IX at ₹4,070 offline.
    Nominal = 4,070 × 12 = ₹48,840. Coupon = 2.5%/2 × 48,840 = ₹610.50."""
    amount = coupon_amount_per_payment(
        units_grams=12,
        issue_price_offline=4070,
        coupon_rate_pct=2.5,
    )
    assert amount == pytest.approx(610.50)


def test_coupon_amount_for_2020_21_VII_holding_6g() -> None:
    """User's HDFC certs (lumped): 6 g of 2020-21-VII at ₹5,051 offline.
    Nominal = 5,051 × 6 = ₹30,306. Coupon = ₹378.825."""
    amount = coupon_amount_per_payment(
        units_grams=6,
        issue_price_offline=5051,
        coupon_rate_pct=2.5,
    )
    assert amount == pytest.approx(378.825)


# ---------------------------------------------------------------------------
# CIV time series
# ---------------------------------------------------------------------------


def _linear_gold(start_date: dt.date, end_date: dt.date,
                 start_inr_per_gram: float, end_inr_per_gram: float) -> pd.Series:
    """Helper: daily gold price series linearly interpolated between endpoints."""
    idx = pd.date_range(start_date, end_date, freq="D")
    n = len(idx)
    values = [start_inr_per_gram + (end_inr_per_gram - start_inr_per_gram) * i / (n - 1)
              for i in range(n)]
    return pd.Series(values, index=idx, name="gold_inr_per_gram")


def test_civ_at_issue_date_equals_units_times_gold_at_issue() -> None:
    """CIV at the issue date = capital only; no coupons paid yet."""
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2028, 10, 20), 5051.0, 15000.0)
    civ = sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold)
    issue_ts = pd.Timestamp("2020-10-20")
    expected = 6 * 5051.0
    assert civ.loc[issue_ts] == pytest.approx(expected)


def test_civ_jumps_by_coupon_amount_on_coupon_date() -> None:
    """The CIV gains exactly one coupon between the day before and the
    day of a coupon payment, in addition to the daily gold drift."""
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2028, 10, 20), 5051.0, 15000.0)
    civ = sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold)

    first_coupon = pd.Timestamp("2021-4-20")
    day_before = first_coupon - pd.Timedelta(days=1)
    coupon_amt = 378.825
    gold_step = (gold.loc[first_coupon] - gold.loc[day_before]) * 6
    actual_jump = civ.loc[first_coupon] - civ.loc[day_before]
    assert actual_jump == pytest.approx(coupon_amt + gold_step, rel=1e-9)


def test_civ_ends_at_maturity_when_gold_extends_past_it() -> None:
    """CIV is not defined past the bond's maturity (principal redeemed)."""
    # Gold extends to 2029 but bond matures 2028-10-20.
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2029, 12, 31), 5051.0, 16000.0)
    civ = sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold)
    assert civ.index.max() == pd.Timestamp("2028-10-20")


def test_civ_ends_at_gold_data_end_when_gold_stops_first() -> None:
    """CIV can't extend past the available gold data either."""
    # Gold stops 2024-06-30; bond matures 2028-10-20.
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2024, 6, 30), 5051.0, 12000.0)
    civ = sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold)
    assert civ.index.max() == pd.Timestamp("2024-6-30")


def test_civ_starts_at_issue_date_even_if_gold_history_predates() -> None:
    """If we pass gold data going back 5 years before issue, the CIV
    still only starts at issue date (no pre-issue valuation)."""
    gold = _linear_gold(dt.date(2015, 1, 1), dt.date(2028, 10, 20), 2500.0, 15000.0)
    civ = sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold)
    assert civ.index.min() == pd.Timestamp("2020-10-20")


def test_civ_terminal_value_matches_closed_form() -> None:
    """At maturity, CIV = units × gold(maturity) + 16 × coupon."""
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2028, 10, 20), 5051.0, 15000.0)
    civ = sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold)
    expected = 6 * 15000.0 + 16 * 378.825
    assert civ.iloc[-1] == pytest.approx(expected)


def test_civ_is_monotonic_under_monotonic_gold() -> None:
    """A monotonically rising gold price with positive coupons yields
    a monotonically rising CIV — no spurious dips."""
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2028, 10, 20), 5051.0, 15000.0)
    civ = sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold)
    diffs = civ.diff().dropna()
    assert (diffs >= 0).all(), f"CIV decreased on {(diffs < 0).sum()} day(s)"


def test_two_holdings_of_same_tranche_compose_linearly() -> None:
    """If you held two separate 6 g and 6 g of the same tranche, the
    combined CIV equals one 12 g holding. (Sanity for the lumping rule.)"""
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2028, 10, 20), 5051.0, 15000.0)
    civ_lumped = sgb_holding_civ("2020-21-VII", units_grams=12, gold_prices=gold)
    civ_a = sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold)
    civ_b = sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold)
    summed = civ_a + civ_b
    pd.testing.assert_series_equal(civ_lumped, summed, check_names=False)


def test_unknown_tranche_raises_keyerror() -> None:
    gold = _linear_gold(dt.date(2020, 1, 1), dt.date(2025, 1, 1), 4000.0, 8000.0)
    with pytest.raises(KeyError, match="unknown tranche"):
        sgb_holding_civ("nonexistent-tranche", units_grams=10, gold_prices=gold)
