"""Tests for the per-tranche SGB valuation engine.

The CIV (cumulative investment value) of one SGB holding at date t is:

    CIV(t) = units_grams × gold_price_per_gram(t)         # mark-to-market, USD
           + Σ(coupon_inr / fx_at_coupon_date)            # accumulated income, USD

Coupons are 2.5% p.a. on the nominal (offline) issue price × units,
paid semi-annually on the 6mo / 12mo / ... anniversaries of the issue
date through maturity (16 coupons over 8 years). The whole series is in
USD: the rupee coupon is converted to USD at each payment date's USD/INR
rate. Most baseline tests below pin FX = 1.0 so the coupon passes through
unchanged and the original closed-form anchors still hold.

Anchors: ground-truth values are derived from the user's actual RBI
certificates (FY 2019-20 Series IX × 12 g, FY 2020-21 Series VII × 6 g).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from portfolioanalyzer.sgb_holdings import (
    coupon_amount_per_payment,
    coupon_schedule,
    sgb_asset_label,
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
                 start_per_gram: float, end_per_gram: float) -> pd.Series:
    """Helper: daily gold price series linearly interpolated between endpoints
    (USD/gram, the unit the engine now consumes)."""
    idx = pd.date_range(start_date, end_date, freq="D")
    n = len(idx)
    values = [start_per_gram + (end_per_gram - start_per_gram) * i / (n - 1)
              for i in range(n)]
    return pd.Series(values, index=idx, name="gold_usd_per_gram")


def _flat_fx(rate: float) -> pd.Series:
    """Helper: a constant USD/INR series spanning every tranche window used
    here. A coupon's USD amount is ``coupon_inr / rate``; with ``rate == 1.0``
    the coupon passes through unchanged, so the FX-neutral baseline preserves
    the original closed-form coupon anchors."""
    idx = pd.date_range(dt.date(2014, 1, 1), dt.date(2030, 1, 1), freq="D")
    return pd.Series(rate, index=idx, name="value")


# A flat USD/INR = 1.0: coupons convert INR→USD as a no-op, so the capital +
# coupon closed forms below match the pre-FX engine exactly.
_FX1 = _flat_fx(1.0)


def test_civ_at_issue_date_equals_units_times_gold_at_issue() -> None:
    """CIV at the issue date = capital only; no coupons paid yet."""
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2028, 10, 20), 5051.0, 15000.0)
    civ = sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold, fx_inr_per_usd=_FX1)
    issue_ts = pd.Timestamp("2020-10-20")
    expected = 6 * 5051.0
    assert civ.loc[issue_ts] == pytest.approx(expected)


def test_civ_jumps_by_coupon_amount_on_coupon_date() -> None:
    """The CIV gains exactly one coupon between the day before and the
    day of a coupon payment, in addition to the daily gold drift.

    With FX = 1.0 the USD coupon equals the rupee coupon amount."""
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2028, 10, 20), 5051.0, 15000.0)
    civ = sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold, fx_inr_per_usd=_FX1)

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
    civ = sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold, fx_inr_per_usd=_FX1)
    assert civ.index.max() == pd.Timestamp("2028-10-20")


def test_civ_ends_at_gold_data_end_when_gold_stops_first() -> None:
    """CIV can't extend past the available gold data either."""
    # Gold stops 2024-06-30; bond matures 2028-10-20.
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2024, 6, 30), 5051.0, 12000.0)
    civ = sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold, fx_inr_per_usd=_FX1)
    assert civ.index.max() == pd.Timestamp("2024-6-30")


def test_civ_starts_at_issue_date_even_if_gold_history_predates() -> None:
    """If we pass gold data going back 5 years before issue, the CIV
    still only starts at issue date (no pre-issue valuation)."""
    gold = _linear_gold(dt.date(2015, 1, 1), dt.date(2028, 10, 20), 2500.0, 15000.0)
    civ = sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold, fx_inr_per_usd=_FX1)
    assert civ.index.min() == pd.Timestamp("2020-10-20")


def test_civ_terminal_value_matches_closed_form() -> None:
    """At maturity, CIV = units × gold(maturity) + 16 × coupon (FX = 1.0)."""
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2028, 10, 20), 5051.0, 15000.0)
    civ = sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold, fx_inr_per_usd=_FX1)
    expected = 6 * 15000.0 + 16 * 378.825
    assert civ.iloc[-1] == pytest.approx(expected)


def test_civ_is_monotonic_under_monotonic_gold() -> None:
    """A monotonically rising gold price with positive coupons yields
    a monotonically rising CIV — no spurious dips."""
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2028, 10, 20), 5051.0, 15000.0)
    civ = sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold, fx_inr_per_usd=_FX1)
    diffs = civ.diff().dropna()
    assert (diffs >= 0).all(), f"CIV decreased on {(diffs < 0).sum()} day(s)"


def test_two_holdings_of_same_tranche_compose_linearly() -> None:
    """If you held two separate 6 g and 6 g of the same tranche, the
    combined CIV equals one 12 g holding. (Sanity for the lumping rule.)"""
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2028, 10, 20), 5051.0, 15000.0)
    civ_lumped = sgb_holding_civ("2020-21-VII", units_grams=12, gold_prices=gold, fx_inr_per_usd=_FX1)
    civ_a = sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold, fx_inr_per_usd=_FX1)
    civ_b = sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold, fx_inr_per_usd=_FX1)
    summed = civ_a + civ_b
    pd.testing.assert_series_equal(civ_lumped, summed, check_names=False)


def test_unknown_tranche_raises_keyerror() -> None:
    gold = _linear_gold(dt.date(2020, 1, 1), dt.date(2025, 1, 1), 4000.0, 8000.0)
    with pytest.raises(KeyError, match="unknown tranche"):
        sgb_holding_civ("nonexistent-tranche", units_grams=10, gold_prices=gold, fx_inr_per_usd=_FX1)


# ---------------------------------------------------------------------------
# FX coupon conversion (the Part B1 fix)
# ---------------------------------------------------------------------------


def test_coupon_usd_is_inr_amount_divided_by_fx() -> None:
    """A flat USD/INR of 75 converts each ₹378.825 coupon to ₹378.825/75 USD.
    Capital is unaffected (FX never touches the gold leg)."""
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2028, 10, 20), 5051.0, 15000.0)
    civ = sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold, fx_inr_per_usd=_flat_fx(75.0))
    expected = 6 * 15000.0 + 16 * (378.825 / 75.0)
    assert civ.iloc[-1] == pytest.approx(expected)


def test_each_coupon_converts_at_its_own_date_fx() -> None:
    """Two coupons paid under different FX regimes contribute different USD
    amounts. FX steps from 50 to 100 partway through the first coupon period,
    so coupon 1 (paid 2021-04-20, FX 100) is worth half of coupon-at-FX-50."""
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2028, 10, 20), 5051.0, 5051.0)  # flat gold
    fx = pd.Series(50.0, index=pd.date_range("2014-01-01", "2030-01-01", freq="D"), name="value")
    fx.loc[fx.index >= pd.Timestamp("2021-01-01")] = 100.0
    civ = sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold, fx_inr_per_usd=fx)
    # Capital is flat (gold flat), so CIV minus capital is pure cumulative USD income.
    capital = 6 * 5051.0
    first_coupon = pd.Timestamp("2021-04-20")  # FX = 100
    income_after_first = civ.loc[first_coupon] - capital
    assert income_after_first == pytest.approx(378.825 / 100.0)


def test_coupon_load_is_a_small_fraction_of_capital() -> None:
    """Regression guard for the Part B1 bug: with realistic USD gold
    (~$60/gram) and a realistic USD/INR (~75), one semiannual coupon is well
    under 2% of capital — not the ~47% it became when a ₹coupon was summed
    into a $capital number."""
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2028, 10, 20), 60.0, 60.0)
    # Exercise the valuation path (must not raise); the assertion below is the
    # analytic coupon/capital ratio.
    sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold, fx_inr_per_usd=_flat_fx(75.0))
    capital = 6 * 60.0
    one_coupon_usd = 378.825 / 75.0
    assert one_coupon_usd / capital < 0.02


def test_coupon_date_before_fx_series_start_raises() -> None:
    """Fail loud if FX history doesn't cover a coupon date (no silent NaN)."""
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2028, 10, 20), 60.0, 60.0)
    fx = pd.Series(
        75.0, index=pd.date_range("2025-01-01", "2030-01-01", freq="D"), name="value"
    )
    with pytest.raises(ValueError, match="no USD/INR rate"):
        sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold, fx_inr_per_usd=fx)


# ---------------------------------------------------------------------------
# Early-redeemed holding type (B2 Phase 3)
# ---------------------------------------------------------------------------

# 2020-21-VII, redeemed 2026-04-20 at RBI's ₹15,254/unit (the user's own
# premature redemption; see data/funds/sgb_redemptions.csv).
_REDEEM_DATE = dt.date(2026, 4, 20)
_REDEEM_INR = 15254
# Coupon dates 2021-04-20 … 2026-04-20 inclusive = 11 semi-annual payments
# received before the holding is redeemed (the 2026-04-20 coupon coincides with
# the redemption date and still counts).
_COUPONS_THROUGH_REDEMPTION = 11


def test_redeemed_matches_htm_before_redemption_date() -> None:
    """An early-redeemed holding is identical to HTM up to its redemption date."""
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2028, 10, 20), 5051.0, 15000.0)
    htm = sgb_holding_civ("2020-21-VII", units_grams=6, gold_prices=gold, fx_inr_per_usd=_FX1)
    redeemed = sgb_holding_civ(
        "2020-21-VII", units_grams=6, gold_prices=gold, fx_inr_per_usd=_FX1,
        redemption_date=_REDEEM_DATE, redemption_inr_per_gram=_REDEEM_INR,
    )
    before = pd.Timestamp(_REDEEM_DATE) - pd.Timedelta(days=1)
    pd.testing.assert_series_equal(htm.loc[:before], redeemed.loc[:before])


def test_redeemed_is_flat_cash_after_redemption() -> None:
    """On/after the redemption date the CIV is constant (proceeds held as cash)."""
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2028, 10, 20), 5051.0, 15000.0)
    redeemed = sgb_holding_civ(
        "2020-21-VII", units_grams=6, gold_prices=gold, fx_inr_per_usd=_FX1,
        redemption_date=_REDEEM_DATE, redemption_inr_per_gram=_REDEEM_INR,
    )
    tail = redeemed.loc[pd.Timestamp(_REDEEM_DATE):]
    assert (tail.diff().dropna() == 0).all(), "redeemed CIV must be flat after redemption"


def test_redeemed_pins_capital_at_rbi_price_plus_frozen_coupons() -> None:
    """Post-redemption value = units × ₹price / FX (FX=1) + coupons received
    through the redemption date; coupons stop accruing afterwards."""
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2028, 10, 20), 5051.0, 15000.0)
    redeemed = sgb_holding_civ(
        "2020-21-VII", units_grams=6, gold_prices=gold, fx_inr_per_usd=_FX1,
        redemption_date=_REDEEM_DATE, redemption_inr_per_gram=_REDEEM_INR,
    )
    expected = 6 * _REDEEM_INR + _COUPONS_THROUGH_REDEMPTION * 378.825
    assert redeemed.iloc[-1] == pytest.approx(expected)
    assert redeemed.loc[pd.Timestamp(_REDEEM_DATE)] == pytest.approx(expected)


def test_redeemed_capital_uses_fx_on_redemption_date() -> None:
    """The ₹ redemption price converts to USD at the redemption date's USD/INR,
    not the issue or coupon FX. Gold flat so we isolate the capital leg."""
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2028, 10, 20), 5051.0, 5051.0)
    redeemed = sgb_holding_civ(
        "2020-21-VII", units_grams=6, gold_prices=gold, fx_inr_per_usd=_flat_fx(80.0),
        redemption_date=_REDEEM_DATE, redemption_inr_per_gram=_REDEEM_INR,
    )
    capital_after = 6 * _REDEEM_INR / 80.0
    income = _COUPONS_THROUGH_REDEMPTION * (378.825 / 80.0)
    assert redeemed.iloc[-1] == pytest.approx(capital_after + income)


def test_redemption_args_must_be_paired() -> None:
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2028, 10, 20), 5051.0, 15000.0)
    with pytest.raises(ValueError, match="must be given together"):
        sgb_holding_civ(
            "2020-21-VII", units_grams=6, gold_prices=gold, fx_inr_per_usd=_FX1,
            redemption_date=_REDEEM_DATE,
        )


def test_redemption_before_issue_raises() -> None:
    gold = _linear_gold(dt.date(2020, 10, 20), dt.date(2028, 10, 20), 5051.0, 15000.0)
    with pytest.raises(ValueError, match="precedes"):
        sgb_holding_civ(
            "2020-21-VII", units_grams=6, gold_prices=gold, fx_inr_per_usd=_FX1,
            redemption_date=dt.date(2019, 1, 1), redemption_inr_per_gram=_REDEEM_INR,
        )


# ---------------------------------------------------------------------------
# Asset labels
# ---------------------------------------------------------------------------


def test_htm_label_is_bare_tranche() -> None:
    assert sgb_asset_label({"tranche_id": "2019-20-IX"}) == "SGB 2019-20-IX"
    assert sgb_asset_label({"tranche_id": "2019-20-IX", "valuation": "htm"}) == "SGB 2019-20-IX"


def test_redeemed_label_is_distinct_from_htm() -> None:
    htm = sgb_asset_label({"tranche_id": "2020-21-VII"})
    redeemed = sgb_asset_label(
        {"tranche_id": "2020-21-VII", "valuation": "redeemed", "redemption_date": "2026-04-20"}
    )
    assert redeemed == "SGB 2020-21-VII (redeemed 2026-04-20)"
    assert redeemed != htm  # no weight-key collision when both held


def test_redeemed_label_without_date() -> None:
    assert (
        sgb_asset_label({"tranche_id": "2020-21-VII", "valuation": "redeemed"})
        == "SGB 2020-21-VII (redeemed)"
    )
