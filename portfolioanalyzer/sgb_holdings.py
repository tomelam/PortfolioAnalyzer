"""Per-tranche SGB valuation engine.

Replaced the bogus ``pct_change``-on-consecutive-tranche-issue-prices
logic in the now-deleted ``sgb_loader.py`` with a proper holder-side
mark-to-market:

    CIV(t) = units_grams × gold_price_per_gram(t)        # capital, USD
           + Σ(coupon_inr / fx_at_coupon_date)            # income, USD

The whole series is in **USD**: the capital leg is USD/gram LBMA gold; the
contractual *rupee* coupons are converted to USD at each payment date's USD/INR
rate (FRED ``DEXINUS``). FX is applied only to the discrete coupon cash amounts,
never to the gold price path.

This module is the pure-function valuation layer. ``main.py`` iterates
the portfolio's ``[[sgb]]`` entries and calls ``sgb_holding_civ`` once
per tranche.

Coupon rules (from the user's RBI certificates and the master ledger):

- **Rate:** 2.5% per annum on the *nominal* (offline) issue price.
- **Frequency:** semi-annual (six-monthly).
- **First payment:** 6 months after the issue date (not on issue).
- **Final payment:** on the maturity date itself, alongside principal.
- **Total payments over an 8-year tenor:** 16.

Note on the 2015 Series I outlier: that single tranche had a 2.75%
coupon. It's not modeled here because it predates the user's window
(Feb 2020+) and isn't in ``data/funds/sgb_tranches.csv``.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from dateutil.relativedelta import relativedelta

from portfolioanalyzer.sgb_tranches import lookup_tranche

_MONTHS_BETWEEN_COUPONS = 6


def coupon_schedule(issue_date: dt.date, maturity_date: dt.date) -> list[dt.date]:
    """Return the ordered list of coupon-payment dates for one tranche.

    Standard SGB schedule: payments at issue + 6mo, +12mo, +18mo, ...,
    up to and including the maturity date. Uses ``relativedelta`` so
    month-end rollovers behave correctly (e.g. 31 Mar → 30 Sep).
    """
    out: list[dt.date] = []
    n = 1
    while True:
        next_date = issue_date + relativedelta(months=_MONTHS_BETWEEN_COUPONS * n)
        if next_date > maturity_date:
            break
        out.append(next_date)
        n += 1
    return out


def coupon_amount_per_payment(
    units_grams: float,
    issue_price_offline: float,
    coupon_rate_pct: float,
) -> float:
    """Per-coupon amount in INR.

    ``(units_grams × issue_price_offline)`` is the *nominal* value of
    the holding. ``coupon_rate_pct / 100`` is the annual coupon rate;
    dividing by 2 gives the semi-annual amount.
    """
    nominal = units_grams * issue_price_offline
    return nominal * (coupon_rate_pct / 100.0) / 2.0


def sgb_holding_civ(
    tranche_id: str,
    units_grams: float,
    gold_prices: pd.Series,
    fx_inr_per_usd: pd.Series,
) -> pd.Series:
    """Daily CIV series for one SGB-tranche holding, **entirely in USD**.

    An SGB is an INR instrument, but the tool reports only normalized returns
    and has no free deep-history daily INR gold series, so the holding is marked
    in USD off the LBMA gold price. The capital leg is ``units_grams ×``
    USD/gram gold. The 2.5% coupon is contractually a *rupee* cash amount; it is
    converted to USD at **its own payment date's** USD/INR rate so the CIV is a
    single consistent USD series. FX touches only the discrete coupon cash
    amounts, never the gold price path (the sanctioned "contractual cash value"
    FX exception).

    Args:
        tranche_id: Composite tranche key, e.g. ``"2020-21-VII"``. Looked
            up via ``sgb_tranches.lookup_tranche``.
        units_grams: Number of grams held (lumped per the holdings rule:
            multiple bank-application certs for the same tranche on the
            same issue date collapse into one ``units_grams``).
        gold_prices: Gold price per gram in **USD**, datetime-indexed. May
            extend before issue or past maturity; the returned series is
            clipped to ``[issue_date, min(maturity_date, gold_end)]``.
        fx_inr_per_usd: USD/INR exchange rate (Indian Rupees per US Dollar,
            FRED ``DEXINUS``), datetime-indexed. Read as-of each coupon date
            (ffilled over weekends/holidays) to convert the rupee coupon to USD.

    Returns:
        ``pd.Series`` of CIV values, daily-indexed, name ``"value"``.
    """
    t = lookup_tranche(tranche_id)
    issue_date: dt.date = t["issue_date"]
    maturity_date: dt.date = t["maturity_date"]
    coupon_rate_pct: float = float(t["coupon_rate_pct"])
    issue_price_offline: float = float(t["issue_price_offline"])

    # Clip the window to [issue, min(maturity, gold_end)].
    gold_end = gold_prices.index.max().date()
    end_date = min(maturity_date, gold_end)
    if end_date < issue_date:
        raise ValueError(
            f"tranche {tranche_id} issue date {issue_date} is past the end "
            f"of the gold price series ({gold_end})"
        )

    # Daily gold prices over the holding period, ffilled to fill any gaps
    # (the source CSV is daily but skips weekends/holidays; ffill is the
    # standard market convention for stale-quote roll-forward).
    daily_idx = pd.date_range(issue_date, end_date, freq="D")
    gold_window = gold_prices.reindex(daily_idx, method="ffill")

    capital = units_grams * gold_window

    # Cumulative coupons received as of each day, in USD. Each rupee coupon is
    # converted at its payment date's USD/INR rate (FX ffilled over non-trading
    # days via Series.asof), so coupons paid in different FX regimes contribute
    # different USD amounts — this is why we accumulate per-coupon rather than
    # multiplying a flat amount by a running count.
    schedule = coupon_schedule(issue_date, maturity_date)
    per_coupon_inr = coupon_amount_per_payment(
        units_grams=units_grams,
        issue_price_offline=issue_price_offline,
        coupon_rate_pct=coupon_rate_pct,
    )
    fx = fx_inr_per_usd.sort_index()
    income = pd.Series(0.0, index=daily_idx)
    for d in schedule:
        ts = pd.Timestamp(d)
        rate = fx.asof(ts)  # last USD/INR on or before the coupon date
        if pd.isna(rate):
            raise ValueError(
                f"tranche {tranche_id}: no USD/INR rate on or before coupon "
                f"date {d}; FX series starts {fx.index.min().date()}"
            )
        per_coupon_usd = per_coupon_inr / rate
        income.loc[daily_idx >= ts] += per_coupon_usd

    civ = (capital + income).rename("value")
    return civ
