"""Unit tests for ``bond_calculators`` — the SCSS valuation engine.

Covers the term-locked rollover model (``term_locked_rate_series``) and the
``calculate_variable_bond_cumulative_gain`` wrapper. SCSS locks its rate at
account opening for the full term; the rate is re-looked-up only at each
rollover boundary, not floated on every quarterly revision. The generic
``term_years=None`` path (continuous re-pricing) is preserved for back-compat.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from portfolioanalyzer.bond_calculators import (
    calculate_variable_bond_cumulative_gain,
    term_locked_rate_series,
)


def _rates(pairs) -> pd.Series:
    """Build a date->annual-percent rate Series from (iso_date, rate) pairs."""
    idx = pd.to_datetime([p[0] for p in pairs])
    return pd.Series([p[1] for p in pairs], index=idx, name="interest")


def _applicable(series: pd.Series, iso: str) -> float:
    """The locked annual rate in effect on the business day on/after `iso`."""
    return float(series.asof(pd.Timestamp(iso)))


# --- term_locked_rate_series ------------------------------------------------

def test_constant_rate_is_flat_across_all_terms():
    rate = _rates([("2010-01-01", 8.0)])
    s = term_locked_rate_series(rate, "2010-01-01", "2025-01-01", term_years=5)
    assert (s == 8.0).all()


def test_mid_term_rate_change_does_not_affect_current_term():
    # Rate drops mid-way through the first 5y term; the open account keeps its
    # opening rate until the first rollover (2015), then re-prices.
    rate = _rates([("2010-01-01", 8.0), ("2012-01-01", 9.0)])
    s = term_locked_rate_series(rate, "2010-01-01", "2016-06-01", term_years=5)
    assert _applicable(s, "2013-06-03") == 8.0  # mid-term change ignored
    assert _applicable(s, "2015-06-01") == 9.0  # picked up at the 2015 rollover


def test_rate_change_exactly_on_boundary_is_picked_up():
    rate = _rates([("2010-01-01", 8.0), ("2015-01-01", 9.5)])
    s = term_locked_rate_series(rate, "2010-01-01", "2018-01-01", term_years=5)
    assert _applicable(s, "2014-12-31") == 8.0
    assert _applicable(s, "2015-01-02") == 9.5


def test_anchor_before_rate_history_uses_earliest_rate():
    rate = _rates([("2012-01-01", 7.5)])
    s = term_locked_rate_series(rate, "2010-01-01", "2014-01-01", term_years=5)
    assert _applicable(s, "2010-06-01") == 7.5


def test_multiple_rollovers_lock_per_term():
    rate = _rates(
        [
            ("2000-01-01", 8.0),
            ("2005-06-01", 7.0),  # falls AFTER the 2005-01 boundary
            ("2010-06-01", 9.0),
            ("2015-06-01", 8.5),
        ]
    )
    s = term_locked_rate_series(rate, "2000-01-01", "2021-01-01", term_years=5)
    # Boundary lookups are as-of 1 Jan; a June change is seen one term later.
    assert _applicable(s, "2002-01-01") == 8.0  # [2000,2005)
    assert _applicable(s, "2007-01-01") == 8.0  # [2005,2010): 2005-06 not yet at 2005-01
    assert _applicable(s, "2012-01-01") == 7.0  # [2010,2015): 2005-06 drop seen at 2010
    assert _applicable(s, "2017-01-01") == 9.0  # [2015,2020): 2010-06 rise seen at 2015
    assert _applicable(s, "2020-06-01") == 8.5  # [2020,..): 2015-06 seen at 2020


def test_supports_native_date_anchor():
    rate = _rates([("2010-01-01", 8.0)])
    s = term_locked_rate_series(rate, dt.date(2010, 1, 1), "2020-01-01", term_years=5)
    assert not s.empty and (s == 8.0).all()


# --- calculate_variable_bond_cumulative_gain --------------------------------

def _rate_df(pairs) -> pd.DataFrame:
    return _rates(pairs).to_frame("interest")


def test_backcompat_continuous_path_unchanged_for_constant_rate():
    # term_years=None must keep the legacy continuous behaviour. For a constant
    # rate the CIV is (1+r/100)^(business_days/252); assert the closed form.
    rate_df = _rate_df([("2010-01-01", 8.0)])
    civ = calculate_variable_bond_cumulative_gain(rate_df, "2015-01-01")
    n = len(civ)
    expected_last = (1 + 8.0 / 100) ** (n / 252)
    assert civ.iloc[-1] == pytest.approx(expected_last, rel=1e-6)


def test_constant_rate_identical_under_lock_and_continuous():
    rate_df = _rate_df([("2010-01-01", 8.0)])
    cont = calculate_variable_bond_cumulative_gain(rate_df, "2015-01-01")
    lock = calculate_variable_bond_cumulative_gain(
        rate_df, "2015-01-01", term_years=5, anchor_date="2015-01-01"
    )
    pd.testing.assert_series_equal(cont, lock)


def test_term_lock_differs_from_continuous_when_rate_varies_mid_term():
    # A change mid-term moves the continuous CIV immediately but not the locked
    # one (it waits for the rollover) -> the two series must diverge.
    rate_df = _rate_df([("2008-01-01", 8.0), ("2018-01-01", 6.0)])
    cont = calculate_variable_bond_cumulative_gain(rate_df, "2010-01-01")
    lock = calculate_variable_bond_cumulative_gain(
        rate_df, "2010-01-01", term_years=5, anchor_date="2010-01-01"
    )
    # The 2018 drop lands inside the [2015,2020) locked term, so within that
    # term the locked series still compounds at 8% while continuous drops to 6%.
    common = cont.index.intersection(lock.index)
    assert not cont.loc[common].equals(lock.loc[common])


def test_term_lock_is_monotonic_increasing():
    rate_df = _rate_df([("2008-01-01", 8.0), ("2018-01-01", 6.0)])
    lock = calculate_variable_bond_cumulative_gain(
        rate_df, "2010-01-01", term_years=5, anchor_date="2010-01-01"
    )
    assert (lock.diff().dropna() >= 0).all()
