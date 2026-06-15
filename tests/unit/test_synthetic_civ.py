"""Tests for ``synthetic_civ.calculate_ppf_relative_civ``.

The PPF formula: monthly interest accrues but is credited only at
financial-year end (March). The CIV reflects accrued-but-uncredited
interest each month.
"""

from __future__ import annotations

import pandas as pd
import pytest

from synthetic_civ import calculate_ppf_relative_civ


def _rates(rates_by_date: dict[str, float]) -> pd.DataFrame:
    idx = pd.to_datetime(list(rates_by_date.keys()))
    return pd.DataFrame({"rate": list(rates_by_date.values())}, index=idx)


def test_returns_dataframe_with_civ_column() -> None:
    df = calculate_ppf_relative_civ(_rates({"2024-04-01": 7.1, "2024-05-01": 7.1}))
    assert isinstance(df, pd.DataFrame)
    assert "civ" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)


def test_starting_civ_is_one_thousand() -> None:
    df = calculate_ppf_relative_civ(_rates({"2024-04-01": 7.1, "2024-05-01": 7.1}))
    # Function seeds with yearly_civ = 1000.0 at the first synthesized point.
    assert df["civ"].iloc[0] == 1000.0


def test_monthly_accrual_at_constant_rate() -> None:
    """At a constant 12% annual rate, monthly interest is 1% of yearly_civ."""
    rates = _rates({f"2024-{m:02d}-01": 12.0 for m in range(4, 10)})  # Apr–Sep
    df = calculate_ppf_relative_civ(rates)
    civs = df["civ"].tolist()

    # First entry is the seed (1000). Each subsequent month adds 1% of 1000 = 10
    # to accrued interest. Crediting happens only in March (no March in this window),
    # so yearly_civ stays at 1000 and accrued grows linearly.
    assert civs[0] == pytest.approx(1000.0)
    # After 5 monthly accruals each month gets +10 more, monthly_civ = 1000 + 10*k.
    expected = [1000.0 + 10.0 * k for k in range(len(civs))]
    for actual, exp in zip(civs, expected):
        assert actual == pytest.approx(exp, rel=1e-9)


def test_yearly_credit_in_march_compounds_balance() -> None:
    """At financial-year-end (March), accrued interest is credited and
    becomes part of the principal for next year's accrual."""
    # Cover Apr 2024 → Apr 2025 (13 months) at 12% constant.
    rates = _rates({f"2024-{m:02d}-01": 12.0 for m in range(4, 13)})
    rates_2025 = _rates({f"2025-{m:02d}-01": 12.0 for m in range(1, 5)})
    rates = pd.concat([rates, rates_2025])

    df = calculate_ppf_relative_civ(rates)
    civs = df["civ"].tolist()

    # After 12 monthly accruals at 1% on principal 1000 → 120 interest by
    # the *next* March (2025-03-31). March CIV: 1000 + 120 = 1120.
    # April CIV (post-credit): 1120 + 1% of 1120 = 1131.2.
    march_2025 = df.index[df.index == pd.Timestamp("2025-03-31")]
    assert len(march_2025) == 1, "expected one March-2025 row"
    march_pos = df.index.get_loc(march_2025[0])
    assert civs[march_pos] == pytest.approx(1120.0, rel=1e-9)
    if march_pos + 1 < len(civs):
        assert civs[march_pos + 1] == pytest.approx(1131.2, rel=1e-9)


def test_step_rate_change_takes_effect_in_following_month() -> None:
    """When the rate changes mid-window, the new rate applies to subsequent
    monthly accruals."""
    rates = _rates({
        "2024-04-01": 12.0,  # 1%/month
        "2024-05-01": 12.0,
        "2024-06-01": 24.0,  # 2%/month after this point
        "2024-07-01": 24.0,
    })
    df = calculate_ppf_relative_civ(rates)
    civs = df["civ"].tolist()

    # Indices: [0] seed, [1] after 1 month @1%, [2] after 2 months @1%,
    # [3] after 3 months — but the rate read at i=3 is the rate AT that date,
    # which is 24% (June rate carried forward via .reindex pad).
    # Note: the loop reads rates from `monthly_rates` (which uses 'pad'), so
    # June onwards reads 24%, contributing +20 to accrued.
    # accrued: 0, 10, 20, 40, 60 → civ: 1000, 1010, 1020, 1040, 1060
    expected = [1000.0, 1010.0, 1020.0, 1040.0, 1060.0]
    for actual, exp in zip(civs[: len(expected)], expected):
        assert actual == pytest.approx(exp, rel=1e-9)


def test_accepts_series_input_not_just_dataframe() -> None:
    """The function accepts either a DataFrame (extracts 'rate') or a Series."""
    rates_series = pd.Series(
        [7.1, 7.1],
        index=pd.to_datetime(["2024-04-01", "2024-05-01"]),
        name="rate",
    )
    df = calculate_ppf_relative_civ(rates_series)
    assert isinstance(df, pd.DataFrame)
    assert df["civ"].iloc[0] == 1000.0
