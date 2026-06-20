"""Tests for the SGB tranche reference and lookup API.

Scope per user request (2026-06-16): the codebase needs a clean way to
look up details for SGB tranches purchasable from February 2020 onward.
67 total tranches were issued (Nov 2015 → Feb 2024) but the user only
ever bought from FY 2019-20 Series IX (Feb 2020) onward, so the reference
data file is scoped to that window.

Anchor values for FY 2020-21 Series VII and FY 2019-20 Series IX come
from the user's actual RBI SGB certificates (HDFC + SBI) and are used
as ground-truth assertions in the tests below.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from portfolioanalyzer.sgb_tranches import (
    PURCHASABLE_FROM,
    describe_tranche,
    list_tranches,
    load_tranches,
    lookup_tranche,
    tranche_status,
)


def test_load_tranches_returns_dataframe_with_expected_columns() -> None:
    df = load_tranches()
    assert isinstance(df, pd.DataFrame)
    expected = {
        "tranche_id",
        "fiscal_year",
        "series",
        "series_num",
        "issue_date",
        "maturity_date",
        "premature_first_date",
        "issue_price_offline",
        "issue_price_online",
        "coupon_rate_pct",
        "rbi_prid",
    }
    missing = expected - set(df.columns)
    assert not missing, f"missing columns: {missing}"


def test_only_tranches_from_feb_2020_onward() -> None:
    """User: 'I'm not interested in anything purchasable before February 2020.'"""
    df = load_tranches()
    assert (df["issue_date"] >= PURCHASABLE_FROM).all(), (
        f"some tranches predate {PURCHASABLE_FROM}: "
        f"{df[df['issue_date'] < PURCHASABLE_FROM]['tranche_id'].tolist()}"
    )
    # Spot-check: earliest tranche is FY 2019-20 Series IX (Feb 2020).
    earliest = df.sort_values("issue_date").iloc[0]
    assert earliest["tranche_id"] == "2019-20-IX"


def test_lookup_2020_21_VII_matches_user_hdfc_certificate() -> None:
    """User holds 6 grams of this tranche (one investment line, lumped from
    two HDFC certificates — HDF2020004107231 + HDF2020004107232, 3 g each,
    both issued 20-Oct-2020 under the same scheme). Lumping rule:
    same (tranche_id, issue_date, holder_pool) → one investment.
    Certificate values are the ground truth for these fields."""
    t = lookup_tranche("2020-21-VII")
    assert t["issue_date"] == dt.date(2020, 10, 20)
    assert t["maturity_date"] == dt.date(2028, 10, 20)
    assert t["issue_price_offline"] == 5051
    assert t["coupon_rate_pct"] == pytest.approx(2.5)


def test_lookup_2019_20_IX_matches_user_sbi_certificate() -> None:
    """User holds 12 grams of this tranche (one investment line; single SBI
    certificate SBI2020000155331 issued 11-Feb-2020). Distinct from
    2020-21-VII — different tranche, different investment by design.
    Certificate values are the ground truth for these fields."""
    t = lookup_tranche("2019-20-IX")
    assert t["issue_date"] == dt.date(2020, 2, 11)
    assert t["maturity_date"] == dt.date(2028, 2, 11)
    assert t["issue_price_offline"] == 4070
    assert t["coupon_rate_pct"] == pytest.approx(2.5)


def test_lookup_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unknown tranche"):
        lookup_tranche("nonexistent-tranche")


def test_premature_first_date_is_issue_plus_5_years() -> None:
    t = lookup_tranche("2020-21-VII")
    assert t["premature_first_date"] == dt.date(2025, 10, 20)


def test_tranche_status_lock_before_5y_anniversary() -> None:
    # FY 2023-24 Series IV issued 2024-02-21; in lock until 2029-02-21.
    assert tranche_status("2023-24-IV", as_of=dt.date(2026, 6, 16)) == "LOCK"


def test_tranche_status_pre_between_5_and_8_years() -> None:
    # FY 2020-21 Series VII issued 2020-10-20; premature window opened
    # 2025-10-20; matures 2028-10-20.
    assert tranche_status("2020-21-VII", as_of=dt.date(2026, 6, 16)) == "PRE"


def test_tranche_status_mat_after_maturity() -> None:
    # FY 2019-20 Series IX matures 2028-02-11.
    assert tranche_status("2019-20-IX", as_of=dt.date(2028, 6, 1)) == "MAT"


def test_list_tranches_filters_by_fiscal_year() -> None:
    fy_2023_24 = list_tranches(fiscal_year="2023-24")
    assert len(fy_2023_24) == 4
    series = sorted(fy_2023_24["series"].tolist())
    assert series == ["I", "II", "III", "IV"]


def test_list_tranches_filters_by_status() -> None:
    """As of 2026-06-16, all FY 2023-24 tranches should still be in LOCK."""
    locked = list_tranches(status="LOCK", as_of=dt.date(2026, 6, 16))
    assert "2023-24-IV" in set(locked["tranche_id"])
    # And the earliest tranche (Feb 2020) shouldn't be locked anymore.
    assert "2019-20-IX" not in set(locked["tranche_id"])


def test_list_tranches_status_filter_requires_as_of() -> None:
    with pytest.raises(ValueError, match="status filter requires as_of"):
        list_tranches(status="LOCK")  # as_of omitted


def test_describe_tranche_renders_key_fields() -> None:
    """The human-readable summary surfaces the anchor tranche's core facts and
    its status as of a given date."""
    text = describe_tranche("2019-20-IX", as_of=dt.date(2026, 6, 16))
    assert "Tranche 2019-20-IX" in text
    assert "Series IX" in text
    assert "₹4,070" in text  # offline issue price
    assert "2.50% p.a." in text  # coupon
    assert "as of 2026-06-16" in text
    # Premature window opened 2025-02-11, before the as_of → eligible for redemption.
    assert "Status:   PRE" in text


def test_describe_tranche_defaults_as_of_to_today() -> None:
    text = describe_tranche("2019-20-IX")
    assert text.startswith("Tranche 2019-20-IX")
