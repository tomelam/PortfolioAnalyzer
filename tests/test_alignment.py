"""Behavioral test for ``data_loader.align_portfolio_civs``.

The legacy version loaded two pickled goldens (``portfolio_civs.pkl`` and
``aligned_civs.pkl``) captured from a live mfapi run and compared the
function's output to the second pickle. That made the test (a)
pickle-dependent and (b) brittle to pandas version drift.

This rewrite uses deterministic synthetic NAV DataFrames so the *contract*
of the function is exercised without touching any binary fixture:

- The result is indexed by the intersection of all funds' date ranges.
- The columns are a MultiIndex of (fund_name, original_col_name).
- Ffill is applied (no NaN remains after alignment).
"""

from __future__ import annotations

import pandas as pd
import pytest

from data_loader import align_portfolio_civs


def _nav_df(dates: list[str], values: list[float]) -> pd.DataFrame:
    """One fund's NAV history as a DataFrame indexed by Timestamp."""
    return pd.DataFrame(
        {"nav": values},
        index=pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="date"),
    )


@pytest.mark.order(6)
def test_align_portfolio_civs_indexes_by_intersection() -> None:
    """Two funds with partially overlapping date ranges → aligned result
    is indexed by the intersection of those ranges."""
    fund_a = _nav_df(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"],
        [100.0, 101.0, 102.0, 103.0, 104.0],
    )
    fund_b = _nav_df(
        ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"],
        [200.0, 201.0, 202.0, 203.0, 204.0],
    )

    result = align_portfolio_civs({"Fund A": fund_a, "Fund B": fund_b})

    expected = pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08"])
    assert list(result.index) == list(expected)
    assert not result.isna().any().any()


def test_align_portfolio_civs_uses_multiindex_columns() -> None:
    """Downstream code inspects the column MultiIndex; pin the shape."""
    fund = _nav_df(["2024-01-02", "2024-01-03"], [100.0, 101.0])
    result = align_portfolio_civs({"Fund A": fund})

    assert isinstance(result.columns, pd.MultiIndex)
    assert ("Fund A", "nav") in result.columns
