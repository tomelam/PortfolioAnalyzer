"""Live network checks for the data auto-update fetchers.

`network`-marked (deselected by default). FRED is reachable and verified;
niftyindices is anti-scraping / commonly unreachable from non-browser
contexts, so its live fetch is asserted leniently (reachable-or-skip).
"""

from __future__ import annotations

import pandas as pd
import pytest

from loaders import data_update as du


@pytest.mark.network
@pytest.mark.integration
def test_fetch_fred_indirltlt01stm_live():
    df = du.fetch_fred_series("INDIRLTLT01STM")
    assert not df.empty
    assert list(df.columns) == ["value"]
    assert isinstance(df.index, pd.DatetimeIndex)
    # India 10Y govt bond rate has sat in a sane band for years.
    assert 3.0 < float(df["value"].iloc[-1]) < 15.0
    # Should be far fresher than the bundled stale CSV (2025-03).
    assert df.index.max() > pd.Timestamp("2025-06-01")


@pytest.mark.network
@pytest.mark.integration
def test_fetch_niftyindices_tri_live():
    """niftyindices is frequently blocked from non-browser IPs; treat an
    unreachable host as a skip rather than a failure."""
    try:
        df = du.fetch_niftyindices_tri()
    except Exception as e:  # noqa: BLE001 — network/anti-scrape is expected here
        pytest.skip(f"niftyindices unreachable from this environment: {e}")
    assert not df.empty
    assert list(df.columns) == ["value"]
    assert isinstance(df.index, pd.DatetimeIndex)
