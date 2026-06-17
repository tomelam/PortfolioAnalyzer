"""Live network checks for the data auto-update fetchers.

`network`-marked (deselected by default). FRED is reachable and verified;
niftyindices is anti-scraping / commonly unreachable from non-browser
contexts, so its live fetch is asserted leniently (reachable-or-skip).
"""

from __future__ import annotations

import pandas as pd
import pytest
import requests

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
    """Real fetch through the cookie-primed JSON API (proven working).

    niftyindices aggressively rate-limits an IP after repeated hits, so a
    timeout/connection failure *after the client's own retries* is upstream
    throttling, not a code defect — we skip with a clear reason in that case.
    Any *other* failure (bad envelope, parse error, empty/garbage data)
    propagates as a real test failure, so regressions in the scraper are
    still caught.
    """
    try:
        df = du.fetch_niftyindices_tri(start="01-Apr-2026", end="30-Apr-2026")
    except (RuntimeError, requests.exceptions.RequestException) as e:
        pytest.skip(f"niftyindices throttled this IP after retries (upstream, not a bug): {e}")
    # Reached only when the scrape succeeded — assert the data is real & sane.
    assert not df.empty
    assert list(df.columns) == ["value"]
    assert isinstance(df.index, pd.DatetimeIndex)
    # NIFTY 50 TRI has been in the tens of thousands for years.
    assert float(df["value"].iloc[-1]) > 10000
