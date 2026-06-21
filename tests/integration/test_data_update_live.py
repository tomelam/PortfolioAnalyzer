"""Live network checks for the data auto-update fetchers.

`network`-marked (deselected by default). FRED is reachable and verified;
niftyindices is anti-scraping and fetched via a stealth browser (optional
``browser`` extra), so its live fetch is skipped when the extra is missing and
asserted leniently when the host throttles this IP.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from portfolioanalyzer.loaders import data_update as du


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
def test_fetch_fred_dexinus_live():
    """USD/INR (FRED DEXINUS) — the SGB cash-flow FX source. Same clean FRED
    CSV path as the risk-free series."""
    df = du.fetch_fred_series("DEXINUS")
    assert not df.empty
    assert list(df.columns) == ["value"]
    assert isinstance(df.index, pd.DatetimeIndex)
    # Rupees per dollar has sat in a sane band (deep history starts ~8, recent ~85).
    assert 5.0 < float(df["value"].iloc[-1]) < 200.0
    # Deep history back to 1973, fresh to recent days.
    assert df.index.min() < pd.Timestamp("1980-01-01")
    assert df.index.max() > pd.Timestamp("2025-06-01")


@pytest.mark.network
@pytest.mark.integration
def test_fetch_lbma_gold_live():
    df = du.fetch_lbma_gold()
    assert not df.empty
    assert list(df.columns) == ["value"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    # Gold (USD/troy-ounce) has been in the four-figure range for years.
    assert 500.0 < float(df["value"].iloc[-1]) < 100000.0
    # Deep history (the LBMA series runs back to 1968) and fresh to recent days.
    assert df.index.min() < pd.Timestamp("2009-01-01")
    assert df.index.max() > pd.Timestamp("2025-06-01")


@pytest.mark.network
@pytest.mark.integration
def test_fetch_niftyindices_tri_live():
    """Real fetch through the stealth-browser path.

    Skips when the optional ``browser`` extra (playwright + stealth) isn't
    installed. niftyindices aggressively rate-limits an IP after repeated
    hits, so a timeout/connection failure is upstream throttling, not a code
    defect — we skip with a clear reason in that case. Any *other* failure
    (bad envelope, parse error, empty/garbage data) propagates as a real test
    failure, so regressions in the scraper are still caught.
    """
    if not du._playwright_available():
        pytest.skip("browser extra not installed (pip install '.[browser]')")
    try:
        df = du.fetch_niftyindices_tri(start="01-Apr-2026", end="30-Apr-2026")
    except RuntimeError as e:
        pytest.skip(f"niftyindices throttled this IP (upstream, not a bug): {e}")
    # Reached only when the scrape succeeded — assert the data is real & sane.
    assert not df.empty
    assert list(df.columns) == ["value"]
    assert isinstance(df.index, pd.DatetimeIndex)
    # NIFTY 50 TRI has been in the tens of thousands for years.
    assert float(df["value"].iloc[-1]) > 10000


@pytest.mark.network
@pytest.mark.integration
def test_fetch_rbi_sgb_redemptions_live():
    """Real fetch of RBI SGB redemption press releases (plain requests; no
    CAPTCHA — see ``scripts/probe_rbi_sgb_redemption.py``, the B2 Step-0 gate).

    Enumerate via the open search endpoint, then parse the newest press release
    end-to-end: it must yield a sane premature-redemption row (recent ₹/gram in
    a four-figure band, a real tranche id, an ISO date).
    """
    from portfolioanalyzer.loaders import sgb_redemptions as R

    prids = R.parse_search_prids(R.fetch_search_html())
    assert prids, "RBI search returned no SGB redemption PRIDs"
    rows = R.parse_redemption_pr(R.fetch_pr_html(prids[0]))
    assert rows, f"newest redemption PR {prids[0]} parsed no rows"
    row = rows[0]
    assert row["kind"] in {"PRE", "MAT"}
    assert re.fullmatch(r"\d{4}-\d{2}-[IVXLC]+", row["tranche_id"]), row["tranche_id"]
    assert pd.Timestamp(row["redemption_date"]) > pd.Timestamp("2017-01-01")
    # Redemption price (₹/gram of 999 gold) has been four-to-five figures.
    assert 1000 < int(row["inr_per_gram"]) < 100000
