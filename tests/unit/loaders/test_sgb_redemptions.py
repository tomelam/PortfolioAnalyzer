"""Tests for the SGB redemption-price ingest (Phase B2: data + ingest).

The pipeline enumerates RBI redemption press releases (open search endpoint)
and parses each press release for the redemption date, ₹/unit price, and the
tranche(s) it covers. Decomposed like ``loaders/scss`` so the parsers run
offline against saved RBI HTML fixtures:

- ``parse_search_prids(html)``  — search HTML → list of redemption PRIDs
- ``parse_redemption_pr(html)`` — one PR's HTML → list of redemption rows
- ``update_sgb_redemptions(replay_from=...)`` — composition, offline via replay

A live ``network``-marked fetch lives in
``tests/integration/test_data_update_live.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests

from portfolioanalyzer.loaders import sgb_redemptions as R

FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures" / "api_responses"
SEARCH_HTML = (FIXTURES / "rbi_sgb_redemption_search.html").read_text()
PR_OLD = (FIXTURES / "rbi_sgb_redemption_pr_52638.html").read_text()  # "Series I of SGB 2015"
PR_RECENT = (FIXTURES / "rbi_sgb_redemption_pr_62937.html").read_text()  # "SGB 2020-21 Series-III"


# --- parse_search_prids ----------------------------------------------------

def test_search_returns_prids() -> None:
    prids = R.parse_search_prids(SEARCH_HTML)
    assert prids, "expected at least one redemption PRID from the search page"
    assert all(p.isdigit() for p in prids)


def test_search_prids_deduplicated_and_ordered() -> None:
    prids = R.parse_search_prids(SEARCH_HTML)
    assert len(prids) == len(set(prids))  # no duplicates
    assert "62937" in prids  # the recent SGB 2020-21 Series-III redemption


def test_search_ignores_non_redemption_links() -> None:
    html = (
        '<a href="BS_PressReleaseDisplay.aspx?prid=111">Monetary Policy Statement</a>'
        '<a href="BS_PressReleaseDisplay.aspx?prid=222">Redemption of Sovereign Gold Bond</a>'
    )
    assert R.parse_search_prids(html) == ["222"]


# --- parse_redemption_pr ---------------------------------------------------

def test_parse_recent_pr() -> None:
    rows = R.parse_redemption_pr(PR_RECENT)
    assert rows == [
        {
            "tranche_id": "2020-21-III",
            "redemption_date": "2026-06-16",
            "kind": "PRE",
            "inr_per_gram": 14774,
        }
    ]


def test_parse_old_pr_inaugural_year_normalized() -> None:
    """The 2015 inaugural phrasing ("Series I of SGB 2015") normalizes the bare
    year to the 2015-16 fiscal-year tranche-id form, and strips the ₹ comma."""
    rows = R.parse_redemption_pr(PR_OLD)
    assert rows == [
        {
            "tranche_id": "2015-16-I",
            "redemption_date": "2021-11-30",
            "kind": "PRE",
            "inr_per_gram": 4808,
        }
    ]


def test_parse_non_redemption_page_returns_empty() -> None:
    assert R.parse_redemption_pr("<html><body>not a redemption notice</body></html>") == []


def test_extract_tranche_ids_handles_multi_tranche() -> None:
    """Both RBI phrasings, including a release naming two tranches that share a
    redemption date (one gold-based price applies to all)."""
    text = "premature redemption of Series VII of SGB 2017-18 and Series III of SGB 2018-19"
    assert R._extract_tranche_ids(text) == ["2017-18-VII", "2018-19-III"]


def test_normalize_fiscal_year() -> None:
    assert R._normalize_fiscal_year("2020-21") == "2020-21"
    assert R._normalize_fiscal_year("2015") == "2015-16"
    assert R._normalize_fiscal_year("2009") == "2009-10"


# --- merge_redemptions -----------------------------------------------------

def test_merge_upserts_on_key_and_sorts() -> None:
    existing = pd.DataFrame(
        [
            {"tranche_id": "2020-21-III", "redemption_date": "2026-06-16", "kind": "PRE",
             "inr_per_gram": "1", "source_prid_or_url": "old", "tier": "T2"},
            {"tranche_id": "2019-20-I", "redemption_date": "2026-06-11", "kind": "PRE",
             "inr_per_gram": "15038", "source_prid_or_url": "seed", "tier": "T1"},
        ],
        columns=R.COLUMNS,
    )
    incoming = [
        {"tranche_id": "2020-21-III", "redemption_date": "2026-06-16", "kind": "PRE",
         "inr_per_gram": 14774, "source_prid_or_url": "new", "tier": "T1"},
    ]
    out = R.merge_redemptions(existing, incoming)
    # incoming supersedes the existing (tranche_id, date) row...
    row = out[out["tranche_id"] == "2020-21-III"].iloc[0]
    assert int(row["inr_per_gram"]) == 14774
    assert row["source_prid_or_url"] == "new"
    # ...the untouched seeded row is preserved...
    assert (out["tranche_id"] == "2019-20-I").any()
    # ...and the result is sorted by redemption_date.
    assert out["redemption_date"].is_monotonic_increasing


# --- update_sgb_redemptions (composition, offline replay) ------------------

def test_update_replay_writes_csv(tmp_path, monkeypatch) -> None:
    # Don't touch the real stamp file.
    monkeypatch.setattr(R.data_update, "_update_stamp", lambda *a, **k: None)
    out_csv = tmp_path / "sgb_redemptions.csv"
    res = R.update_sgb_redemptions(path=str(out_csv), replay_from=str(FIXTURES))

    # The fixtures cover two of the search PRIDs (52638 not on this search page,
    # 62937 is); missing PR fixtures are skipped, not fatal.
    assert res["prids"] >= 1
    df = pd.read_csv(out_csv)
    assert list(df.columns) == R.COLUMNS
    assert (df["tranche_id"] == "2020-21-III").any()
    assert df["tier"].eq("T1").all()


def test_update_network_path_mocked(tmp_path, monkeypatch) -> None:
    """The default (non-replay) path drives the network fetchers; mock them."""
    monkeypatch.setattr(R.data_update, "_update_stamp", lambda *a, **k: None)
    monkeypatch.setattr(R, "fetch_search_html", lambda **k: SEARCH_HTML)

    def fake_pr(prid, **k):
        return PR_RECENT if prid == "62937" else "<html>no rows</html>"

    monkeypatch.setattr(R, "fetch_pr_html", fake_pr)
    out_csv = tmp_path / "r.csv"
    res = R.update_sgb_redemptions(path=str(out_csv))
    assert res["rows"] == 1  # only 62937 parsed a row
    assert res["total"] == 1


def test_fetch_propagates_network_error(monkeypatch) -> None:
    def boom(url, **kwargs):
        raise requests.ConnectionError("network unreachable")

    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(requests.ConnectionError):
        R.fetch_search_html()


def test_fetch_pr_uses_prid(monkeypatch) -> None:
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        resp = MagicMock(spec=requests.Response)
        resp.text = PR_RECENT
        resp.raise_for_status.return_value = None
        return resp

    monkeypatch.setattr(requests, "get", fake_get)
    R.fetch_pr_html("62937")
    assert "prid=62937" in captured["url"]


# --- lookup_redemption (valuation-layer accessor) --------------------------

def _seed_csv(tmp_path, rows) -> str:
    df = pd.DataFrame(rows, columns=R.COLUMNS)
    out = tmp_path / "r.csv"
    R.write_redemptions(df, str(out))
    return str(out)


def test_lookup_redemption_single_row(tmp_path) -> None:
    csv = _seed_csv(tmp_path, [
        {"tranche_id": "2020-21-VII", "redemption_date": "2026-04-20", "kind": "PRE",
         "inr_per_gram": "15254", "source_prid_or_url": "u", "tier": "T1"},
    ])
    row = R.lookup_redemption("2020-21-VII", path=csv)
    assert row["inr_per_gram"] == 15254  # coerced to int
    assert row["redemption_date"] == "2026-04-20"


def test_lookup_redemption_missing_raises(tmp_path) -> None:
    csv = _seed_csv(tmp_path, [
        {"tranche_id": "2020-21-VII", "redemption_date": "2026-04-20", "kind": "PRE",
         "inr_per_gram": "15254", "source_prid_or_url": "u", "tier": "T1"},
    ])
    with pytest.raises(ValueError, match="no SGB redemption price"):
        R.lookup_redemption("2019-20-IX", path=csv)


def test_lookup_redemption_ambiguous_without_date_raises(tmp_path) -> None:
    csv = _seed_csv(tmp_path, [
        {"tranche_id": "2020-21-VII", "redemption_date": "2026-04-20", "kind": "PRE",
         "inr_per_gram": "15254", "source_prid_or_url": "u", "tier": "T1"},
        {"tranche_id": "2020-21-VII", "redemption_date": "2026-10-20", "kind": "PRE",
         "inr_per_gram": "15999", "source_prid_or_url": "u", "tier": "T1"},
    ])
    with pytest.raises(ValueError, match="redemption rows"):
        R.lookup_redemption("2020-21-VII", path=csv)
    # ...but disambiguating by date succeeds.
    row = R.lookup_redemption("2020-21-VII", "2026-10-20", path=csv)
    assert row["inr_per_gram"] == 15999


# --- committed seed CSV conforms to the loader's schema --------------------

def test_committed_seed_csv_matches_schema() -> None:
    """The checked-in seed must stay readable by the loader: exact column set,
    ISO dates, integer ₹ prices, and known kind values (guards the data file
    from drifting away from ``COLUMNS``)."""
    df = R.read_redemptions()  # the real data/funds/sgb_redemptions.csv
    assert list(df.columns) == R.COLUMNS
    assert not df.empty, "seed CSV should carry the live-seeded RBI redemptions"
    assert df["kind"].isin({"PRE", "MAT"}).all()
    assert df["inr_per_gram"].astype(int).gt(0).all()
    assert pd.to_datetime(df["redemption_date"], format="%Y-%m-%d", errors="raise").notna().all()
