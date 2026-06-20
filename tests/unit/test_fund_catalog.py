"""Integrity checks for data/fund_catalog.csv.

The catalog maps popular mutual funds (across categories) to their stated
benchmark and whether that benchmark has a free data source — the input for
adding per-fund α/β parity cases and building test portfolios. These offline
checks keep it internally consistent and in sync with data/vro_funds.csv.

Fetchability of the `niftyindices_name` spellings was confirmed live via
`scripts/probe_niftyindices_catalog_names.py` (kept as documented diagnostics);
we deliberately do NOT re-fetch niftyindices from the test suite (it is an
anti-scrape host contacted at most once/day by the app).
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data" / "fund_catalog.csv"
VRO_FUNDS = ROOT / "data" / "vro_funds.csv"

REQUIRED_COLUMNS = {
    "category", "fund_name", "mfapi_code", "isin", "stated_benchmark",
    "niftyindices_name", "benchmark_free_source", "fetchable", "notes",
}


def _rows() -> list[dict]:
    with CATALOG.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_columns_present():
    with CATALOG.open(encoding="utf-8") as f:
        header = set(next(csv.reader(f)))
    assert header == REQUIRED_COLUMNS, f"unexpected columns: {header ^ REQUIRED_COLUMNS}"


def test_representative_spread():
    rows = _rows()
    assert len(rows) >= 15, "catalog should hold a representative spread (>=15 funds)"
    # A spread, not all one category.
    assert len({r["category"] for r in rows}) >= 10


def test_core_fields_populated():
    for r in _rows():
        assert r["fund_name"].strip(), "blank fund_name"
        assert r["mfapi_code"].strip().isdigit(), f"non-numeric mfapi_code: {r['fund_name']}"
        assert r["stated_benchmark"].strip(), f"blank stated_benchmark: {r['fund_name']}"
        assert r["category"].strip(), f"blank category: {r['fund_name']}"


def test_no_duplicates():
    rows = _rows()
    codes = [r["mfapi_code"] for r in rows]
    names = [r["fund_name"] for r in rows]
    assert len(codes) == len(set(codes)), "duplicate mfapi_code"
    assert len(names) == len(set(names)), "duplicate fund_name"


def test_fetchable_consistency():
    """fetchable is yes/no and consistent with niftyindices_name + free_source."""
    for r in _rows():
        fetchable = r["fetchable"].strip()
        assert fetchable in {"yes", "no"}, f"bad fetchable {fetchable!r}: {r['fund_name']}"
        has_name = bool(r["niftyindices_name"].strip())
        source = r["benchmark_free_source"].strip()
        if fetchable == "yes":
            assert has_name, f"fetchable fund lacks niftyindices_name: {r['fund_name']}"
            assert source == "niftyindices", f"expected niftyindices source: {r['fund_name']}"
        else:
            assert not has_name, f"non-fetchable fund has niftyindices_name: {r['fund_name']}"
            assert source == "none", f"expected free_source=none: {r['fund_name']}"


def test_catalog_supersets_vro_funds():
    """Every fund tracked for VRO parity must appear in the catalog."""
    with VRO_FUNDS.open(encoding="utf-8") as f:
        vro_codes = {r["mfapi_code"] for r in csv.DictReader(f)}
    catalog_codes = {r["mfapi_code"] for r in _rows()}
    missing = vro_codes - catalog_codes
    assert not missing, f"vro_funds.csv codes missing from the catalog: {missing}"
