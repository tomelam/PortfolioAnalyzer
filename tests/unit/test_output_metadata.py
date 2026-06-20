"""Unit tests for the pure output-metadata formatters (``output_metadata``)."""

from __future__ import annotations

from portfolioanalyzer import output_metadata as om

PROV = [
    {
        "name": "benchmark_nifty_tri",
        "label": "NIFTY 50 TRI (benchmark)",
        "last_date": "2026-06-13",
        "fetched_at": "2026-06-17T14:03:02+00:00",
        "attempted_at": "2026-06-17T14:03:02+00:00",
    },
    {
        "name": "risk_free_fred",
        "label": "FRED India 10Y (risk-free)",
        "last_date": "2026-05-01",
        "fetched_at": None,
        "attempted_at": None,
    },
]


def test_format_kv_block_aligns_colons():
    block = om.format_kv_block([("CAGR", "12.34%"), ("Sharpe", "1.2345")])
    lines = block.splitlines()
    # colon-aligned: both values start at the same column
    assert lines[0].index("12.34%") == lines[1].index("1.2345")
    assert "CAGR:" in block and "Sharpe:" in block


def test_format_provenance_includes_all_three_stamps():
    text = om.format_provenance(PROV)
    assert "NIFTY 50 TRI (benchmark): last 2026-06-13" in text
    assert "fetched 2026-06-17T14:03:02+00:00" in text
    assert "attempted 2026-06-17T14:03:02+00:00" in text
    # missing stamps degrade to n/a, not a crash
    assert "FRED India 10Y (risk-free): last 2026-05-01 | fetched n/a | attempted n/a" in text


def test_format_provenance_footnote_is_dates_only():
    foot = om.format_provenance_footnote(PROV, run_label="live")
    assert foot.startswith("Reference data (live) — ")
    assert "NIFTY 50 TRI (benchmark): last 2026-06-13, fetched 2026-06-17" in foot
    assert "fetched n/a" in foot  # FRED has no fetched stamp
    # footnote trims timestamps to the date (no time component)
    assert "14:03" not in foot


def test_build_png_metadata_has_expected_keys_and_is_latin1_safe():
    md = om.build_png_metadata(
        portfolio_label="₹ Test Portfolio",  # non-Latin-1 rupee sign
        run_label="as-of 2026-06-13",
        generated="2026-06-18T00:00:00Z",
        metric_pairs=[("CAGR", "12.34%"), ("Beta", "0.8765")],
        provenance=PROV,
    )
    assert set(md) == {"portfolio", "run", "generated", "metrics", "reference_data"}
    assert "CAGR:" in md["metrics"] and "Beta:" in md["metrics"]
    assert "NIFTY 50 TRI (benchmark)" in md["reference_data"]
    # every value must be Latin-1 encodable (PNG tEXt requirement)
    for v in md.values():
        v.encode("latin-1")  # must not raise
