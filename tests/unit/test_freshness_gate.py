"""Unit tests for the block-by-default freshness gate glue in ``main.py``.

The per-source cadence logic is tested in ``test_data_update.py``; here we
pin the gate behaviour ``_enforce_reference_freshness`` adds on top: print
provenance, block on stale reference data, and let ``--allow-stale`` through
with a warning that names the degraded metrics.
"""

from __future__ import annotations

import pytest

import main
from loaders import data_update as du


def _settings(**over):
    s = {
        "use_benchmark": True,
        "benchmark_file": "data/NIFTY Total Returns Historical Data.csv",
        "risk_free_rates_file": "data/INDIRLTLT01STM.csv",
        "allow_stale": False,
    }
    s.update(over)
    return s


def _patch_results(monkeypatch, results, stamps=None):
    monkeypatch.setattr(du, "ensure_reference_data_fresh", lambda paths: results)
    monkeypatch.setattr(du, "read_stamp", lambda name: (stamps or {}).get(name, {}))


def test_gate_passes_and_prints_provenance(monkeypatch, capsys):
    results = [
        {"name": "benchmark_nifty_tri", "status": "refreshed", "message": "🔄 fresh", "affects": "alpha, beta"},
        {"name": "risk_free_fred", "status": "current", "message": "", "affects": "Sharpe, Sortino, alpha"},
    ]
    stamps = {
        "benchmark_nifty_tri": {"last_date": "2026-06-12", "fetched_at": "2026-06-15T03:00:00+00:00"},
        "risk_free_fred": {"last_date": "2026-05-01", "fetched_at": "2026-06-17T14:00:00+00:00"},
    }
    _patch_results(monkeypatch, results, stamps)
    main._enforce_reference_freshness(_settings())
    out = capsys.readouterr().out
    # provenance: both last_date and fetched_at per source
    assert "last data 2026-06-12" in out and "fetched 2026-06-15" in out
    assert "last data 2026-05-01" in out


def test_gate_blocks_on_stale_by_default(monkeypatch):
    results = [
        {"name": "risk_free_fred", "status": "stale", "message": "⚠️  down", "affects": "Sharpe, Sortino, alpha"},
    ]
    _patch_results(monkeypatch, results)
    with pytest.raises(RuntimeError, match="stale and could not be refreshed"):
        main._enforce_reference_freshness(_settings())


def test_gate_block_message_names_degraded_metrics(monkeypatch):
    results = [
        {"name": "benchmark_nifty_tri", "status": "stale", "message": "", "affects": "alpha, beta"},
        {"name": "risk_free_fred", "status": "stale", "message": "", "affects": "Sharpe, Sortino, alpha"},
    ]
    _patch_results(monkeypatch, results)
    with pytest.raises(RuntimeError) as exc:
        main._enforce_reference_freshness(_settings())
    msg = str(exc.value)
    # de-duplicated union of affected metrics across both stale sources
    for metric in ("Sharpe", "Sortino", "alpha", "beta"):
        assert metric in msg
    assert msg.count("alpha") == 1  # not duplicated across the two sources


def test_allow_stale_proceeds_with_warning(monkeypatch, capsys):
    results = [
        {"name": "risk_free_fred", "status": "stale", "message": "", "affects": "Sharpe, Sortino, alpha"},
    ]
    _patch_results(monkeypatch, results)
    main._enforce_reference_freshness(_settings(allow_stale=True))  # must not raise
    out = capsys.readouterr().out
    assert "--allow-stale" in out and "Degraded metrics" in out
    assert "Sharpe" in out


def test_benchmark_path_omitted_when_use_benchmark_false(monkeypatch):
    seen = {}

    def _capture(paths):
        seen["paths"] = paths
        return []

    monkeypatch.setattr(du, "ensure_reference_data_fresh", _capture)
    monkeypatch.setattr(du, "read_stamp", lambda name: {})
    main._enforce_reference_freshness(_settings(use_benchmark=False))
    assert seen["paths"] == ["data/INDIRLTLT01STM.csv"]  # no benchmark path
