"""Unit tests for the block-by-default freshness gate glue in ``main.py``.

The per-source cadence logic is tested in ``test_data_update.py``; here we
pin the gate behaviour ``_enforce_reference_freshness`` adds on top: print
provenance, block on stale reference data, and let ``--allow-stale`` through
with a warning that names the degraded metrics.
"""

from __future__ import annotations

import pytest

from portfolioanalyzer import main
from portfolioanalyzer.loaders import data_update as du


def _settings(**over):
    s = {
        "use_benchmark": True,
        "benchmark_file": "data/reference/NIFTY Total Returns Historical Data.csv",
        "risk_free_rates_file": "data/reference/INDIRLTLT01STM.csv",
        "gold_prices_file": "data/reference/gold_lbma_usd_daily.csv",
        "fx_usd_inr_file": "data/reference/DEXINUS.csv",
        "allow_stale": False,
    }
    s.update(over)
    return s


def _patch_results(monkeypatch, results):
    monkeypatch.setattr(du, "ensure_reference_data_fresh", lambda paths: results)


def test_gate_passes_when_all_current(monkeypatch, capsys):
    results = [
        {"name": "benchmark_nifty_tri", "status": "refreshed", "message": "🔄 fresh", "affects": "alpha, beta"},
        {"name": "risk_free_fred", "status": "current", "message": "", "affects": "Sharpe, Sortino, alpha"},
    ]
    _patch_results(monkeypatch, results)
    main._enforce_reference_freshness(_settings(), {})  # must not raise
    # status messages ride on stderr, keeping stdout clean for the report
    assert "🔄 fresh" in capsys.readouterr().err


def test_report_provenance_echoes_to_stderr_and_returns(monkeypatch, capsys):
    prov = [
        {"name": "risk_free_fred", "label": "FRED India 10Y (risk-free)",
         "last_date": "2026-05-01", "fetched_at": "2026-06-17T14:00:00+00:00",
         "attempted_at": "2026-06-17T14:00:00+00:00"},
    ]
    monkeypatch.setattr(du, "reference_provenance", lambda paths: prov)
    out = main._report_reference_provenance(_settings(), {})
    assert out == prov
    captured = capsys.readouterr()
    assert "FRED India 10Y (risk-free)" in captured.err and "last 2026-05-01" in captured.err
    assert "2026-06-17" not in captured.out  # stderr only, not stdout


def test_gate_blocks_on_stale_by_default(monkeypatch):
    results = [
        {"name": "risk_free_fred", "status": "stale", "message": "⚠️  down", "affects": "Sharpe, Sortino, alpha"},
    ]
    _patch_results(monkeypatch, results)
    with pytest.raises(RuntimeError, match="stale and could not be refreshed"):
        main._enforce_reference_freshness(_settings(), {})


def test_gate_block_message_names_degraded_metrics(monkeypatch):
    results = [
        {"name": "benchmark_nifty_tri", "status": "stale", "message": "", "affects": "alpha, beta"},
        {"name": "risk_free_fred", "status": "stale", "message": "", "affects": "Sharpe, Sortino, alpha"},
    ]
    _patch_results(monkeypatch, results)
    with pytest.raises(RuntimeError) as exc:
        main._enforce_reference_freshness(_settings(), {})
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
    main._enforce_reference_freshness(_settings(allow_stale=True), {})  # must not raise
    err = capsys.readouterr().err
    assert "--allow-stale" in err and "Degraded metrics" in err
    assert "Sharpe" in err


def test_benchmark_path_omitted_when_use_benchmark_false(monkeypatch):
    seen = {}

    def _capture(paths):
        seen["paths"] = paths
        return []

    monkeypatch.setattr(du, "ensure_reference_data_fresh", _capture)
    main._enforce_reference_freshness(_settings(use_benchmark=False), {})
    assert seen["paths"] == ["data/reference/INDIRLTLT01STM.csv"]  # no benchmark path


def test_gold_path_gated_only_for_gold_or_sgb_portfolios(monkeypatch):
    """The gold feed is appended (and thus block-gated) only when the portfolio
    actually holds gold or SGBs; other portfolios never touch it."""
    seen = {}

    def _capture(paths):
        seen["paths"] = list(paths)
        return []

    monkeypatch.setattr(du, "ensure_reference_data_fresh", _capture)
    gold_csv = "data/reference/gold_lbma_usd_daily.csv"

    main._enforce_reference_freshness(_settings(), {"funds": []})
    assert gold_csv not in seen["paths"]  # no gold/sgb ⇒ not gated

    main._enforce_reference_freshness(_settings(), {"gold": {}})
    assert gold_csv in seen["paths"]  # gold present ⇒ gated

    main._enforce_reference_freshness(_settings(), {"sgb": []})
    assert gold_csv in seen["paths"]  # sgb present ⇒ gated (valued off gold)


def test_sgb_portfolios_additionally_gate_on_usd_inr_fx(monkeypatch):
    """An SGB is valued in USD off LBMA gold but converts its rupee coupons to
    USD, so an SGB-bearing run gates on the FX feed *in addition to* gold. A
    gold-only portfolio does not pull in FX (the honest decoupling)."""
    seen = {}

    def _capture(paths):
        seen["paths"] = list(paths)
        return []

    monkeypatch.setattr(du, "ensure_reference_data_fresh", _capture)
    fx_csv = "data/reference/DEXINUS.csv"
    gold_csv = "data/reference/gold_lbma_usd_daily.csv"

    main._enforce_reference_freshness(_settings(), {"gold": {}})
    assert gold_csv in seen["paths"] and fx_csv not in seen["paths"]  # gold-only: no FX

    main._enforce_reference_freshness(_settings(), {"sgb": []})
    assert gold_csv in seen["paths"] and fx_csv in seen["paths"]  # sgb: gold + FX
