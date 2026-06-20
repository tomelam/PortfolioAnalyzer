"""Tests for the portfolio-analyzer-update console entry point."""

from __future__ import annotations

from portfolioanalyzer import data_update_cli


def test_cli_reports_and_exits_zero_on_success(capsys, monkeypatch):
    monkeypatch.setattr(
        data_update_cli,
        "update_all",
        lambda: [{"name": "risk_free_fred", "ok": True, "rows": 174, "last_date": "2026-05-01"}],
    )
    rc = data_update_cli.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "risk_free_fred" in out and "174 rows" in out
    assert "1/1 source(s) updated." in out


def test_cli_exits_one_when_all_fail(capsys, monkeypatch):
    monkeypatch.setattr(
        data_update_cli,
        "update_all",
        lambda: [{"name": "benchmark_nifty_tri", "ok": False, "error": "throttled"}],
    )
    rc = data_update_cli.main()
    assert rc == 1
    assert "0/1 source(s) updated." in capsys.readouterr().out


def test_cli_exits_zero_when_some_succeed(monkeypatch):
    monkeypatch.setattr(
        data_update_cli,
        "update_all",
        lambda: [
            {"name": "risk_free_fred", "ok": True, "rows": 1, "last_date": "2026-05-01"},
            {"name": "benchmark_nifty_tri", "ok": False, "error": "throttled"},
        ],
    )
    assert data_update_cli.main() == 0
