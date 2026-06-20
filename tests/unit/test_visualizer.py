"""Smoke tests for the plotting path (visualizer.py).

The golden/e2e runs use --disable-plot-display, so the actual plotting code
is otherwise unexercised. These tests render to the headless Agg backend and
assert the figure is produced (a PNG is written) without raising — catching
signature/import/layout regressions in the path main.py uses for its PNGs.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; must precede any pyplot import

from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from portfolioanalyzer import visualizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TOML = str(REPO_ROOT / "port" / "port-1.toml")


def _cumulative(n=120, start=100.0, step=0.1, seed_offset=0.0):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    vals = [start + i * step + seed_offset for i in range(n)]
    return pd.Series(vals, index=idx, name="value")


def test_print_major_drawdowns_runs(capsys):
    # Shape produced by TimeseriesCIV.max_drawdowns (start/end keys), which is
    # what main.py feeds print_major_drawdowns.
    drawdowns = [
        {
            "start": pd.Timestamp("2020-02-01"),
            "end": pd.Timestamp("2020-06-01"),
            "drawdown": 0.18,
            "drawdown_days": 29,
            "recovery_days": 121,
        }
    ]
    visualizer.print_major_drawdowns(drawdowns)
    # exercises the formatting/print path without raising
    capsys.readouterr()


def test_print_major_drawdowns_empty():
    visualizer.print_major_drawdowns([])  # no rows → no crash


def test_plot_cumulative_returns_writes_png(tmp_path):
    out = tmp_path / "plot.png"
    visualizer.plot_cumulative_returns(
        portfolio_label="Test Portfolio",
        cumulative_historical=_cumulative(),
        title="Test Plot",
        toml_file=TOML,
        benchmark_cumulative=_cumulative(seed_offset=2.0),
        benchmark_name="NIFTY TRI",
        metrics={
            "CAGR": 0.12,
            "Volatility": 0.10,
            "Sharpe Ratio": 0.5,
            "Sortino Ratio": 0.7,
        },
        save_path=str(out),
    )
    assert out.exists() and out.stat().st_size > 0


def test_plot_embeds_png_metadata_and_footnote(tmp_path):
    """PNG tEXt metadata round-trips via PIL, and a footnote renders without
    raising — the visible + invisible provenance copies main.py relies on."""
    from PIL import Image

    out = tmp_path / "meta.png"
    md = {
        "metrics": "CAGR: 12.34%\nSharpe: 1.2345",
        "reference_data": "NIFTY 50 TRI: last 2026-06-13 | fetched 2026-06-17",
    }
    visualizer.plot_cumulative_returns(
        portfolio_label="Test Portfolio",
        cumulative_historical=_cumulative(),
        title="Test Plot",
        toml_file=TOML,
        benchmark_cumulative=_cumulative(seed_offset=2.0),
        benchmark_name="NIFTY TRI",
        metrics={"Sharpe Ratio": 0.5, "Sortino Ratio": 0.7},
        save_path=str(out),
        png_metadata=md,
        footnote="Reference data (live) — NIFTY 50 TRI: last 2026-06-13",
    )
    assert out.exists() and out.stat().st_size > 0
    text = Image.open(out).text
    assert text["metrics"] == "CAGR: 12.34%\nSharpe: 1.2345"
    assert "NIFTY 50 TRI" in text["reference_data"]


def test_plot_cumulative_returns_without_benchmark(tmp_path):
    out = tmp_path / "plot_nobench.png"
    visualizer.plot_cumulative_returns(
        portfolio_label="No Bench",
        cumulative_historical=_cumulative(),
        title="No Benchmark",
        toml_file=TOML,
        benchmark_cumulative=None,
        save_path=str(out),
    )
    assert out.exists() and out.stat().st_size > 0
