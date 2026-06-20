"""Unit tests for the reporting helpers on ``TimeseriesReturn``.

These six methods (``info_summary``, ``describe_as_report``, ``to_csv_report``,
``to_latex_table``, ``compare_to``, ``as_rolling``) were parked in
``attic/timeseries_return_helpers.py`` and revived (re-wired to the current
Series-based API) as Thread 6 of the 2026-06 cleanup round. The parked code
referenced an older surface (``self.columns`` / ``self.shape`` /
``self['value']`` / ``self.annualized()``) that no longer exists.

``info``/``describe``/``compare`` write to **stderr** (via ``utils.info``), so
they are asserted with ``capsys.readouterr().err``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolioanalyzer.timeseries.returns import TimeseriesReturn


def _prices(n=400, seed=0, start="2020-01-01"):
    """A varied (up-and-down) price/CIV series of length n, named 'value'."""
    rng = np.random.default_rng(seed)
    vals = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.Series(vals, index=idx, name="value")


# --- info_summary ----------------------------------------------------------

def test_info_summary_prints_structure(capsys):
    ts = TimeseriesReturn(_prices())
    ts.info_summary(name="MyTS")
    err = capsys.readouterr().err
    assert "MyTS shape:" in err
    assert "date range:" in err
    assert "2020-01-01" in err
    assert "NaNs in 'value': 0" in err


def test_info_summary_counts_nans(capsys):
    s = _prices(n=50)
    s.iloc[3] = np.nan
    TimeseriesReturn(s).info_summary()
    assert "NaNs in 'value': 1" in capsys.readouterr().err


# --- describe_as_report ----------------------------------------------------

def test_describe_as_report_prints_stats(capsys):
    ts = TimeseriesReturn(_prices(n=120))
    ts.describe_as_report(name="Fund X")
    err = capsys.readouterr().err
    assert "Report: Fund X" in err
    assert "Observations: 120" in err
    assert "- Mean:" in err
    assert "- Std Dev:" in err


# --- to_csv_report ---------------------------------------------------------

def test_to_csv_report_roundtrip(tmp_path):
    s = _prices(n=200)
    out = tmp_path / "report.csv"
    TimeseriesReturn(s).to_csv_report(out, name="P1")

    df = pd.read_csv(out)
    assert list(df.columns) == [
        "name", "start_date", "end_date", "observations",
        "missing", "nonzero", "mean", "std_dev", "min", "max",
    ]
    row = df.iloc[0]
    assert row["name"] == "P1"
    assert row["observations"] == 200
    assert row["missing"] == 0
    assert row["nonzero"] == 200
    assert row["min"] == pytest.approx(s.min())
    assert row["max"] == pytest.approx(s.max())


# --- to_latex_table --------------------------------------------------------

def test_to_latex_table_single_column():
    tex = TimeseriesReturn(_prices()).to_latex_table(name="Series A")
    assert "\\begin{table}[ht]" in tex
    assert "\\begin{tabular}{lr}" in tex          # single value column
    assert "Metric & Series A" in tex
    assert "Comparison" not in tex
    assert "CAGR" in tex
    assert "Sharpe" in tex
    assert tex.strip().endswith("\\end{table}")


def test_to_latex_table_comparison_column():
    a = TimeseriesReturn(_prices(seed=1))
    b = TimeseriesReturn(_prices(seed=2))
    tex = a.to_latex_table(compare_to=b, name="A", title="My Caption", label="tab:x")
    assert "\\begin{tabular}{lrr}" in tex          # two value columns
    assert "& Comparison" in tex
    assert "\\caption{My Caption}" in tex
    assert "\\label{tab:x}" in tex


def test_to_latex_table_percent_vs_ratio_formatting():
    """CAGR/Max Drawdown/Volatility render as percents; Sharpe/Sortino as ratios."""
    tex = TimeseriesReturn(_prices()).to_latex_table()
    lines = {line.split("&")[0].strip(): line for line in tex.splitlines() if "&" in line}
    assert "\\%" in lines["CAGR"]
    assert "\\%" in lines["Max Drawdown"]
    assert "\\%" in lines["Volatility"]
    assert "\\%" not in lines["Sharpe"]
    assert "\\%" not in lines["Sortino"]


# --- compare_to ------------------------------------------------------------

def test_compare_to_prints_both(capsys):
    a = TimeseriesReturn(_prices(seed=1))
    b = TimeseriesReturn(_prices(seed=2))
    a.compare_to(b, name_self="Alpha", name_other="Beta")
    err = capsys.readouterr().err
    assert "Comparison: Alpha vs Beta" in err
    assert "Alpha" in err and "Beta" in err
    assert "CAGR" in err and "Sharpe" in err


def test_compare_to_rejects_non_timeseries():
    a = TimeseriesReturn(_prices())
    with pytest.raises(TypeError):
        a.compare_to(pd.Series([1.0, 2.0, 3.0], name="value"))


def test_compare_to_raises_on_low_overlap():
    a = TimeseriesReturn(_prices(n=40, start="2020-01-01"))
    b = TimeseriesReturn(_prices(n=40, start="2021-06-01"))  # no/low date overlap
    with pytest.raises(ValueError, match="overlap"):
        a.compare_to(b)


def test_compare_to_raises_on_empty():
    a = TimeseriesReturn(_prices())
    empty = TimeseriesReturn(pd.Series([], dtype=float, name="value"))
    with pytest.raises(ValueError, match="empty"):
        a.compare_to(empty)


# --- as_rolling ------------------------------------------------------------

def test_as_rolling_mean_of_constant_is_constant():
    idx = pd.date_range("2022-01-01", periods=10, freq="D")
    const = TimeseriesReturn(pd.Series([5.0] * 10, index=idx, name="value"))
    rolled = const.as_rolling(window=3, method="mean")
    assert isinstance(rolled, TimeseriesReturn)
    assert rolled.name == "value"
    # after the first full window every value is the constant
    assert rolled.value_series().iloc[2:].eq(5.0).all()


def test_as_rolling_rejects_bad_method():
    ts = TimeseriesReturn(_prices())
    with pytest.raises(ValueError, match="Unsupported method"):
        ts.as_rolling(method="kurtosis")


def test_as_rolling_supports_std():
    rolled = TimeseriesReturn(_prices()).as_rolling(window=5, method="std")
    assert isinstance(rolled, TimeseriesReturn)
    assert len(rolled.value_series()) == 400
