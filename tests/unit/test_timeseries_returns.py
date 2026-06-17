"""Unit tests for the live surface of ``timeseries.returns.TimeseriesReturn``.

Covers the accessors, ``_standardized_returns`` (daily + monthly), the
metric delegators, and the error branches of the alpha/beta methods. The
class's former reporting/alignment helpers were parked in
``attic/timeseries_return_helpers.py`` and are intentionally not tested here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from timeseries.returns import TimeseriesReturn


def _prices(n=400, seed=0, start="2020-01-01"):
    """A varied (up-and-down) price/CIV series of length n."""
    rng = np.random.default_rng(seed)
    vals = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.Series(vals, index=idx, name="value")


def _returns(n=300, seed=1, start="2021-01-01"):
    """A varied daily-return series of length n."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.Series(rng.normal(0.0004, 0.008, n), index=idx, name="value")


# --- accessors -------------------------------------------------------------

def test_init_renames_to_value():
    ts = TimeseriesReturn(pd.Series([1.0, 2.0, 3.0], name="something_else"))
    assert ts.name == "value"


def test_init_rejects_non_series():
    with pytest.raises(TypeError, match="expects a pd.Series"):
        TimeseriesReturn([1, 2, 3])


def test_index_values_iloc_loc():
    s = _prices(5)
    ts = TimeseriesReturn(s)
    assert list(ts.index) == list(s.index)
    assert isinstance(ts.values, np.ndarray)
    np.testing.assert_array_equal(ts.values, s.values)
    assert ts.iloc[0] == s.iloc[0]
    assert ts.loc[s.index[2]] == s.iloc[2]


def test_set_series_replaces_and_sorts():
    ts = TimeseriesReturn(_prices(5))
    idx = pd.date_range("2022-03-01", periods=3, freq="D")[::-1]  # reversed
    ts.set_series(pd.Series([3.0, 2.0, 1.0], index=idx, name="value"))
    assert list(ts.index) == sorted(ts.index)


def test_set_series_rejects_non_series():
    ts = TimeseriesReturn(_prices(5))
    with pytest.raises(TypeError, match="expects a pd.Series"):
        ts.set_series(123)


def test_dropna_returns_new_instance_without_nans():
    s = _prices(5)
    s.iloc[2] = np.nan
    ts = TimeseriesReturn(s).dropna()
    assert isinstance(ts, TimeseriesReturn)
    assert not ts.values.tolist().__contains__(np.nan)
    assert len(ts.index) == 4


def test_mean():
    s = _prices(10)
    assert TimeseriesReturn(s).mean() == pytest.approx(s.mean())


def test_value_series_raises_when_not_named_value():
    ts = TimeseriesReturn(_prices(3))
    # set_series does not rename, so the name guard in value_series fires.
    ts.set_series(pd.Series([1.0, 2.0], index=pd.date_range("2020-01-01", periods=2), name="foo"))
    with pytest.raises(ValueError, match="Expected series name to be 'value'"):
        ts.value_series()


# --- metric delegators -----------------------------------------------------

@pytest.mark.parametrize("frequency", ["daily", "monthly"])
def test_volatility_sharpe_sortino_both_frequencies(frequency):
    ts = TimeseriesReturn(_prices(400))
    assert np.isfinite(ts.volatility(frequency=frequency))
    assert np.isfinite(ts.sharpe(frequency=frequency))
    assert np.isfinite(ts.sortino(frequency=frequency))


def test_cagr_and_max_drawdown():
    ts = TimeseriesReturn(_prices(400))
    assert np.isfinite(ts.cagr())
    mdd = ts.max_drawdown()
    assert mdd <= 0 or np.isfinite(mdd)  # metrics.max_drawdown returns a finite number


def test_max_drawdowns_returns_recovered_dips():
    # peak 100 -> trough 80 (20% dip) -> full recovery to 105
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    s = pd.Series([100.0, 90.0, 80.0, 95.0, 105.0], index=idx, name="value")
    dips = TimeseriesReturn(s).max_drawdowns(threshold=0.05)
    assert isinstance(dips, list)
    assert len(dips) == 1
    assert set(dips[0]) >= {"start_date", "trough_date", "recovery_date", "drawdown"}


# --- alpha / beta: happy paths ---------------------------------------------

def test_alpha_and_beta_happy_paths():
    bench = TimeseriesReturn(_returns(300, seed=1))
    port_vals = 0.8 * bench.value_series() + _returns(300, seed=2).values
    port = TimeseriesReturn(pd.Series(port_vals, index=bench.index, name="value"))

    assert np.isfinite(port.alpha_regression(bench))
    assert np.isfinite(port.beta_regression(bench))
    assert np.isfinite(port.alpha_capm(bench))
    assert np.isfinite(port.beta_capm(bench))


def test_alpha_capm_fallback_to_regression_beta():
    # fallback path: if beta_capm raises, fall back to beta_regression.
    bench = TimeseriesReturn(_returns(300, seed=3))
    port = TimeseriesReturn(pd.Series(_returns(300, seed=4).values, index=bench.index, name="value"))
    # Force beta_capm to raise, exercising the fallback branch.
    port.beta_capm = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    assert np.isfinite(port.alpha_capm(bench, fallback_to_simple_beta=True))


# --- alpha / beta: error branches ------------------------------------------

def _empty():
    return TimeseriesReturn(pd.Series([], dtype=float, name="value"))


def test_alpha_regression_empty_raises():
    with pytest.raises(ValueError, match="input series is empty"):
        _empty().alpha_regression(_empty())


def test_alpha_regression_no_variation_raises():
    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    flat = TimeseriesReturn(pd.Series([0.01] * 5, index=idx, name="value"))
    with pytest.raises(ValueError, match="no variation"):
        flat.alpha_regression(flat)


def test_alpha_regression_too_few_aligned_raises():
    idx = pd.date_range("2021-01-01", periods=2, freq="D")
    x = TimeseriesReturn(pd.Series([0.01, 0.02], index=idx, name="value"))
    y = TimeseriesReturn(pd.Series([0.01, 0.03], index=idx, name="value"))
    with pytest.raises(ValueError, match="Too few aligned data points"):
        y.alpha_regression(x)


def test_alpha_regression_trims_misaligned_then_succeeds():
    # x on days 0-4, y on days 2-6 -> 3 overlapping points (triggers the
    # alignment-trim dbg branch but still has >= 3 aligned points).
    xi = pd.date_range("2021-01-01", periods=5, freq="D")
    yi = pd.date_range("2021-01-03", periods=5, freq="D")
    x = TimeseriesReturn(pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=xi, name="value"))
    y = TimeseriesReturn(pd.Series([0.02, 0.01, 0.05, 0.03, 0.04], index=yi, name="value"))
    assert np.isfinite(y.alpha_regression(x))


def test_alpha_capm_too_few_aligned_raises():
    idx = pd.date_range("2021-01-01", periods=2, freq="D")
    x = TimeseriesReturn(pd.Series([0.01, 0.02], index=idx, name="value"))
    y = TimeseriesReturn(pd.Series([0.01, 0.03], index=idx, name="value"))
    with pytest.raises(ValueError, match="Too few aligned data points to compute alpha_capm"):
        y.alpha_capm(x)


def test_alpha_capm_empty_raises():
    with pytest.raises(ValueError, match="input series is empty"):
        _empty().alpha_capm(_empty())


def test_beta_regression_too_few_points_raises():
    idx = pd.date_range("2021-01-01", periods=1, freq="D")
    x = TimeseriesReturn(pd.Series([0.01], index=idx, name="value"))
    y = TimeseriesReturn(pd.Series([0.02], index=idx, name="value"))
    with pytest.raises(ValueError, match="at least two aligned data points"):
        y.beta_regression(x)


def test_beta_capm_too_few_aligned_raises():
    idx = pd.date_range("2021-01-01", periods=2, freq="D")
    x = TimeseriesReturn(pd.Series([0.01, 0.02], index=idx, name="value"))
    y = TimeseriesReturn(pd.Series([0.01, 0.03], index=idx, name="value"))
    with pytest.raises(ValueError, match="Too few aligned data points to compute beta_capm"):
        y.beta_capm(x)


def test_beta_capm_empty_raises():
    with pytest.raises(ValueError, match="input series is empty"):
        _empty().beta_capm(_empty())
