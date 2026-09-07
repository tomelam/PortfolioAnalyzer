"""Offline unit tests for the VRO-parity tooling (``loaders.vro``).

The network/browser fetch is covered by the wire test
(``tests/integration/test_vro_parity.py``); here we pin the pure parts: the
peer-comparison parser (against a committed fixture), the trailing-CAGR helper,
and the fund-map loader. No network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolioanalyzer.loaders.vro import (
    VRO_RISK_KEYS,
    load_vro_fund_map,
    parse_peer_comparison_returns,
    parse_risk_ratios,
    trailing_cagr_pct,
    trailing_risk_ratios,
    vro_returns_api,
    vro_risk_ratios_api,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
FIXTURE = FIXTURES / "vro_peer_comparison_5y.json"
RISK_FIXTURE = FIXTURES / "vro_risk_ratios.html"


# ---- parse_peer_comparison_returns --------------------------------------


def test_parse_extracts_own_fund_return() -> None:
    text = FIXTURE.read_text()
    # ICICI Bluechip Direct (plan 15841) 5Y annualised = 14.08% in the fixture.
    assert parse_peer_comparison_returns(text, "15841") == pytest.approx(14.08)


def test_parse_picks_the_right_plan_among_peers() -> None:
    text = FIXTURE.read_text()
    # A peer in the same payload — must not be confused with our fund.
    assert parse_peer_comparison_returns(text, "16192") == pytest.approx(16.24)


def test_parse_missing_plan_raises() -> None:
    text = FIXTURE.read_text()
    with pytest.raises(ValueError, match="not found"):
        parse_peer_comparison_returns(text, "999999")


def test_parse_malformed_payload_raises() -> None:
    with pytest.raises(ValueError, match="malformed"):
        parse_peer_comparison_returns("not json", "15841")
    with pytest.raises(ValueError, match="malformed"):
        parse_peer_comparison_returns('{"wrong_key": []}', "15841")


# ---- trailing_cagr_pct (point-to-point, the VRO-matching method) --------


def test_trailing_cagr_point_to_point() -> None:
    # 100 -> 200 over exactly ~5 years ⇒ CAGR ≈ 2**(1/5) - 1 ≈ 14.87%.
    idx = pd.bdate_range("2021-01-01", "2026-01-01")
    nav = pd.Series(100.0, index=idx)
    nav.iloc[-1] = 200.0
    cagr = trailing_cagr_pct(nav, years=5)
    assert cagr == pytest.approx(14.87, abs=0.2)


def test_trailing_cagr_trims_to_window() -> None:
    # 10 years of data; the 3Y window must ignore everything older.
    idx = pd.bdate_range("2016-01-01", "2026-01-01")
    nav = pd.Series(range(1, len(idx) + 1), index=idx, dtype=float)
    anchor = nav.index.max()
    three_y_ago = anchor - pd.DateOffset(years=3)
    window = nav[(nav.index >= three_y_ago) & (nav.index <= anchor)]
    expected = ((window.iloc[-1] / window.iloc[0]) ** (365.25 / (window.index[-1] - window.index[0]).days) - 1) * 100
    assert trailing_cagr_pct(nav, years=3) == pytest.approx(expected, abs=1e-6)


def test_trailing_cagr_too_few_points_raises() -> None:
    nav = pd.Series([100.0], index=pd.to_datetime(["2026-01-01"]))
    with pytest.raises(ValueError, match="too few"):
        trailing_cagr_pct(nav, years=5)


# ---- fund map + url helpers ---------------------------------------------


def test_fund_map_loads_expected_columns() -> None:
    funds = load_vro_fund_map()
    assert len(funds) >= 2
    by_code = {f.mfapi_code: f for f in funds}
    icici = by_code["120586"]
    assert icici.vro_plan_id == "15841"
    assert icici.isin == "INF109K016L0"
    assert icici.mfapi_url == "https://api.mfapi.in/mf/120586"
    # Stated benchmark column (scaffolding for Beta/Alpha parity).
    assert icici.benchmark == "NIFTY 100 TRI"
    assert all(f.benchmark for f in funds)  # every mapped fund has one
    # Fetchable-benchmark column: only NIFTY 100 (ICICI Bluechip) is on
    # niftyindices' feed; the rest are empty (not sourceable / out of scope).
    assert icici.benchmark_index == "NIFTY 100"
    assert [f.benchmark_index for f in funds if f.benchmark_index] == ["NIFTY 100"]


def test_returns_api_url_shape() -> None:
    url = vro_returns_api("15841", "5Y")
    assert "fund_id=15841" in url and "period=5Y" in url


# ---- parse_risk_ratios (HTML fragment) ----------------------------------


def test_parse_risk_ratios_extracts_all_measures() -> None:
    # The committed fragment is ICICI Pru Large Cap Dir's 3Y risk table.
    risk = parse_risk_ratios(RISK_FIXTURE.read_text())
    assert risk == {
        "mean": 14.51,
        "std_dev": 13.51,
        "sharpe": 0.64,
        "sortino": 0.83,
        "beta": 0.92,
        "alpha": 3.33,
    }
    assert set(risk) == set(VRO_RISK_KEYS)


def test_parse_risk_ratios_missing_measure_raises() -> None:
    # A fragment that only carries Mean — the rest are absent.
    partial = (
        '<table><tbody><tr>'
        '<td><div class="first-col-custom-width">Mean (%)</div></td>'
        '<td class="mono-font">14.51</td></tr></tbody></table>'
    )
    with pytest.raises(ValueError, match="missing"):
        parse_risk_ratios(partial)


def test_parse_risk_ratios_empty_raises() -> None:
    with pytest.raises(ValueError, match="missing"):
        parse_risk_ratios("<div>nothing here</div>")


def test_risk_ratios_api_uses_short_name_not_plan_id() -> None:
    # The route keys on the fund's short name (url-encoded), with optional peers.
    url = vro_risk_ratios_api("ICICI Pru Large Cap Dir", ("HDFC Large Cap Dir",))
    assert "fund_name=ICICI%20Pru%20Large%20Cap%20Dir" in url
    assert "fund_name1=HDFC%20Large%20Cap%20Dir" in url
    assert url.endswith("lang=en")


# ---- trailing_risk_ratios (VRO methodology: trailing 3Y, monthly) -------


def _synthetic_monthly_nav(periods: int = 40, seed: int = 0) -> pd.Series:
    """A month-end NAV whose monthly returns are deterministic but non-constant."""
    idx = pd.date_range(end="2026-05-31", periods=periods, freq="ME")
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.01, 0.03, size=periods - 1)
    nav = pd.Series(100.0, index=idx)
    nav.iloc[1:] = 100.0 * np.cumprod(1.0 + rets)
    return nav


def test_trailing_risk_ratios_units_and_window() -> None:
    nav = _synthetic_monthly_nav()
    rf_annual = 0.06
    out = trailing_risk_ratios(nav, years=3, risk_free_annual=rf_annual)

    # Independently recompute on the trailing-3Y monthly returns.
    anchor = nav.index.max()
    window = nav[nav.index >= anchor - pd.DateOffset(years=3)]
    rr = window.pct_change().dropna()
    rf_m = rf_annual / 12.0

    assert out["mean"] == pytest.approx(rr.mean() * 12 * 100.0)
    assert out["std_dev"] == pytest.approx(rr.std() * np.sqrt(12) * 100.0)
    assert out["sharpe"] == pytest.approx(
        (rr - rf_m).mean() * 12 / (rr.std() * np.sqrt(12))
    )
    # Benchmark-free call omits Beta/Alpha.
    assert "beta" not in out and "alpha" not in out


def test_trailing_risk_ratios_beta_alpha_need_benchmark() -> None:
    nav = _synthetic_monthly_nav(seed=1)
    bench = _synthetic_monthly_nav(seed=2)
    out = trailing_risk_ratios(nav, years=3, benchmark_nav=bench)
    assert "beta" in out and "alpha" in out
    assert np.isfinite(out["beta"]) and np.isfinite(out["alpha"])


def test_trailing_risk_ratios_beta_alpha_known_answer() -> None:
    # A fund benchmarked against *itself* has CAPM beta 1 and alpha 0 by
    # construction — pins that benchmark_nav is wired into the CAPM math
    # correctly (not just that it returns finite numbers).
    nav = _synthetic_monthly_nav(seed=3)
    out = trailing_risk_ratios(nav, years=3, benchmark_nav=nav)
    assert out["beta"] == pytest.approx(1.0)
    assert out["alpha"] == pytest.approx(0.0, abs=1e-9)


def test_trailing_risk_ratios_too_few_points_raises() -> None:
    nav = _synthetic_monthly_nav(periods=6)  # only ~5 monthly returns
    with pytest.raises(ValueError, match="too few"):
        trailing_risk_ratios(nav, years=3)


class TestImpliedRiskFree:
    """VRO's assumed risk-free is not published, so our constant was back-solved
    by hand from a captured fragment -- and a hand-derived constant goes stale
    silently.

    It did: measured 2026-09-07, VRO's implied risk-free across four funds was
    5.62-5.69% while our constant said 5.90%. The failure did not look like a
    stale constant. It looked like "Sharpe is wrong for the corporate bond fund",
    because dSharpe = dRf / StdDev, so a fixed 0.23pp error shows up as 0.17 on a
    StdDev of 1.34 and as 0.01 on a StdDev of 17.94. The most stable fund in the
    set was the loudest symptom.

    implied_vro_risk_free() makes the derivation a one-liner over VRO's own
    published triple, so it can be re-checked instead of re-guessed."""

    def test_back_solves_the_risk_free_from_vros_own_numbers(self):
        from portfolioanalyzer.loaders.vro import implied_vro_risk_free
        # Sharpe = (mean - rf) / std  =>  rf = mean - sharpe * std
        assert implied_vro_risk_free(
            {"mean": 7.26, "std_dev": 1.34, "sharpe": 1.17}
        ) == pytest.approx(5.69, abs=0.01)

    def test_matches_a_high_volatility_fund_too(self):
        from portfolioanalyzer.loaders.vro import implied_vro_risk_free
        assert implied_vro_risk_free(
            {"mean": 20.87, "std_dev": 17.94, "sharpe": 0.85}
        ) == pytest.approx(5.62, abs=0.01)

    def test_missing_keys_raise_rather_than_returning_a_plausible_number(self):
        from portfolioanalyzer.loaders.vro import implied_vro_risk_free
        with pytest.raises(KeyError):
            implied_vro_risk_free({"mean": 7.26, "std_dev": 1.34})

    def test_a_zero_std_dev_raises_instead_of_dividing(self):
        from portfolioanalyzer.loaders.vro import implied_vro_risk_free
        with pytest.raises(ValueError, match="std_dev"):
            implied_vro_risk_free({"mean": 7.0, "std_dev": 0.0, "sharpe": 1.0})

    def test_our_constant_is_within_a_quarter_point_of_the_measurement(self):
        """Guards the constant itself. If VRO moves its assumption again this
        fails saying so, instead of surfacing as one fund's Sharpe."""
        from portfolioanalyzer.loaders.vro import VRO_RISK_FREE_ANNUAL
        # 0.10pp is derived, not picked: dSharpe = dRf / StdDev, the smallest
        # StdDev among the tracked funds is 1.34, and the Sharpe tolerance is
        # 0.10 -- so 0.10 * 1.34 = 0.134pp is where Sharpe starts to fail. A
        # 0.10pp guard therefore fires FIRST, and says what is actually wrong.
        assert abs(VRO_RISK_FREE_ANNUAL * 100 - 5.67) < 0.10, (
            f"VRO's implied risk-free has drifted from our constant "
            f"({VRO_RISK_FREE_ANNUAL*100:.2f}%). Re-derive it with "
            f"implied_vro_risk_free() over the tracked funds and update the "
            f"constant; do NOT widen the Sharpe tolerance."
        )
