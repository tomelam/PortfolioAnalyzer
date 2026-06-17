"""Golden-master regression: re-run main.py and compare CSV output.

Phase C safety net captured 2026-06-14. Covers three portfolios:

- port-1: 5 mutual funds + NIFTY benchmark (simplest path)
- port-mf-ppf-gold: 5 MFs + PPF (synthetic CIV) + physical gold (monthly)
- port-everything: 5 MFs + PPF + Gold + SGB + SCSS + REC Bond
  (exercises every loader)

Limitations to lift in Phase D:

- Hits mfapi.in for live NAV data → marked `network`. CAGR / Sharpe /
  Sortino / Volatility drift slightly as the API ingests new NAV
  points; numeric tolerances accommodate ~1 week of drift. Drawdown
  start dates and durations are anchored to historical events and
  compared exactly.
- Capture used `tests/fixtures/golden_master_config.toml` to bypass
  the data-staleness gates; tests use the same config until the data
  files are refreshed (see KANBAN "Data freshness").
- As of 2026-06-15, daily/monthly Sharpe and Vol agree to within ~1pp
  across all 3 portfolios after the CIV scale + frequency fixes.
- True determinism requires either a `--as-of YYYY-MM-DD` flag in
  main.py or a pickle-replay path. Both are KANBAN items.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VENV_PYTHON = REPO_ROOT / "venv" / "bin" / "python"
GOLDEN_CONFIG = REPO_ROOT / "tests" / "fixtures" / "golden_master_config.toml"

# Positional CSV columns:
#   0  label                (exact)
#   1  CAGR %               (numeric, drifts daily)
#   2  Volatility %         (numeric, drifts daily)
#   3  Sharpe               (numeric, drifts daily)
#   4  Sortino              (numeric, drifts daily)
#   5  Alpha %              (numeric, drifts daily)
#   6  Beta                 (numeric, drifts daily)
#   7  Drawdowns count      (int, may drift if a new drawdown forms)
#   8  Max Drawdown %       (numeric, anchored)
#   9  Max Drawdown Start   (date, anchored)
#   10 Drawdown Days        (int, anchored)
#   11 Recovery Days        (int, anchored)

PORTFOLIOS = ["port-1", "port-mf-ppf-gold", "port-everything"]
METHODS = ["daily", "monthly"]


def _parse_pct(s: str) -> float:
    return float(s.rstrip("%"))


def _parse_csv_row(path: Path) -> list[str]:
    with path.open() as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1, f"Expected one data row in {path}, got {len(rows)}"
    return rows[0]


def _run_main(toml: str, method: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(VENV_PYTHON),
        "main.py",
        "--config",
        str(GOLDEN_CONFIG),
        "--quiet",
        "--disable-plot-display",
        "--output-dir",
        str(out_dir),
        "--output-csv",
        "--metrics-method",
        method,
        "--lookback",
        "5Y",
        f"port/{toml}.toml",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, (
        f"main.py exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.golden
@pytest.mark.network
@pytest.mark.parametrize("portfolio", PORTFOLIOS)
@pytest.mark.parametrize("method", METHODS)
def test_golden(tmp_path: Path, portfolio: str, method: str) -> None:
    out = tmp_path / "out"
    _run_main(portfolio, method, out)

    actual_csv = out / f"{portfolio}.csv"
    assert actual_csv.exists(), f"main.py did not produce {actual_csv}"

    expected_csv = REPO_ROOT / "tests" / "golden" / portfolio / method / f"{portfolio}.csv"
    actual = _parse_csv_row(actual_csv)
    expected = _parse_csv_row(expected_csv)

    assert len(actual) == len(expected), f"column count drift: {len(actual)} vs {len(expected)}"

    # Label and anchored historical columns: exact.
    assert actual[0] == expected[0], "portfolio label mismatch"
    assert actual[9] == expected[9], (
        f"max-drawdown start date drift: {actual[9]} vs {expected[9]}"
    )
    assert actual[10] == expected[10], "drawdown days drift"
    assert actual[11] == expected[11], "recovery days drift"

    # Numeric tolerances reflect ~1 week of live mfapi.in NAV drift.
    # Synthetic-CIV portfolios produce larger Sharpe/Vol magnitudes, so
    # we scale tolerance to the magnitude of the expected value (5% relative).
    def rel_tol(v: float) -> float:
        return max(0.5, abs(v) * 0.05)

    cagr_e = _parse_pct(expected[1])
    vol_e = _parse_pct(expected[2])
    sharpe_e = float(expected[3])
    sortino_e = float(expected[4])
    alpha_e = _parse_pct(expected[5])
    beta_e = float(expected[6])
    mdd_e = _parse_pct(expected[8])

    assert abs(_parse_pct(actual[1]) - cagr_e) < rel_tol(cagr_e), "CAGR drift > tolerance"
    assert abs(_parse_pct(actual[2]) - vol_e) < rel_tol(vol_e), "Volatility drift > tolerance"
    assert abs(float(actual[3]) - sharpe_e) < rel_tol(sharpe_e), "Sharpe drift > tolerance"
    assert abs(float(actual[4]) - sortino_e) < rel_tol(sortino_e), "Sortino drift > tolerance"
    assert abs(_parse_pct(actual[5]) - alpha_e) < rel_tol(alpha_e), "Alpha drift > tolerance"
    assert abs(float(actual[6]) - beta_e) < rel_tol(beta_e), "Beta drift > tolerance"
    assert abs(_parse_pct(actual[8]) - mdd_e) < 0.2, "max drawdown drift"
