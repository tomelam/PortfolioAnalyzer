"""Golden-master regression: re-run main.py and compare CSV output.

This is the Phase C safety net captured 2026-06-14. It is intentionally
narrow — only `port/port-1.toml` is covered. `port-mf-ppf-gold` and
`port-everything` aborted during capture (gold loader returns a DataFrame
where main.py expects a Series, see KANBAN); coverage extends once those
are fixed in Phase D.

Limitations to lift in Phase D:

- Test hits mfapi.in for live NAV data, so it's marked `network` and skipped
  by default. The CAGR / Sharpe / Sortino / Volatility columns drift slightly
  as the API ingests new NAV points; we compare numerics with absolute
  tolerance. The drawdown columns are anchored to a fixed historical event
  and should stay exact.
- The capture used `tests/fixtures/golden_master_config.toml` to bypass
  stale-data gates; tests must use the same config until the data files
  are refreshed.
- True determinism requires either a `--today` flag in main.py or a cached
  response replay (sqlite-vec-style fixture). Both are KANBAN items.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VENV_PYTHON = REPO_ROOT / "venv" / "bin" / "python"
GOLDEN_CONFIG = REPO_ROOT / "tests" / "fixtures" / "golden_master_config.toml"

# Column meanings (positional, matching the user's existing CSV format):
#   0  label                (exact)
#   1  CAGR %               (numeric, drifts daily)
#   2  Volatility %         (numeric, drifts daily)
#   3  Sharpe               (numeric, drifts daily)
#   4  Sortino              (numeric, drifts daily)
#   5  Alpha %              (numeric, drifts daily)
#   6  Beta                 (numeric, drifts daily)
#   7  Drawdowns count      (int, may drift if a new drawdown forms)
#   8  Max Drawdown %       (numeric, anchored to historical event)
#   9  Max Drawdown Start   (date, anchored)
#   10 Drawdown Days        (int, anchored)
#   11 Recovery Days        (int, anchored)


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
        f"port/{toml}",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (
        f"main.py exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.golden
@pytest.mark.network
@pytest.mark.parametrize("method", ["daily", "monthly"])
def test_port1_golden(tmp_path: Path, method: str) -> None:
    out = tmp_path / "out"
    _run_main("port-1.toml", method, out)

    actual_csv = out / "port-1.csv"
    assert actual_csv.exists(), f"main.py did not produce {actual_csv}"

    expected_csv = REPO_ROOT / "tests" / "golden" / "port-1" / method / "port-1.csv"
    actual = _parse_csv_row(actual_csv)
    expected = _parse_csv_row(expected_csv)

    assert len(actual) == len(expected), f"column count drift: {len(actual)} vs {len(expected)}"

    # Label and anchored historical columns: exact match.
    assert actual[0] == expected[0], "portfolio label mismatch"
    assert actual[9] == expected[9], f"max-drawdown start date drift: {actual[9]} vs {expected[9]}"
    assert actual[10] == expected[10], "drawdown days drift"
    assert actual[11] == expected[11], "recovery days drift"

    # Numeric columns that drift with live mfapi.in data. Tolerance ~0.5
    # percentage points / Sharpe units accommodates ~1 week of NAV drift.
    tol = 0.5
    assert abs(_parse_pct(actual[1]) - _parse_pct(expected[1])) < tol, "CAGR drift > tolerance"
    assert abs(_parse_pct(actual[2]) - _parse_pct(expected[2])) < tol, "Volatility drift > tolerance"
    assert abs(float(actual[3]) - float(expected[3])) < tol, "Sharpe drift > tolerance"
    assert abs(float(actual[4]) - float(expected[4])) < tol, "Sortino drift > tolerance"
    assert abs(_parse_pct(actual[5]) - _parse_pct(expected[5])) < tol, "Alpha drift > tolerance"
    assert abs(float(actual[6]) - float(expected[6])) < tol, "Beta drift > tolerance"

    # Max drawdown % is anchored to a historical event (2021-10-18 → 2022-08-10).
    # A small tolerance covers any rounding differences.
    assert abs(_parse_pct(actual[8]) - _parse_pct(expected[8])) < 0.2, "max drawdown drift"
