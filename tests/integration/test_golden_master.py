"""Golden-master regression: re-run main.py and compare CSV output.

Phase C safety net captured 2026-06-14. Covers three portfolios:

- port-1: 5 mutual funds + NIFTY benchmark (simplest path)
- port-mf-ppf-gold: 5 MFs + PPF (synthetic CIV) + physical gold (LBMA USD/oz daily)
- port-everything: 5 MFs + PPF + Gold + SGB + SCSS
  (exercises every loader)

Determinism: runs are pinned with `--as-of {AS_OF}`, which trims every
loaded series to `<= AS_OF`. Indian MF NAVs for settled past dates are
immutable, so newly-ingested NAV points are excluded and a run
reproduces the captured goldens bit-for-bit regardless of when it
executes. Every numeric column is therefore compared at 1e-9; the
anchored drawdown columns (start date, durations) are compared exactly.

Offline: runs pass `--replay-from tests/golden/replay`, so NAV history
(`replay/navs/<fund>.csv`) and the SCSS rate table (`replay/scss_nsi.html`)
are read from committed fixtures instead of mfapi.in / nsiindia.gov.in.
No `network` marker — this runs in the default suite, and the conftest
network guard would raise if any `requests.get` slipped through.

Notes:

- Capture used `tests/fixtures/golden_master_config.toml`, which pins the
  benchmark + risk-free reference inputs to FROZEN copies under
  `tests/golden/replay/reference/` (not the live `data/` files, which the
  on-run freshness refresh rewrites in place). This insulates the goldens
  from `data/` mutation; `test_golden_config_does_not_read_live_data_dir`
  enforces it.
- Re-capture the goldens, the replay fixtures, and the frozen reference
  inputs together:
    1. Copy the current live reference CSVs into the freeze:
       `cp "data/reference/NIFTY Total Returns Historical Data.csv" \
           "data/reference/INDIRLTLT01STM.csv" \
           "data/reference/gold_lbma_usd_daily.csv" \
           "data/reference/DEXINUS.csv" tests/golden/replay/reference/`
    2. Run main.py for each (portfolio, method) with the same `--config`,
       `--lookback 5Y`, and `--as-of` used here, plus `--save-replay
       tests/golden/replay` once (any portfolio covering all funds + SCSS,
       e.g. port-everything), writing each CSV into
       tests/golden/<portfolio>/<method>/.
  Bump AS_OF only on re-capture, never independently.
"""

from __future__ import annotations

import csv
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VENV_PYTHON = REPO_ROOT / "venv" / "bin" / "python"
GOLDEN_CONFIG = REPO_ROOT / "tests" / "fixtures" / "golden_master_config.toml"
REPLAY_DIR = REPO_ROOT / "tests" / "golden" / "replay"

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

# Reference date the goldens were captured at. Pinning --as-of trims all
# series to <= AS_OF, making runs deterministic. Bump only on re-capture.
AS_OF = "2026-06-13"

# Floating-point comparison tolerance for every numeric column. With the
# --as-of pin the inputs are identical run-to-run, so output reproduces
# exactly; 1e-9 absorbs only CSV float-formatting noise.
TOL = 1e-9


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
        "-m", "portfolioanalyzer.main",
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
        "--as-of",
        AS_OF,
        "--replay-from",
        str(REPLAY_DIR),
        f"examples/port/{toml}.toml",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, (
        f"main.py exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.golden
def test_golden_config_does_not_read_live_data_dir() -> None:
    """Guard: the golden config must pin its reference inputs (benchmark +
    risk-free + gold) to the FROZEN copies under tests/golden/replay/, never the
    live data/ dir.

    The on-run freshness refresh (loaders.data_update) rewrites the live
    data/ CSVs in place. If the goldens read those files, an ordinary
    ``./pa`` run would silently change the golden inputs and break the
    suite (the documented gotcha this guard exists to prevent). This test
    fails loudly if anyone re-points the goldens back at data/, or if a
    frozen reference file goes missing.
    """
    with GOLDEN_CONFIG.open("rb") as f:
        cfg = tomllib.load(f)

    ref_paths = [
        cfg.get("benchmark_returns_file"),
        cfg.get("risk_free_rates_file"),
        cfg.get("gold_prices_file"),
        cfg.get("fx_usd_inr_file"),
    ]
    assert all(ref_paths), (
        "golden config must explicitly set benchmark_returns_file, "
        "risk_free_rates_file, gold_prices_file, and fx_usd_inr_file; "
        f"got {ref_paths}"
    )

    for rel in ref_paths:
        p = Path(rel)
        assert p.parts[0] != "data", (
            f"golden reference input {rel!r} points at the live data/ dir, which "
            "the freshness refresh mutates — pin it to a frozen copy under "
            "tests/golden/replay/ instead"
        )
        assert (REPO_ROOT / p).is_file(), f"frozen golden reference input missing: {rel}"


@pytest.mark.golden
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

    # With --as-of pinned the run is deterministic, so every numeric
    # column must reproduce the golden to within float-formatting noise.
    assert abs(_parse_pct(actual[1]) - _parse_pct(expected[1])) < TOL, "CAGR drift"
    assert abs(_parse_pct(actual[2]) - _parse_pct(expected[2])) < TOL, "Volatility drift"
    assert abs(float(actual[3]) - float(expected[3])) < TOL, "Sharpe drift"
    assert abs(float(actual[4]) - float(expected[4])) < TOL, "Sortino drift"
    assert abs(_parse_pct(actual[5]) - _parse_pct(expected[5])) < TOL, "Alpha drift"
    assert abs(float(actual[6]) - float(expected[6])) < TOL, "Beta drift"
    assert abs(_parse_pct(actual[8]) - _parse_pct(expected[8])) < TOL, "max drawdown drift"
    # Drawdowns count is an int that no longer drifts under the pin.
    assert actual[7] == expected[7], "drawdowns count drift"
