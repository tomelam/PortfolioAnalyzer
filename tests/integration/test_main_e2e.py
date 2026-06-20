"""End-to-end CLI integration tests.

These tests invoke ``main.py`` via subprocess to catch regressions at
the argparse / import / entry-point boundary that the unit suite and
golden-master numeric tests can miss. Intentionally redundant with
``test_golden_master.py`` at the CLI surface — different failure modes.

Network-free tests (help, missing-file) run by default. The full
smoke run that hits mfapi.in is marked ``network``.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Use the current interpreter so CI works without a checked-out venv.
PYTHON = sys.executable
GOLDEN_CONFIG = REPO_ROOT / "tests" / "fixtures" / "golden_master_config.toml"
# Deterministic + offline knobs (shared with test_golden_master): --as-of pins
# the eval date and skips the freshness gate; --replay-from reads NAV/SCSS from
# committed fixtures, so the failure-mode tests below need no network.
REPLAY_DIR = REPO_ROOT / "tests" / "golden" / "replay"
AS_OF = "2026-06-13"


@pytest.mark.integration
def test_help_exits_clean() -> None:
    """``python main.py --help`` should exit 0 and emit usage text."""
    result = subprocess.run(
        [PYTHON, "-m", "portfolioanalyzer.main", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert "usage" in result.stdout.lower()


@pytest.mark.integration
def test_missing_portfolio_file_exits_nonzero(tmp_path: Path) -> None:
    """A non-existent portfolio TOML should fail cleanly, not crash mid-run."""
    bogus = tmp_path / "nope.toml"
    result = subprocess.run(
        [
            PYTHON,
            "-m", "portfolioanalyzer.main",
            "--config",
            str(GOLDEN_CONFIG),
            "--quiet",
            "--disable-plot-display",
            str(bogus),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0, (
        f"Expected non-zero exit on missing TOML; got 0.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.integration
def test_help_lists_allow_stale_flag() -> None:
    """``--allow-stale`` must surface in ``--help`` — it is the single
    documented escape from the block-by-default freshness invariant."""
    result = subprocess.run(
        [PYTHON, "-m", "portfolioanalyzer.main", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--allow-stale" in result.stdout


@pytest.mark.integration
@pytest.mark.parametrize(
    "removed_flag",
    ["--skip-age-check", "--no-auto-update", "--max-riskfree-delay"],
)
def test_removed_freshness_flags_error_with_pointer(removed_flag: str) -> None:
    """The retired freshness knobs are hard-removed: passing one fails fast
    with a one-line error pointing at ``--allow-stale`` (no silent aliasing)."""
    result = subprocess.run(
        [PYTHON, "-m", "portfolioanalyzer.main", removed_flag, "port/port-1.toml"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "--allow-stale" in (result.stdout + result.stderr)


def _run_offline(config_path: Path, tmp_path: Path) -> subprocess.CompletedProcess:
    """Run port-1 deterministically and offline (--as-of + --replay-from) with
    the given config overlay. Returns the completed process for assertions."""
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    return subprocess.run(
        [
            PYTHON, "-m", "portfolioanalyzer.main",
            "--config", str(config_path),
            "--quiet", "--disable-plot-display",
            "--output-dir", str(out_dir), "--output-csv",
            "--as-of", AS_OF,
            "--replay-from", str(REPLAY_DIR),
            "port/port-1.toml",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )


@pytest.mark.integration
def test_missing_benchmark_file_errors_clearly(tmp_path: Path) -> None:
    """A benchmark_returns_file that doesn't exist must fail with a non-zero
    exit and a clear error, not a traceback or a silent wrong result."""
    cfg = tmp_path / "bad_bench.toml"
    cfg.write_text('benchmark_returns_file = "data/__no_such_benchmark__.csv"\n')
    result = _run_offline(cfg, tmp_path)
    assert result.returncode != 0, f"expected failure\nstdout:\n{result.stdout}"
    combined = (result.stdout + result.stderr).lower()
    assert "error" in combined
    assert "__no_such_benchmark__" in (result.stdout + result.stderr)


@pytest.mark.integration
def test_missing_risk_free_file_errors_clearly(tmp_path: Path) -> None:
    """A risk_free_rates_file that doesn't exist must fail cleanly (non-zero
    exit, clear error) rather than crashing mid-computation."""
    cfg = tmp_path / "bad_rf.toml"
    cfg.write_text('risk_free_rates_file = "data/__no_such_riskfree__.csv"\n')
    result = _run_offline(cfg, tmp_path)
    assert result.returncode != 0, f"expected failure\nstdout:\n{result.stdout}"
    assert "error" in (result.stdout + result.stderr).lower()
    assert "__no_such_riskfree__" in (result.stdout + result.stderr)


def _run_port1_offline(
    out_root: Path, extra_args: tuple[str, ...] = ()
) -> tuple[subprocess.CompletedProcess, Path]:
    """Run port-1 deterministically/offline (GOLDEN_CONFIG + --as-of +
    --replay-from), append ``extra_args``, return (process, output dir)."""
    out_dir = out_root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            PYTHON, "-m", "portfolioanalyzer.main",
            "--config", str(GOLDEN_CONFIG),
            "--quiet", "--disable-plot-display",
            "--output-dir", str(out_dir), "--output-csv",
            "--as-of", AS_OF,
            "--replay-from", str(REPLAY_DIR),
            *extra_args,
            "port/port-1.toml",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )
    return result, out_dir


def _drawdown_count(out_dir: Path) -> int:
    """The drawdowns-count column (index 7) from the one-row metrics CSV."""
    with (out_dir / "port-1.csv").open() as f:
        row = next(csv.reader(f))
    return int(row[7])


@pytest.mark.integration
def test_drawdown_threshold_honored_and_percent_scaled(tmp_path: Path) -> None:
    """End-to-end proof that ``--max-drawdown-threshold`` is consumed AND
    interpreted as a percent (the ``/100`` conversion).

    - At 3% (→ 0.03 fraction) real drawdowns are found. Were the flag
      uninterpreted, 3.0 would mean a 300% drop and find none; were the
      threshold hardcoded, the flag would be ignored.
    - At 80% (→ 0.80 fraction) nothing qualifies, so the count is 0. A
      hardcoded 0.05 would still report the 5% drawdowns here.
    Only "consumed and percent-scaled" satisfies both assertions.
    """
    res3, out3 = _run_port1_offline(tmp_path / "t3", ("--max-drawdown-threshold", "3.0"))
    assert res3.returncode == 0, f"stderr:\n{res3.stderr}"
    res80, out80 = _run_port1_offline(tmp_path / "t80", ("--max-drawdown-threshold", "80.0"))
    assert res80.returncode == 0, f"stderr:\n{res80.stderr}"

    count3 = _drawdown_count(out3)
    count80 = _drawdown_count(out80)
    assert count80 == 0, f"expected no >=80% drawdowns, got {count80}"
    assert count3 >= 1, f"expected >=3% drawdowns to be found, got {count3}"


@pytest.mark.integration
def test_garbage_benchmark_file_fails_cleanly(tmp_path: Path) -> None:
    """A well-formed path to a benchmark file with garbage contents (no
    recognizable date column) must fail with a clean, useful error and a
    non-zero exit — not a bare traceback or a silent wrong result."""
    bad_csv = tmp_path / "garbage_benchmark.csv"
    bad_csv.write_text("foo,bar\n1,2\n3,4\n")
    cfg = tmp_path / "bad_format.toml"
    cfg.write_text(f'benchmark_returns_file = "{bad_csv}"\n')
    result = _run_offline(cfg, tmp_path)
    assert result.returncode != 0, f"expected failure\nstdout:\n{result.stdout}"
    combined = result.stdout + result.stderr
    assert "Error:" in combined, "cli() should print a clean 'Error:' line"
    assert "date column" in combined.lower()


@pytest.mark.integration
def test_wrong_benchmark_date_format_fails_loudly(tmp_path: Path) -> None:
    """A benchmark CSV whose dates don't match the configured format fails
    fast with a clear error naming the format — deliberately NOT a silent
    warning that proceeds on mis-parsed dates, which would corrupt every
    benchmark-derived metric (block-by-default freshness philosophy)."""
    bench = tmp_path / "bench.csv"
    # ISO dates, but the config tells the loader to expect US m/d/Y.
    bench.write_text("Date,Close\n2024-01-01,100\n2024-01-02,101\n2024-01-03,102\n")
    cfg = tmp_path / "wrong_fmt.toml"
    cfg.write_text(
        f'benchmark_returns_file = "{bench}"\n'
        'benchmark_date_format = "%m/%d/%Y"\n'
    )
    result = _run_offline(cfg, tmp_path)
    assert result.returncode != 0, f"expected failure\nstdout:\n{result.stdout}"
    combined = (result.stdout + result.stderr).lower()
    assert "error:" in combined
    assert "date parsing failed" in combined or "format" in combined


@pytest.mark.integration
@pytest.mark.network
def test_full_run_produces_csv(tmp_path: Path) -> None:
    """End-to-end smoke: a real portfolio run produces the expected CSV file
    and exits cleanly. Numeric correctness is covered by the goldens.

    Pinned with ``--as-of`` so the freshness path neither fetches nor blocks
    on the reference feeds — this keeps the run from rewriting the tracked
    ``data/`` CSVs as a side effect (which would silently break the goldens)
    while still exercising the live mfapi.in NAV fetch (hence ``network``).
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = subprocess.run(
        [
            PYTHON,
            "-m", "portfolioanalyzer.main",
            "--config",
            str(GOLDEN_CONFIG),
            "--quiet",
            "--disable-plot-display",
            "--output-dir",
            str(out_dir),
            "--output-csv",
            "--as-of",
            "2026-06-13",
            "--metrics-method",
            "monthly",
            "--lookback",
            "5Y",
            "port/port-1.toml",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"main.py exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    csv_path = out_dir / "port-1.csv"
    assert csv_path.exists(), f"Expected {csv_path} to exist"
    assert csv_path.stat().st_size > 0, "CSV is empty"
