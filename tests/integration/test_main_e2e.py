"""End-to-end CLI integration tests.

These tests invoke ``main.py`` via subprocess to catch regressions at
the argparse / import / entry-point boundary that the unit suite and
golden-master numeric tests can miss. Intentionally redundant with
``test_golden_master.py`` at the CLI surface — different failure modes.

Network-free tests (help, missing-file) run by default. The full
smoke run that hits mfapi.in is marked ``network``.
"""

from __future__ import annotations

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
        [PYTHON, "main.py", "--help"],
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
            "main.py",
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
        [PYTHON, "main.py", "--help"],
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
        [PYTHON, "main.py", removed_flag, "port/port-1.toml"],
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
            PYTHON, "main.py",
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
            "main.py",
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
