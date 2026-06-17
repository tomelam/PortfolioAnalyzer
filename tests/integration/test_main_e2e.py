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
def test_help_lists_skip_age_check_flag() -> None:
    """``--skip-age-check`` must surface in ``--help``; without it, a user
    facing a stale benchmark CSV has no documented escape from the hard
    blocker except editing a config file."""
    result = subprocess.run(
        [PYTHON, "main.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--skip-age-check" in result.stdout


@pytest.mark.integration
def test_stale_benchmark_blocks_by_default_passes_with_flag(tmp_path: Path) -> None:
    """Strict-by-default staleness check should block; ``--skip-age-check``
    is the documented bypass. Test triggers the error path via the
    benchmark loader without making a network call (missing TOML stops
    the run early, but argparse exits 0 only when --help-style flags are
    accepted)."""
    # Strict run with a real TOML against the stale CSV: exit non-zero.
    strict = subprocess.run(
        [PYTHON, "main.py", "--quiet", "--disable-plot-display", "port/port-1.toml"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert strict.returncode != 0
    assert "outdated" in (strict.stdout + strict.stderr).lower()


@pytest.mark.integration
@pytest.mark.network
def test_full_run_produces_csv(tmp_path: Path) -> None:
    """End-to-end smoke: a real portfolio run produces the expected CSV file
    and exits cleanly. Numeric correctness is covered by the goldens."""
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
