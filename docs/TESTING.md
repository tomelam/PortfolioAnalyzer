# Testing

Three test tiers, separated by pytest markers. The default `pytest` run is
the unit tier only — fast (<3 s) and network-free.

## Tiers

| Tier | Marker | What it covers | Network? |
|---|---|---|---|
| Unit | *(unmarked)* | Pure-function `metrics.py`, loader parsing, Timeseries class surface, synthetic CIVs | No |
| Integration | `integration` | CLI subprocess invocation, missing-file error path, end-to-end smoke | Some `network` |
| Golden | `golden` | Full pipeline run vs. captured CSV outputs for 3 portfolios × 2 methods | No (replay fixtures) |

```bash
# Default — unit + golden tiers; fast and network-free.
pytest

# Just the goldens.
pytest -m golden tests/integration/

# Include the remaining live-network tests too.
pytest -m ""
```

The default `addopts` in `pyproject.toml` is `-m 'not network'`, so CI
and the routine local loop never hit a live endpoint. The goldens run in
this default loop: they replay committed fixtures (see below) instead of
fetching from mfapi.in / nsiindia.gov.in.

## Unit tests

Pure-function tests under `tests/unit/`. Notable files:

- `test_metrics.py` — 20 hand-verified tests of CAGR/vol/Sharpe/Sortino/drawdowns
- `test_portfolio_civ_normalization.py` — pins scale-invariance of `combined_civ_series`
- `test_portfolio_civ_frequency.py` — pins the common-bday-calendar contract
- `test_timeseries_classes.py` — class-level surface (constructors, validation, hand-rolled methods)
- `test_synthetic_civ.py` — PPF monthly accrual + yearly credit
- `test_civ_to_returns.py` — CIV → returns round-trip identity
- `loaders/test_*.py` — one per loader, with `responses`/`monkeypatch` API mocks

Run a single test:

```bash
pytest tests/unit/test_metrics.py::test_sharpe_exactly_one -v
```

## Golden-master tests

Captured 2026-06-14 (re-captured 2026-06-15 and -16 after two CIV bug fixes).
Cover three portfolios:

- `port-1`: 5 mutual funds + NIFTY benchmark
- `port-mf-ppf-gold`: 5 MFs + PPF + physical gold
- `port-everything`: 5 MFs + PPF + Gold + SGB + SCSS + REC Bond

Each runs in both `--metrics-method daily` and `--metrics-method monthly` →
6 goldens total. Runs are deterministic — pinned with `--as-of 2026-06-13`
and fed from committed replay fixtures — so every column reproduces exactly:

| Column | Tolerance |
|---|---|
| CAGR, Vol, Sharpe, Sortino, Alpha, Beta, Max Drawdown % | `< 1e-9` |
| Drawdown count, Max DD start, drawdown days, recovery days | exact |

### Offline replay

The goldens read NAV/SCSS data from fixtures under `tests/golden/replay/`
(`navs/<fund>.csv` per mutual fund + `scss_nsi.html`) via main.py's
`--replay-from DIR` flag, so no run touches the network. Fixtures are
captured with the sibling `--save-replay DIR` flag during a live run.
Combined with `--as-of`, this is what makes the 1e-9 tolerances valid.

### Regenerating goldens

Don't regenerate casually — the captured numbers are the regression net.
Legitimate reasons to regenerate:

1. A deliberate, correct change to the math (e.g., the two Phase D CIV fixes).
2. Refreshing the underlying data files (see `KANBAN.md` → Data freshness).

To regenerate, refresh the replay fixtures from live data first (one
network run — any portfolio covering all funds + SCSS works), then
re-capture the goldens offline from those fixtures:

```bash
# 1. Refresh replay fixtures from the network (once).
./venv/bin/python main.py \
  --config tests/fixtures/golden_master_config.toml \
  --quiet --disable-plot-display --output-dir /tmp/cap --output-csv \
  --metrics-method daily --lookback 5Y --as-of 2026-06-13 \
  --save-replay tests/golden/replay \
  port/port-everything.toml

# 2. Re-capture goldens for each portfolio × method (offline via replay).
for p in port-1 port-mf-ppf-gold port-everything; do
  for m in daily monthly; do
    ./venv/bin/python main.py \
      --config tests/fixtures/golden_master_config.toml \
      --quiet --disable-plot-display \
      --output-dir tests/golden/$p/$m \
      --output-csv \
      --metrics-method $m \
      --lookback 5Y \
      --as-of 2026-06-13 \
      --replay-from tests/golden/replay \
      port/$p.toml
  done
done

pytest -m golden tests/integration/ -v
```

Inspect `tests/golden/$p/$m/$p.csv` to confirm the new numbers are what you
expect, then commit with a message that explains *why* the numbers changed.
Bump `--as-of` only when you re-capture, never independently.

### Why a CSV, not a pickle

The legacy version used pickled DataFrames. They're brittle across pandas
versions and don't diff cleanly in PRs. CSV is more verbose but reviewable
and stable. The replay fixtures follow the same principle — NAV history is
stored as per-fund CSV, not pickle.

## Network-blocking fixture

`tests/conftest.py` autouses a fixture that fails any `requests.get` call
in tests that aren't marked `network`. This prevents accidental live API
hits in routine runs.

## Coverage

CI gates at `--cov-fail-under=85` (`.github/workflows/ci.yml`, `pytest
--cov=. --cov-report=term-missing`).

### Baseline (2026-06-17, 287 passed)

**TOTAL: 87%**. Reproduce with:

```bash
pytest --cov=. --cov-report=term-missing
```

Notable modules:

| Module | Cover | Note |
|---|---|---|
| `data_loader.py` | 99% | validation/loader branches in `tests/unit/test_data_loader.py` |
| `timeseries/returns.py` | 99% | dead helpers parked in `attic/`; live surface in `tests/unit/test_timeseries_returns.py` |
| `loaders/data_update.py` | 93% | auto-update fetchers/refresh; mocked unit tests + live (network) |
| `metrics.py` | 94% | core formulas, well covered |
| `timeseries/` (civ, asset, classes) | 84–98% | |
| `loaders/*` | 79–100% | parser layers unit-tested |
| `visualizer.py` | 71% | plotting smoke tests (Agg) in `tests/unit/test_visualizer.py` |
| `main.py` | 1% | **measurement artifact**: exercised end-to-end by the golden + e2e tests, but those run main.py in a *subprocess*, so coverage isn't attributed here |

`main.py`'s 1% is misleading (subprocess execution), not untested behavior —
it's the only large remaining "gap". `attic/` (parked, unwired code,
including the formerly-dead `ppf_calculator.py` and
`extract_gold_inr_from_excel.py`) is excluded from the measurement via
`[tool.coverage.run] omit` in `pyproject.toml`.
