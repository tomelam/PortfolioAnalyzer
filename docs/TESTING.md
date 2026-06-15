# Testing

Three test tiers, separated by pytest markers. The default `pytest` run is
the unit tier only — fast (<3 s) and network-free.

## Tiers

| Tier | Marker | What it covers | Network? |
|---|---|---|---|
| Unit | *(unmarked)* | Pure-function `metrics.py`, loader parsing, Timeseries class surface, synthetic CIVs | No |
| Integration | `integration` | CLI subprocess invocation, missing-file error path, end-to-end smoke | Some `network` |
| Golden | `golden` + `network` | Full pipeline run vs. captured CSV outputs for 3 portfolios × 2 methods | Yes (mfapi.in) |

```bash
# Default — fast unit suite only.
pytest

# Full local run including integration + goldens.
pytest -m "not network or network"   # or simply: pytest -m ""

# Just the goldens.
pytest -m "golden or network" tests/integration/
```

The default `addopts` in `pyproject.toml` is `-m 'not network'`, so CI
and the routine local loop never hit the live mfapi.in endpoint.

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
6 goldens total. Numeric drift tolerances:

| Column | Tolerance | Anchor |
|---|---|---|
| CAGR, Vol, Sharpe, Sortino, Alpha, Beta | `max(0.5, expected · 5%)` | drift acceptable |
| Drawdown count | exact | anchored |
| Max Drawdown % | < 0.2 absolute | anchored |
| Max DD start, drawdown days, recovery days | exact | anchored |

Tolerance scales with magnitude to accommodate live mfapi.in NAV drift over
~a week.

### Regenerating goldens

Don't regenerate casually — the captured numbers are the regression net.
Legitimate reasons to regenerate:

1. A deliberate, correct change to the math (e.g., the two Phase D CIV fixes).
2. Refreshing the underlying data files (see `KANBAN.md` → Data freshness).

To regenerate:

```bash
# For each portfolio × method:
for p in port-1 port-mf-ppf-gold port-everything; do
  for m in daily monthly; do
    ./venv/bin/python main.py \
      --config tests/fixtures/golden_master_config.toml \
      --quiet --disable-plot-display \
      --output-dir tests/golden/$p/$m \
      --output-csv \
      --metrics-method $m \
      --lookback 5Y \
      port/$p.toml
  done
done

pytest -m "golden or network" -v
```

Inspect `tests/golden/$p/$m/$p.csv` to confirm the new numbers are what you
expect, then commit with a message that explains *why* the numbers changed.

### Why a CSV, not a pickle

The legacy version used pickled DataFrames. They're brittle across pandas
versions and don't diff cleanly in PRs. CSV is more verbose but reviewable
and stable.

## Network-blocking fixture

`tests/conftest.py` autouses a fixture that fails any `requests.get` call
in tests that aren't marked `network`. This prevents accidental live API
hits in routine runs.

## Coverage

Tracked but not gated. The Timeseries family was 23–81% covered at Phase D
start; it's now 71–98%. The CI gate is set conservatively at 20% during
salvage; the ratchet path (40 → 70 → 85%) is in `KANBAN.md`.
