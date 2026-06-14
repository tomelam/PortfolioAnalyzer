# PortfolioAnalyzer Kanban

Single source of truth for project maturity. Edit inline; commit alongside code changes.

The detailed plan lives at `/Users/tom/.claude/plans/concurrent-stargazing-shore.md`.

## Backlog

### Phase C — Golden-master safety net (all 3 portfolios DONE)
- [ ] Add `--today YYYY-MM-DD` flag to `main.py` so the safety net is deterministic without live mfapi drift; once it exists, tighten test tolerances to 1e-9
- [ ] Build a pickle-replay path (e.g. `main.py --replay-from tests/golden/port-X/pickles/`) so the test no longer needs network
- [ ] Record coverage baseline (after Phase D test additions)
- [ ] Investigate implausibly large daily-method Sharpe/Vol on synthetic-CIV portfolios — port-mf-ppf-gold daily Sharpe=7.25, Vol=52%; port-everything daily Sharpe=4.99, Vol=37.76%. Monthly figures look reasonable (0.96 and 0.27). Likely cause: forward-filled synthetic CIVs produce zero-return days that distort the variance denominator. Phase D investigation; the golden currently pins this (incorrect-looking) behavior.

### Bugs FIXED during Phase C capture
- [x] **main.py line 359 stale name `portfolio` → `portfolio_ts`** — silent failure of `--save-golden-data` pickle dump
- [x] **gold_loader returned DataFrame; downstream expected Series.** Loader now returns `pd.Series` named `"price"`; main.py call-site simplified. Unit tests at `tests/unit/loaders/test_gold.py` pin the contract
- [x] **PPF `load_ppf_civ` returned monthly series → NaN-laden after daily reindex.** Now reindexes to daily, forward-fills, extends to today carrying the most recent rate. Unit tests at `tests/unit/loaders/test_ppf.py`
- [x] **`synthetic_civ.py:70` deprecated chained-inplace `.fillna(method='ffill', inplace=True)`** — silent no-op since pandas 2.x. Replaced with assignment-based `.ffill()`
- [x] **`synthetic_civ.py:43` deprecated `freq="M"`** — replaced with `"ME"`

### Phase D — Decomposition + TDD cycles
- [ ] Create `portfolio_analyzer/loaders/` package; move all `*_loader.py` files into it as a single big-bang rename commit
- [x] Extract `fetch_navs_of_mutual_fund` → `mutual_fund_loader.fetch_navs` with 7 unit tests (DataFrame contract, nav float, sorted DatetimeIndex, dayfirst parsing, retry-on-transient, exhausted-retries error, missing-data error). `data_loader` re-exports for compat.
- [x] Extract `load_ppf_interest_rates` + `load_ppf_civ` → `ppf_loader.py` with 3 added rate-CSV contract tests. Total PPF tests: 8.
- [x] Extract `load_timeseries_csv` → `benchmark_loader.py` with 6 tests. Deleted dead `load_index_data` and its unused main.py import.
- [x] Extract `fetch_and_standardize_risk_free_rates` + `align_dynamic_risk_free_rates` → `risk_free_loader.py` with 5 tests (Series return, percent→decimal, staleness, weekend-gap interpolation, alignment).
- [ ] Extract remaining loaders from `data_loader.py`: NSI SCSS (`load_scss_interest_rates` — 102-line web scrape), REC bond (currently inline in `main.py`).
- [ ] TDD-first contract for each — for scrapers/APIs use mocked HTML/JSON fixtures under `tests/fixtures/api_responses/`
- [ ] Consolidate the four `*timeseries*.py` files (timeseries.py 22.9 KB, timeseries_civ.py, asset_timeseries.py, portfolio_timeseries.py) into a `portfolio_analyzer/timeseries/` package
- [ ] Test `TimeseriesCIV`, `TimeseriesReturn`, `AssetTimeseries`, `PortfolioTimeseries` independently
- [ ] Unit tests for `synthetic_civ.py` (PPF/SCSS interest-rate compounding)
- [ ] Unit tests for `civ_to_returns.py` (CIV→returns round-trip identity)
- [ ] Replace dead/scattered metrics code with tested `metrics.py` (CAGR, vol, Sharpe, Sortino, alpha, beta, drawdowns)
- [ ] Reference: `~/Projects/PortfolioAnalyzer.attic/tmp-mar2025/test_metrics.py` has golden formulas
- [ ] Decide fate of `portfolio_calculator.py` (4.2 KB) — appears to overlap with `portfolio_timeseries.py`; possibly dead
- [ ] Decide fate of `bond_calculators.py` (12.7 KB) — verify still used
- [ ] Decide fate of `visualizer.py` (10.9 KB) — split from analysis if heavyweight matplotlib import slows tests
- [ ] Review `salvage/tmp3-uncommitted` branch (commit 9e899fb) file-by-file; cherry-pick or discard
- [ ] Decide fate of `config/` directory (17 CLI-flag-combo test fixtures) — keep, restructure, or rely on pytest parametrize
- [ ] Decide fate of `tests/data/*.pkl` pickled fixtures — replace with on-the-fly construction or keep with version pin
- [ ] Walk through `tests/TODO.md` checklist and convert each item to a real test

### Phase E — Salvage from old checkpoints
- [ ] Port `fetch_gold_spot.py` live-gold from `~/Projects/PortfolioAnalyzer.attic/feb2025/` into `gold_loader.py` behind a `live = true` TOML flag (opt-in)
- [ ] Add `with_live_gold` golden fixture (yfinance mocked)
- [ ] Quick `diff -r` between attic/tmp4-apr2025 and current repo: anything `TimeseriesFrame` variant worth porting?

### Phase F — Integration + docs
- [ ] `tests/integration/test_main_e2e.py` — subprocess invocation of CLI
- [ ] `docs/ARCHITECTURE.md` (new) — module-level diagram
- [ ] `docs/TESTING.md` (new) — golden-master rationale and how-to-regenerate procedure
- [ ] `docs/CONTRIBUTING.md` (new) — TDD-first rule, pre-commit, KANBAN expectation
- [ ] Push, tag `v0.1-salvage`
- [ ] Decide whether to make GitHub repo private during salvage

### Data freshness (separate from salvage; user responsibility)
- [ ] **Refresh stale data files.** As of 2026-06-14, the canonical data files in `data/` are 13+ months out of date and the code's own staleness checks refuse to run against them:
  - `data/NIFTY Total Returns Historical Data.csv` — last date 2025-05-02 (NIFTY Total Returns Index)
  - `data/India 10-Year Bond Yield Historical Data.csv` — last date 2025-03-28 (risk-free rate)
- [ ] These are manual downloads (investing.com / similar). Document the refresh procedure in `docs/DATA_CLEANING.txt` or a new `docs/DATA_REFRESH.md`.
- [ ] Golden-master tests captured before refresh (Phase C of this salvage) are pinned to the stale data via `tests/fixtures/golden_master_config.toml` (`skip_age_check = true`, `max_riskfree_delay = 99999`). **After refreshing the CSVs, re-capture the goldens** — the existing ones will no longer match.
- [ ] Consider automating the NIFTY + bond-yield refresh (scraper or scheduled fetch) so the staleness gate becomes early-warning, not a hard blocker, for routine runs.
- [ ] Also refresh any other under-watched data sources used by `data_loader.py`: `ppf_interest_rates.csv`, REC bond coupon table, SCSS rate table (if file-backed) — audit `data/` for last-modified dates.

### Hygiene / tech debt
- [ ] Replace deprecated `.fillna(method='ffill')` (if still present) with `.ffill()`
- [ ] Add type hints incrementally; switch `ignore_missing_imports` off per-module
- [ ] Raise CI coverage gate from 20% → 40% → 70%
- [ ] Decide fate of `defunct_feature_var_rate_bonds` and `defunct_main` branches (likely delete after v0.1)
- [ ] Decide fate of `port/` directory naming — rename to `portfolios/` for clarity?
- [ ] Investigate `outputs/port-*/` cached runs — keep as canonical examples or move to attic?
- [ ] Remove `Makefile~` editor backup
- [ ] Add `.env.example` if FRED API tokens are needed
- [ ] Consider relocating `docs/*.pdf` originals to a `docs/sources/` subdirectory once `.md` sidecars exist

## In Progress

- [ ] Phase B: Foundation scaffolding (pyproject, CI, pre-commit, KANBAN, docs/ conversions)

## Done

- [x] Phase 1 exploration: mapped four checkpoints across `~/Projects/PortfolioAnalyzer{,-tmp/{tmp,tmp3,tmp4}}/`
- [x] Decision: promote tmp3 as canonical (Jun 7 2025, commit d37278b, branch main, GitHub remote)
- [x] Written plan committed (`/Users/tom/.claude/plans/concurrent-stargazing-shore.md`)
- [x] Phase A: Repo consolidation
  - tmp3 → `~/Projects/PortfolioAnalyzer/`
  - Old Feb tree, tmp/Mar, tmp4/Apr + `.bak`, chatgpt-amfi → `~/Projects/PortfolioAnalyzer.attic/` (chmod read-only)
  - Untracked experimental dirs (RBI/, JUNK/, SAVE/, CSV/, METRICS/, code.zip) → `attic/tmp3-untracked/`
  - Dirty working-tree files preserved on `salvage/tmp3-uncommitted` branch (commit 9e899fb)
  - Verified: `origin/main` = `d37278b`, `git status` clean of these, branch = `main`
