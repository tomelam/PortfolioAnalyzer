# PortfolioAnalyzer Kanban

Single source of truth for project maturity. Edit inline; commit alongside code changes.

The detailed plan lives at `/Users/tom/.claude/plans/concurrent-stargazing-shore.md`.

## Backlog

### Phase C — Golden-master safety net
- [ ] Pick 3–4 representative configs from `port/` (suggested: `port-1.toml` simple, `port-everything.toml` broad coverage, `port-mf-ppf-gold.toml` synthetic CIVs, `port-gold.toml` gold-only)
- [ ] Capture golden outputs via `--save-golden-data` for each (daily + monthly metrics methods)
- [ ] Write `tests/integration/test_golden_master.py` with `freezegun` date pinning
- [ ] Record coverage baseline in this file

### Phase D — Decomposition + TDD cycles
- [ ] Create `portfolio_analyzer/loaders/` package; move `gold_loader.py`, `sgb_loader.py` into it
- [ ] Extract per-asset loaders from `data_loader.py` (32.5 KB): mutual fund, NIFTY benchmark, FRED risk-free, NSI SCSS, REC bond → `loaders/{mutual_fund,nifty,risk_free,scss,rec_bond}.py`
- [ ] Each loader: TDD-first — mock HTTP with `responses`, capture fixtures under `tests/fixtures/api_responses/`
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
