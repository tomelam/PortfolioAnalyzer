# PortfolioAnalyzer Kanban

Single source of truth for project maturity. Edit inline; commit alongside code changes.

The detailed plan lives at `/Users/tom/.claude/plans/concurrent-stargazing-shore.md`.

## Backlog

### Phase C — Golden-master safety net (all 3 portfolios DONE)
- [ ] Add `--today YYYY-MM-DD` flag to `main.py` so the safety net is deterministic without live mfapi drift; once it exists, tighten test tolerances to 1e-9
- [ ] Build a pickle-replay path (e.g. `main.py --replay-from tests/golden/port-X/pickles/`) so the test no longer needs network
- [ ] Record coverage baseline (after Phase D test additions)
- [x] **Daily-method Sharpe/Vol inflation on synthetic-CIV portfolios FIXED.** Root cause was *not* zero-return-days as suspected — it was `pd.concat(join="inner")` in `combined_civ_series` collapsing the portfolio CIV to the *intersection* of dates. With monthly-sampled gold or PPF present, that intersection was monthly, so applying `sqrt(252)` annualization yielded 10× inflated Vol. Fix: reindex every asset onto a common business-day calendar with ffill before joining. Result: port-mf-ppf-gold daily Sharpe 4.98 → 0.46 (matches monthly 0.43); port-everything daily Sharpe 7.33 → 0.76 (matches monthly 0.69); daily/monthly Vol now within ~1pp. TDD tests at `tests/unit/test_portfolio_civ_frequency.py`.
- [x] **`combined_civ_series` scale-mismatch bug FIXED.** Previously summed `asset.civ.value_series() * weight` without normalizing, so PPF (raw NAV ₹2566) dominated MFs (NAV ₹18) — contributing 95% of starting CIV despite 15% allocation. Now normalizes each asset's CIV to 1.0 at the common start before weighting. Two TDD tests pin the scale-invariance contract (`tests/unit/test_portfolio_civ_normalization.py`). All 6 goldens re-captured.
- [x] **12 skip-marked metric tests** unblocked by rewriting each to call `metrics.sharpe/sortino/volatility` directly (option b). Suite now 115 passed / 3 skipped (where the 3 are unrelated legacy items below).
- [x] **Reimplemented `max_drawdowns(threshold)`** — implemented as `metrics.max_drawdowns` (pure function), `TimeseriesReturn.max_drawdowns` delegates. Returns dicts with `start_date`, `trough_date`, `recovery_date`, `drawdown` (positive fraction), plus legacy aliases `depth_pct` (-percent × 100), `trough_value`, `recovery_value`. All 4 legacy tests unblocked.
- [x] **`test_calculate_portfolio_allocations` unblocked** — mock previously set `self.assets = {name: None}`, but `calculate_portfolio_allocations` now reads `asset.asset_allocation`. Rewrote the mock with `SimpleNamespace` assets carrying real allocation dicts.
- [x] **`test_stale_data_with_no_input_aborts` unblocked** — pinned time with `freezegun` to a Tuesday afternoon so the function's Sunday/Monday-before-9am skip doesn't bypass the gate.
- [x] **`test_get_aligned_portfolio_civs` unblocked** — rewrote the brittle pickled-golden comparison as two behavioral tests with mocked `requests.get`. Verify the contract (DataFrame structure, column-per-fund, date-intersection index, no NaN, dtype) on deterministic synthetic NAVs instead of byte-comparing against a stale historical run.

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
- [x] Extract NSI SCSS scraper → `scss_loader.py`, decomposed into pure `parse_scss_html` + network `fetch_scss_html` + composed `load_scss_interest_rates`. 8 tests using static HTML fixture, all in 0.5s.
- [x] Extract REC bond from `main.py` → `rec_bond_loader.py` honoring the TOML `coupon` field (was silently hardcoded 5.25%). 5 tests.
- [x] Unit tests for `sgb_loader.py` (added in commit 791d955; KANBAN was stale).
- [ ] Consolidate all `*_loader.py` files into a `loaders/` package as one big-bang rename commit.
- [ ] Consolidate the four `*timeseries*.py` files (timeseries.py 22.9 KB, timeseries_civ.py, asset_timeseries.py, portfolio_timeseries.py) into a `portfolio_analyzer/timeseries/` package
- [x] Test `TimeseriesCIV`, `TimeseriesReturn`, `AssetTimeseries`, `PortfolioTimeseries` independently — 11 tests in `tests/unit/test_timeseries_classes.py` covering class-level surface (constructors, validation, `combined_daily_returns`, hand-rolled `TimeseriesCIV.max_drawdowns`, `summary`). Coverage of the four files now 71–98% (was 23–81%).
- [x] Unit tests for `synthetic_civ.py` — 6 tests pin PPF monthly accrual, March yearly credit, mid-year rate changes, and Series/DataFrame input flexibility.
- [x] Unit tests for `civ_to_returns.py` — 6 tests including the round-trip identity (CIV → returns → cumprod ≈ normalized CIV).
- [ ] Replace dead/scattered metrics code with tested `metrics.py` (CAGR, vol, Sharpe, Sortino, alpha, beta, drawdowns)
- [ ] Reference: `~/Projects/PortfolioAnalyzer.attic/tmp-mar2025/test_metrics.py` has golden formulas
- [x] Decided: `portfolio_calculator.py` — KEEP. Two functions live (`calculate_portfolio_allocations`, `calculate_gains_cumulative`); dropped dead `calculate_gain_daily_portfolio_series` and its unused import in main.py, plus the commented-out duplicate header.
- [ ] Decide fate of `bond_calculators.py` (12.7 KB) — `calculate_variable_bond_cumulative_gain` is live (used by rec_bond_loader); the other 4 SGB-related funcs are unused but may inform Phase E live-gold port. Audit during Phase E.
- [x] Decided: `visualizer.py` — KEEP. `plot_cumulative_returns` + `print_major_drawdowns` both live in main.py. Matplotlib import cost is acceptable for now; tests use `--disable-plot-display`.
- [x] Reviewed `salvage/tmp3-uncommitted` branch (commit 9e899fb) — file-by-file: nothing cherry-picked. Findings:
  - `data_loader.py` (155 lines): ~90% ruff/black cosmetic; one substantive change (`load_index_data` calls `warn_if_stale(..., quiet=quiet)`) references an undefined `quiet` param — abandoned WIP.
  - `utils.py` (11 lines): 100% formatting.
  - `portfolio_timeseries.py` (21 lines): formatting plus DEBUG NaN-count prints in `from_multiple_nav_series` — noise.
  - `ppf_calculator.py` (26 lines): formatting plus `ppf_df.reindex(master_dates).ffill()` referencing an undefined `master_dates` — broken WIP.
  - `main.py` (2 lines): changes default `benchmark_date_format` from `%m/%d/%Y` to `%d-%m-%Y`. Current NIFTY CSV uses `MM/DD/YYYY`, so this change goes with a different data file we don't have.
  - Two CSV diffs: `INDIRLTLT01STM.csv` (1 line), `NIFTY ... CSV` (large sort/encoding change) — appear to be the user's local data refreshes; KANBAN already tracks data refresh as a separate item.
  - **Branch preserved** (not deleted) for the historical record; nothing actionable remains.
- [x] Decided: `config/` directory — KEEP. 18 files: `example_config.toml` (documents the schema) + `mid-cap_config.toml` (named portfolio config) are useful; the 16 CLI-flag-combo files (`no_output_csv-...toml`) are not referenced from code but document hand-test scenarios. Move to `attic/config-handtest-fixtures/` if/when pytest-parametrize replaces them; not blocking salvage.
- [x] Decided: `tests/data/*.pkl` — 5 of 7 are unused. Only `portfolio_civs.pkl` and `aligned_civs.pkl` are read (by `tests/test_alignment.py`). Others (`aligned_portfolio_civs.pkl`, `aligned_ppf_portfolio_civs.pkl`, `benchmark_data.pkl`, `benchmark_returns.pkl`, `benchmark_returns_series.pkl`) are written by main.py's `--save-golden-data` path but never read by any test. Safe to delete post-v0.1; left for now.
- [ ] Walk through `tests/TODO.md` checklist — deferred to Phase F. Most items are CLI-flag/TOML-override and failure-mode testing that overlaps with the planned `tests/integration/test_main_e2e.py`.

### Phase E — Salvage from old checkpoints
- [x] **Live-gold via yfinance: DROPPED.** User: "yfinance probably cannot be depended upon by a stable program." yfinance scrapes an undocumented Yahoo endpoint that breaks without warning. Keep the static monthly `data/gold_monthly_inr.csv` path; document the manual refresh procedure in `docs/DATA_REFRESH.md` (see Data freshness section). If a stable public gold-price API surfaces later, port it then — but not yfinance.
- [x] **Audit of `attic/tmp4-apr2025` complete.** `TimeseriesFrame` in tmp4 ≈ current `TimeseriesReturn` (the rename happened during the OOP rewrite). Substantively additional surface in tmp4 is *reporting utilities*, not math: `info_summary`, `describe_as_report`, `to_csv_report`, `to_latex_table`, `compare_to`, `as_rolling`, `align_with`, `clip_to_overlap`, `aligned_to`, `interpolated`, `plot_with`. The math (cagr/vol/sharpe/sortino/max_drawdown/alpha/beta) is equivalent to current `metrics.py`. One semantic difference in tmp4: it divides annual `risk_free_rate` by `periods_per_year` internally; current pipeline does the conversion correctly upstream in `main.py` (geometric per-period rate), so no port needed. **Nothing blocks v0.1-salvage.** Reporting utilities backlogged below.
- [x] **Audit of `attic/tmp4-apr2025/bonus/` complete.** 4 shell helpers (`plot_all.sh`, `run_all_configs.sh`, `run_all_metrics_to_csv.sh`, `single-asset-type.sh`) and one diagnostic script (`ppf_annualized_interest_rate.py`). Useful as references but non-blocking; backlogged for post-v0.1.

### Phase E — backlog (post-v0.1 only)
- [ ] Port tmp4's reporting helpers to current `TimeseriesReturn` if CSV/LaTeX export is wanted: `info_summary`, `describe_as_report`, `to_csv_report`, `to_latex_table`, `compare_to`, `as_rolling`.
- [ ] Port tmp4's series-alignment helpers if cross-asset analysis grows beyond `combined_civ_series`: `align_with`, `clip_to_overlap`, `aligned_to`, `interpolated`.
- [ ] Port tmp4 `bonus/` shell helpers (plot_all / run_all_configs / run_all_metrics_to_csv / single-asset-type) into a `scripts/` directory if the user wants CLI orchestration.
- [ ] Port tmp4 `bonus/ppf_annualized_interest_rate.py` as an analysis tool under `scripts/`.

### Phase F — Integration + docs
- [x] `tests/integration/test_main_e2e.py` — 3 subprocess tests: --help exits clean; missing TOML → non-zero exit; full-run smoke produces CSV. Integration-marked; one is `network`-marked.
- [x] `docs/ARCHITECTURE.md` — module map + pipeline diagram + design decisions (pure-function math layer; unit-free daily-calendar CIV; synthetic CIVs; risk-free rate convention).
- [x] `docs/TESTING.md` — three test tiers, marker matrix, golden tolerances, regeneration recipe, why-CSV-not-pickle.
- [x] `docs/CONTRIBUTING.md` — TDD-first rule, decomposition guidance, pre-commit, KANBAN expectation, anti-patterns (yfinance, hypothetical abstractions, what-comments).
- [ ] Push, tag `v0.1-salvage` (user-driven; awaiting go-ahead)
- [ ] Decide whether to make GitHub repo private during salvage (user decision)

### Data freshness (separate from salvage; user responsibility)
- [ ] **Refresh stale data files.** As of 2026-06-14, the canonical data files in `data/` are 13+ months out of date and the code's own staleness checks refuse to run against them:
  - `data/NIFTY Total Returns Historical Data.csv` — last date 2025-05-02 (NIFTY Total Returns Index)
  - `data/India 10-Year Bond Yield Historical Data.csv` — last date 2025-03-28 (risk-free rate)
- [ ] These are manual downloads (investing.com / similar). Document the refresh procedure in `docs/DATA_CLEANING.txt` or a new `docs/DATA_REFRESH.md`.
- [ ] Golden-master tests captured before refresh (Phase C of this salvage) are pinned to the stale data via `tests/fixtures/golden_master_config.toml` (`skip_age_check = true`, `max_riskfree_delay = 99999`). **After refreshing the CSVs, re-capture the goldens** — the existing ones will no longer match.
- [ ] Consider automating the NIFTY + bond-yield refresh (scraper or scheduled fetch) so the staleness gate becomes early-warning, not a hard blocker, for routine runs.
- [ ] Also refresh any other under-watched data sources used by `data_loader.py`: `ppf_interest_rates.csv`, REC bond coupon table, SCSS rate table (if file-backed) — audit `data/` for last-modified dates.

### Hygiene / tech debt
- [x] **Plot ↔ metrics consistency** (Phase F follow-up). `main.py` fed the plot with `cumprod(1 + combined_daily_returns)` while the metrics box used `combined_civ_series`. For mixed-frequency portfolios (daily MFs + monthly gold), the two diverged because weighted-sum-of-asset-returns ≠ return-of-weighted-sum, and the legacy `combined_daily_returns` inner-joined to the monthly intersection. Fixed `main.py` to feed `portfolio_civ_series.series` directly to the plotter. Three TDD tests pin the contract (`tests/unit/test_plot_metric_consistency.py`); the third test documents the *reason* the old path was wrong and will fail loudly if `combined_daily_returns` is ever independently re-aligned.
- [ ] **Portfolio CIV truncates at the earliest-ending asset's last date** — surfaced 2026-06-16 while verifying the plot fix. `port-everything` ends 2024-02-21 because the SGB CSV stops being updated there, so `min(ends)` in `combined_civ_series` truncates the whole portfolio CIV ~16 months short of "today". Either: (a) the SGB data file needs a refresh (data freshness item), (b) `combined_civ_series` should ffill-extend assets past their last observation rather than hard-truncate, or (c) main.py should warn loudly when this happens. Decide which; the user-visible symptom is "plot ends 16 months ago even though benchmark continues to today."
- [x] **sgb_loader `dayfirst=True` warning** (surfaced during the post-v0.1 smoke run) — fixed 2026-06-16. Real data and the test fixture now share one canonical format (`YYYY-MM-DD`); the loader uses an explicit `format="%Y-%m-%d"`. (First attempt used `format="mixed"` to bridge a fixture/data format mismatch; user pushed back — fixture re-encoded as ISO is the cleaner answer.)
- [x] **Two redundant `warnings.warn` calls** in `tests/test_timeseries.py::test_alpha_regression_known_series` and `::test_beta_regression_known_series` removed. The docstrings already say "minimal coverage"; the warn() call duplicated that information into stderr on every run. Full suite now passes under `pytest -W error` (151/0/0).
- [x] **Pytest now treats `DeprecationWarning` / `PendingDeprecationWarning` / `FutureWarning` as test failures** (`pyproject.toml` `filterwarnings = ["error::..."]`). Catches stdlib + pandas/numpy "behavior will change in vN.M" notices automatically — they can't quietly accumulate. Verified: 151/0/0 today, and all 6 live `main.py` invocations clean under `python -W error::DeprecationWarning -W error::FutureWarning`.
- [x] **Dependabot configured** for `github-actions` and `pip` (weekly, 5 PRs max each). Without it, deprecation annotations like the Node-20 one accumulate silently until the runtime is removed and CI hard-fails. PRs go through CI like any other change.
- [x] **GitHub Actions Node-20 deprecation fixed** — `actions/checkout@v4 → @v5`, `actions/setup-python@v5 → @v6`. Both newer majors ship Node 24.
- [ ] Re-enable `SIM` (flake8-simplify) lint family disabled during the v0.1 push; ~15 style hints to triage (mostly ternaries vs if/else, nested-with, dict.keys()).
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

- [ ] Phase F: integration test, docs, tag `v0.1-salvage`.

## Done

- [x] Phase 1 exploration: mapped four checkpoints across `~/Projects/PortfolioAnalyzer{,-tmp/{tmp,tmp3,tmp4}}/`
- [x] Decision: promote tmp3 as canonical (Jun 7 2025, commit d37278b, branch main, GitHub remote)
- [x] Written plan committed (`/Users/tom/.claude/plans/concurrent-stargazing-shore.md`)
- [x] **Phase B: Foundation scaffolding** — pyproject.toml, CI, pre-commit, KANBAN, docs/ conversions all in place.
- [x] **Phase C: Golden-master safety net** — 3 portfolios × 2 methods, 6 goldens captured + re-captured after the two CIV fixes. All green.
- [x] **Phase E: salvage audit (2026-06-16).** Closed.
  - yfinance live-gold path dropped per user feedback (yfinance unreliable).
  - tmp4-apr2025 reviewed: math equivalent to current `metrics.py`; reporting/alignment utilities backlogged for post-v0.1; no blockers.
  - `bonus/` shell helpers + diagnostic script backlogged; non-blocking.
- [x] **Phase D: Decomposition + TDD cycles (2026-06-15 → 2026-06-16).** Closed.
  - Two portfolio-CIV bugs fixed (scale-invariance, daily-frequency consistency); daily ≡ monthly Sharpe/Vol now agree to within ~1pp on every golden.
  - Pure-function `metrics.py` extracted; all `TimeseriesReturn` metric methods delegate.
  - Loaders extracted into 7 standalone modules (mutual_fund, ppf, benchmark, risk_free, scss, rec_bond, sgb) with unit tests each.
  - 22 of 22 originally-broken legacy tests unblocked.
  - Suite: 142 pass / 0 skip / 6 goldens green (was 19+ skips at Phase D start).
  - Tests added: synthetic_civ (6), civ_to_returns (6), Timeseries classes (11), PortfolioCIV normalization (2) + frequency (2), get-aligned-portfolio-civs behavioral (2), allocations + staleness unblocks (2).
  - All "decide fate" investigations resolved (portfolio_calculator, visualizer kept; bond_calculators audit deferred; salvage branch reviewed and closed; config/ kept; *.pkl audited).
- [x] Phase A: Repo consolidation
  - tmp3 → `~/Projects/PortfolioAnalyzer/`
  - Old Feb tree, tmp/Mar, tmp4/Apr + `.bak`, chatgpt-amfi → `~/Projects/PortfolioAnalyzer.attic/` (chmod read-only)
  - Untracked experimental dirs (RBI/, JUNK/, SAVE/, CSV/, METRICS/, code.zip) → `attic/tmp3-untracked/`
  - Dirty working-tree files preserved on `salvage/tmp3-uncommitted` branch (commit 9e899fb)
  - Verified: `origin/main` = `d37278b`, `git status` clean of these, branch = `main`
