# PortfolioAnalyzer Kanban

Single source of truth for project maturity. Edit inline; commit alongside code changes.

The detailed plan lives at `/Users/tom/.claude/plans/concurrent-stargazing-shore.md`.

## Backlog

### Post-v0.1 product improvements (user-prioritized 2026-06-17)

Grouped by impact. Top items meaningfully change what PortfolioAnalyzer does
or how trustworthy its output is; bottom items are hygiene/cleanup.

#### A. Output / reporting enhancements

- [x] **Drawdown table sibling CSV** (2026-06-17, cycle 2). `main.py` writes
  `<portfolio>.drawdowns.csv` alongside the metrics CSV: one row per
  recovered drawdown plus the final unrecovered one if any. Columns:
  `start_date, trough_date, recovery_date, depth_pct, drawdown_days,
  recovery_days`. `depth_pct` uses the negative-percent form (e.g.
  `-19.23`) matching the stdout summary. 5 TDD unit tests in
  `drawdowns_csv.py`.

- [x] **Fund inauguration + DEFUNCT status** (2026-06-17, cycle 3). Surfaced
  in three places: plot's allocation table gains "Inaugurated" and "Closed"
  columns; sibling `<portfolio>.assets.csv` with one row per asset
  (`asset_type, asset_name, allocation, inauguration_date, last_nav_date,
  status`); run-time stdout warning for every DEFUNCT fund. A fund is
  DEFUNCT if its most recent NAV is older than 30 days (parameterizable).
  Implemented as pure-function helpers in `fund_lifecycle.py` with 7 TDD
  tests. Re-uses already-fetched NAV DataFrames — no extra mfapi.in
  round-trips.

#### B. Determinism / trustworthiness

- [x] **Stale NIFTY benchmark is a hard blocker — add a bypass flag**
  (2026-06-17, cycle 10). Added `--skip-age-check` CLI flag. When
  active it bypasses both the benchmark and risk-free CSV staleness
  gates (the latter via auto-bumping `max_riskfree_delay` to 99999
  unless the user explicitly set it). main.py prints a one-line
  warning ("⚠️ --skip-age-check active …") so the bypass is never
  silent. 2 integration tests: --help lists the flag; strict-by-default
  still blocks with "outdated" on the stale CSV. Default remains
  strict so silent drift can't creep in.

- [ ] **Auto-update the benchmark CSV.** `data/NIFTY Total Returns Historical
  Data.csv` is currently a manual investing.com download (last refreshed
  2025-05-02; 13+ months stale as of today). Build a fetcher that pulls
  fresh NIFTY 50 TRI data from a stable source — niftyindices.com publishes
  daily total-return CSVs at
  `https://www.niftyindices.com/IndexConstituent/ind_close_all_<DDMMYYYY>.csv`
  or via their historical-data endpoint. Where a stable, no-auth source
  exists for other benchmarks (e.g. India 10Y bond yield via FRED's
  `INDIRLTLT01STM`), auto-refresh those too. Per-source rule: scheduled
  background refresh (cron-able), with last-modified-date stamps so the
  staleness gate can become "early warning, not a hard blocker." See
  also the existing "Refresh stale data files" item under *Data freshness*.
  User intent (2026-06-17): the "up to date benchmark data should be
  auto-updated" path is the default; the bypass-flag item above is the
  fallback for when the auto-update source is itself unavailable.

- [ ] **Add `--today YYYY-MM-DD` flag** so the safety-net tests are
  deterministic without live mfapi drift. Once it exists, tighten golden
  tolerances to 1e-9 (currently 5% relative). *(Already on Phase C
  backlog; restated here for visibility.)*

- [ ] **Build a pickle-replay path** (`main.py --replay-from
  tests/golden/port-X/pickles/`). Removes the network dependency from
  golden tests entirely. *(Already on Phase C backlog.)*

#### C. Correctness bugs surfaced but not fixed

- [x] **Fixed `PortfolioTimeseries.__init__` weight-sum check** (2026-06-17,
  cycle 1). Strict `!= 1` replaced with `abs(total_weight - 1) > 0.01`.
  All 6 previously-failing portfolios now render. Two new unit tests:
  one pinning that FP-rounding portfolios are accepted, one pinning that
  off-by-1% portfolios are still rejected. Suite went from 24/30 to 30/30
  rendered via `scripts/render-all.sh`.

- [ ] **SGB premature-redemption pricing.** Currently every held SGB
  tranche is marked to IBJA gold spot. When the holder actually
  pre-redeems, RBI announces a price ~3 business days before each
  coupon-payment date past the 5-year window — that price is the real
  exit value. Extend `data/sgb_tranches.csv` with a sibling
  `data/sgb_redemptions.csv` (tranche_id, redemption_date,
  inr_per_gram, redemption_kind ∈ {PRE, MAT}); update `sgb_holding_civ`
  to use the actual redemption price on those dates instead of the
  IBJA-spot proxy. Reference: 4 confirmed redemptions are already noted
  in `~/Downloads/sgb-master-ledger.md` (2017-18 Series IV ₹12,704,
  Series XI ₹12,801, Series XIV ₹13,486, 2019-20 Series VII ₹15,275).

- [x] **Deleted `combined_daily_returns()`** (2026-06-17, cycle 4). And
  `civ_and_returns()` which was its only caller. Test churn:
  `test_timeseries_classes.py` swapped to test the canonical
  CIV-pct_change path; `test_plot_metric_consistency.py` regression-doc
  test reframed as pure math (weighted-sum-of-returns ≠
  return-of-weighted-sum) without needing the deleted function.

#### D. Structural improvements

- [x] **Consolidated `*_loader.py` into `loaders/` package** (2026-06-17,
  cycle 14). 7 files renamed: `benchmark_loader.py` →
  `loaders/benchmark.py` (and the same for `gold`, `mutual_fund`,
  `ppf`, `rec_bond`, `risk_free`, `scss`). All importers updated
  (data_loader, main, 7 loader unit-tests). `data_loader.py`
  re-export shim still lives at the top level. `pyproject.toml`
  switched to `packages = ["loaders"]`. Suite 193 pass + 7 network
  pass after the move.
- [ ] **Consolidate `*timeseries*.py` into `timeseries/` package** — four
  files doing related work. *(Already on Phase D backlog.)*
- [ ] **Walk `tests/TODO.md` checklist** — 25+ CLI-flag/TOML-override/
  failure-mode scenarios never converted to real tests. Most overlap with
  `tests/integration/test_main_e2e.py` which currently has only 3 tests.
- [x] **Audited `bond_calculators.py`** (2026-06-17, cycle 5). 4 unused
  functions deleted (`calculate_bond_cumulative_gain` +
  `calculate_sgb_cumulative_gain` + `calculate_merged_sgb_series` +
  `calculate_realistic_sgb_series` — all SGB-related and superseded by
  `sgb_holdings.sgb_holding_civ`). File shrank 298 → 53 lines. Surviving
  `calculate_variable_bond_cumulative_gain` also cleaned (no
  redundant-import, ternary→`max`, drop commented-out debug).

#### E. Quality gates

- [x] **Re-enabled SIM ruff family** (2026-06-17, cycle 6). 4 findings
  autofixed (`--unsafe-fixes` for `with`-block merging, ternary
  conversion, dict-keys iteration, isinstance-merge); 3 manual fixes
  for nested-if guards. Suite clean.
- [x] **Ratcheted CI coverage gate 20% → 65%** (2026-06-17, cycle 7).
  Current coverage on the no-network slice is 73%; 65% gives a small
  headroom for new scaffolding without immediately busting the gate.
- [ ] **Incremental type hints + per-module mypy strictness.** Currently
  `ignore_missing_imports = true` globally. Tighten module-by-module.

#### F. Optional features from the tmp4 attic

- [ ] **Port tmp4's reporting helpers** to current `TimeseriesReturn`:
  `info_summary`, `describe_as_report`, `to_csv_report`, `to_latex_table`,
  `compare_to`, `as_rolling`. *(Already on Phase E backlog.)*
- [ ] **Port tmp4's alignment helpers** if cross-asset analysis grows
  beyond `combined_civ_series`: `align_with`, `clip_to_overlap`,
  `aligned_to`, `interpolated`. *(Already on Phase E backlog.)*

#### I. User-raised 2026-06-17 (post-cycle-7 review)

- [x] **Preserve generated PNGs/CSVs by default** (2026-06-17, cycle 13).
  `make clean` now only removes `portfolio_metrics.csv`; `outputs/` is
  preserved. Added `make rerender` (force-rebuild every PNG + CSV
  without deleting other files in `outputs/`) and `make distclean`
  (the only target that does `rm -rf outputs/`, with a `y`
  confirmation prompt). `scripts/render-all.sh` already iterated
  unconditionally — no change needed there. New `docs/OUTPUTS.md`
  documents the policy plus the per-portfolio sibling-file table
  (`.png`, `.csv`, `.drawdowns.csv`, `.assets.csv`); README +
  QUICKSTART cross-link to it.

- [x] **Clip portfolio at earliest component "death"** (2026-06-17,
  cycle 9). `PortfolioTimeseries.combined_civ_series` already trimmed
  the portfolio CIV to `[max(asset.start), min(asset.end)]`; the
  missing piece was *surfacing* it. Added
  `PortfolioTimeseries.effective_window()` returning
  `{start, end, start_limited_by, end_limited_by}` so `main.py` can
  print the banner "Effective window: <s> → <e>. End set by '<asset>';
  start set by '<asset>'." Also trims `benchmark_returns_series` at
  `<= end` so the alpha/beta computation reflects the same window.
  4 TDD unit tests in `test_portfolio_effective_window.py`. Suite
  186 → 190; 6 goldens still green. CSV outputs unchanged because
  metrics already derive from the (already-clipped) `combined_civ_series`.

- [x] **Simplified CLI invocation** (2026-06-17, cycle 8). Added
  `[project.scripts] portfolio-analyzer = "main:cli"` to
  `pyproject.toml`; refactored `main.py`'s `if __name__ == "__main__":`
  block into a `cli()` function and renamed `def main(args):` →
  `def main(settings):` to make the parameter name match its actual
  use (the old name only worked because `if __name__ == "__main__":`
  variables happen to land in module globals). New invocation:
  `./venv/bin/portfolio-analyzer port/port-1.toml`. Updated
  `QUICKSTART.md`, `Makefile`, and all `scripts/*.sh` to prefer the
  entry-point form. `python main.py …` still works (script delegates
  to `cli()`). README + ARCHITECTURE refresh tracked separately below.

- [x] **Explained: 7 "deselected" tests are network-marked, not
  broken** (2026-06-17). Investigated for KANBAN clarity. They are
  deselected by the `-m 'not network'` default in
  `pyproject.toml [tool.pytest.ini_options].addopts`. The set is the
  6 golden-master tests (`tests/integration/test_golden_master.py`,
  3 portfolios × 2 methods — they each invoke `main.py` end-to-end,
  which fetches NAVs from mfapi.in) plus
  `tests/integration/test_main_e2e.py::test_full_run_produces_csv`
  (also a real subprocess run hitting the network). They are *not*
  skipped or broken — they pass when run with `pytest -m network` or
  `pytest -m 'not network or network'`. Whether to keep them in the
  default-deselected bucket vs. wire them through the planned
  `--replay-from <pickles>` path is the open question (see Phase C
  pickle-replay item).

- [x] **Audited pickle dependency in tests / golden capture**
  (2026-06-17, cycle 12). All three call sites removed:
  (a) `main.py --save-golden-data` flag + `dump_pickle` writer
  deleted; (b) `tests/test_utils.py` `load_pickle` + `pickle` import
  deleted; (c) `tests/test_alignment.py` rewritten as a synthetic-
  fixture behavioral test (2 tests pinning intersection-index +
  MultiIndex columns + ffill no-NaN contract). All 7 pickles under
  `tests/data/` deleted along with the three
  `tests/golden/port-*/pickles/` directories. Suite 192 → 193 with
  zero pickle imports remaining. CSV goldens
  (`tests/golden/port-*/expected_*.csv`) are now the sole golden
  mechanism — the "why CSV not pickle" section in `docs/TESTING.md`
  is now factual rather than aspirational.

- [x] **Audited README.md and docs/ARCHITECTURE.md for staleness**
  (2026-06-17, cycle 11). README: replaced the "metrics_calculator.py
  + sgb_loader.py" Modular Design bullet with the live module list
  (loaders + math + bookkeeping), regenerated the Project Structure
  tree to match (added sgb_holdings/sgb_tranches/fund_lifecycle/
  drawdowns_csv/synthetic_civ/timeseries/etc., dropped ppf_calculator/
  metrics_calculator/sgb_loader). CLI examples now show
  `portfolio-analyzer port/port-1.toml …`; `--do-not-plot -np`
  corrected to `--disable-plot-display -dpd`; documented
  `--skip-age-check`; replaced the stale `-bn`/`-rf` shortcut claim
  with a pointer to `config/example_config.toml`. ARCHITECTURE:
  swapped `sgb_loader` for `sgb_holdings + sgb_tranches`; swapped
  the deleted `combined_daily_returns` row for `effective_window`;
  added module-map rows for `fund_lifecycle`, `drawdowns_csv`,
  `sgb_holdings`, `sgb_tranches`, `gold_loader`; added a "Portfolio
  effective window" subsection explaining the banner.

#### H. Integration with money-vault (do NOT work without explicit go-ahead)

User-raised 2026-06-17: PortfolioAnalyzer could serve as the
*quantitative* backtester for portfolio allocations that the
*qualitative* money-vault LLM-wiki-cum-RAG system at
`~/Projects/money-vault/` recommends. Current state: the two repos
are unaware of each other.

- [ ] **Money-vault integration (capability question).** Treat
  PortfolioAnalyzer as a downstream tool for money-vault portfolio
  suggestions: vault recommends a sized allocation across asset
  classes (Indian equity/debt/gold/PPF/SGB tranches/REC bonds);
  PortfolioAnalyzer renders the historical CAGR / vol / Sharpe /
  drawdowns / alpha-vs-NIFTY for that allocation. Open questions:
  what's the input/output contract (human-translated TOML, or auto-
  generated)? does the analyzer need any vault-side conventions
  (asset-class labels, default weights)? what's the scope —
  Indian-only or extended? **Do not begin work without user's
  explicit go-ahead.**

- [ ] **Design the money-vault ↔ PortfolioAnalyzer bridge** —
  separate, larger cycle. Inputs needed before sketching: the
  relevant money-vault wiki pages (which sections produce portfolio
  recommendations, in what shape), the user's preferred integration
  posture (manual TOML hand-off vs. auto-generated TOML vs. inline
  API call), and whether to extend the analyzer's loader set for
  any asset classes the vault recommends that PortfolioAnalyzer
  doesn't currently cover. **Do not begin work without user's
  explicit go-ahead.**

#### G. Housekeeping / decisions

- [ ] Rename `port/` → `portfolios/` for clarity. *(Already on Hygiene
  backlog.)*
- [ ] Decide fate of `outputs/port-*/` cached runs. *(Already on Hygiene
  backlog.)*
- [ ] Add `.env.example` if any FRED-style API tokens are needed.
- [ ] Relocate `docs/*.pdf` originals once `.md` sidecars exist.
- [ ] Decide fate of `defunct_feature_var_rate_bonds` and `defunct_main`
  branches.

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
- [x] **Portfolio CIV truncates at the earliest-ending asset's last date** — surfaced 2026-06-16 while verifying the plot fix. `port-everything` ended 2024-02-21 because the old SGB data stopped at the final tranche issue date (scheme discontinued). **Fixed indirectly by Phase 2 of the SGB modeling refactor:** SGB is now modeled per-tranche driven by IBJA gold spot, so the SGB CIV extends as long as the gold data does, and the portfolio CIV is no longer dragged backward by the SGB end-date.
- [x] **SGB tranche reference + lookup API** (2026-06-16). `data/sgb_tranches.csv` covers all 33 tranches purchasable from Feb 2020 onward (the user's window); `sgb_tranches.py` exposes `load_tranches()`, `lookup_tranche(id)`, `tranche_status(id, as_of)`, `list_tranches(fiscal_year=, status=, as_of=)`, and `describe_tranche(id)` for the CLI/REPL. Two of the user's actual holdings are anchor-pinned to their RBI certificates: FY 2019-20-IX (11 Feb 2020 @ ₹4,070, Hutoxi 12g) and FY 2020-21-VII (20 Oct 2020 @ ₹5,051, Tom 6g across 2 certs). 11 unit tests.
- [x] **SGB modeling refactor — Phase 2: integration** (2026-06-16). Portfolio TOML schema migrated from `[sgb]` dict to `[[sgb]]` list per the "different tranches → different investments" rule. `main.py` iterates the list, calls `sgb_holding_civ` per entry with `load_gold_prices_per_gram()`, registers each tranche as its own asset in `PortfolioTimeseries`. Old `sgb_loader.create_sgb_daily_returns` deleted along with `data/sgb_data.csv`, `tests/unit/loaders/test_sgb.py`, and the test fixture. `visualizer.py` updated to render per-tranche rows in the asset table ("SGB 2019-20-IX (1 g)" etc.). 7 new schema-validation tests including legacy-form rejection with a helpful migration message. Goldens re-captured for all 3 portfolios × 2 methods. Plot consequence: `port-everything` now extends through 2025-06 (no longer truncated at 2024-02) because the SGB CIV is a function of gold (which has data to today), not bonded to a stale-issue-date series.
- [x] **Enormous `alpha_capm` on mixed-frequency portfolios — fixed (2026-06-16).** Root cause was *not* alpha_capm itself — the function correctly annualizes daily returns with `^252`. The bug was upstream in `main.py`: `portfolio_daily_ret` was built from `combined_daily_returns()`, which inner-joins per-asset return series down to the *monthly* intersection when any monthly asset (gold) is present. Feeding ~51 monthly returns into a function that ^252-annualizes them inflated the mean by ~21×, yielding Alpha = 139.15% on port-everything. Fixed by deriving the daily returns from `combined_civ_series.series.pct_change()` (which is properly business-day cadence thanks to Phase D's frequency fix). Same fix already applied to the plot; now applied to the alpha/beta computation too. Results across portfolios: port-1 Alpha 4.13%→3.21%; port-mf-ppf-gold 68.09%→2.29%; port-everything 139.15%→3.48%. New TDD test (`test_alpha_capm_is_sensible_for_mixed_frequency_portfolio`) pins the contract.
- [x] **SGB modeling refactor — Phase 1: pure-function valuation engine** (2026-06-16). `sgb_holdings.sgb_holding_civ(tranche_id, units_grams, gold_prices)` → daily CIV series. CIV = `units × gold_per_gram(t) + Σ(coupons paid ≤ t)`. Coupon schedule: 16 semi-annual payments over the 8-year tenor, computed with `relativedelta` for correct month-end rollover. 13 unit tests against synthetic gold; verified end-to-end on user's real holdings (Hutoxi 12g of 2019-20-IX → ₹42,910 → ₹105,678, CAGR 19.19%; Tom 6g of 2020-21-VII → ₹27,258 → ₹52,817, CAGR 16.05%). The earlier ambiguous gold CSV (column header "Spot Price"; actually INR per troy ounce) now has an explicit `load_gold_prices_per_gram()` helper.
  - **Hard constraint** (user, 2026-06-16): **each tranche is a distinct portfolio investment.** Different tranches have different issue dates, prices, coupon schedules, and lock-in/maturity timelines — they're financially distinct securities, not interchangeable units of "SGB". The current single-asset `[sgb]` section in the portfolio TOML must be replaced with a list of per-tranche holdings. Lumping rule for the holdings file: lump iff `(tranche_id, issue_date, holder_pool)` matches — i.e., multiple bank-application certificates for the *same tranche on the same issue date* (e.g. the user's HDF...231 + HDF...232, both 2020-21-VII, both 20-Oct-2020) collapse to one investment line. Different tranches NEVER lump.
  - Worked example for the user's three certificates: **two** investment lines, not three.
    - `2019-20-IX × 12 g` (Hutoxi, SBI 2020-02-11, single cert)
    - `2020-21-VII × 6 g` (Tom, HDFC 2020-10-20, two certs lumped per the rule above)
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
