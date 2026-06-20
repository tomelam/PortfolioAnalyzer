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

- [x] **Replicate Value Research Online mutual-fund metrics to high accuracy**
  (2026-06-18). Built a VRO-parity test utility + live wire test; our trailing
  returns reproduce VRO's published figures to **≤0.25pp** (1Y essentially exact).
  - **Matched methodology (the hypothesis was wrong):** VRO does **NOT** use
    month-end NAVs. It uses **point-to-point daily** NAVs — latest NAV to exactly
    N years prior, annualized. Reconciliation: month-end (`--metrics-method
    monthly`) ran ~0.5–0.8pp *low*; daily point-to-point matched within ≤0.25pp
    (5Y to ≤0.01pp). So the canonical analog is plain `metrics.cagr` over the
    trailing window — captured as `loaders.vro.trailing_cagr_pct`.
  - **Access:** VRO is Cloudflare-walled (plain `requests` → 403; legacy `.asp`
    → 410). Cleared with stealth Chromium (the existing `browser` extra), then
    the React app's same-origin JSON API is fetched from inside the page (the
    niftyindices cookie-carry trick): `api/funds/peer-comparison-returns/
    ?fund_id=<id>&period=<P>` → the fund's annualized return is the `returns`
    entry whose `plan_id == fund_id`.
  - **Deliverables:** `loaders/vro.py` (pure `parse_peer_comparison_returns` +
    stealth `fetch_vro_trailing_returns` + `trailing_cagr_pct` + `load_vro_fund_map`);
    `data/funds/vro_funds.csv` (mfapi↔VRO map, ISIN-verified; **all 5 port-1 funds** —
    ICICI Bluechip 120586↔15841, Franklin US Opp FoF 118551↔16027, HDFC Balanced
    Advantage 118968↔16055, ICICI Corp Bond 120692↔15568, HDFC Hybrid Debt
    119118↔16453); `scripts/fetch_vro_metrics.py` (collector CLI: VRO vs ours vs
    Δ, optional `--json` snapshot); `tests/unit/test_vro.py` (9 offline tests,
    fixture-based); `tests/integration/test_vro_parity.py` (live, `network`+`vro`
    marked, skips without the browser extra; tolerance **0.5pp**).
  - **Live result 2026-06-18** (latest NAV 2026-06-17), all 5 funds within 0.25pp:
    ICICI Bluechip Δ +0.00/+0.24/+0.01 (1Y/3Y/5Y); Franklin +0.02/+0.14/+0.01;
    HDFC Bal Adv +.. /+0.05/+0.02; ICICI Corp Bond +0.00/+0.01; HDFC Hybrid Debt
    +0.04/+0.00. (HDFC Bal Adv had two VRO Direct ids — 16055 vs 16056; parity
    disambiguated to **16055** Growth; 16056 returns no peer data.)
  - **Done (next-steps batch, 2026-06-18):** widened the map to all 5 funds
    (each disambiguated by parity vs our Growth-NAV CAGR) and tightened the
    wire-test tolerance 1.0 → 0.5pp.
  - **Done — risk-ratio parity (Mean / SD / Sharpe / Sortino), 2026-06-19.**
    The earlier "no risk-ratios endpoint" finding was right about the *overview
    API* but the data does exist — it just isn't under `/api/funds/*`. The Risk
    tab's ratios load lazily (bundle fn `risk_ratio_tab_ajax`) from
    **`GET /funds/risk-ratios-tab-data/`**, which returns an **HTML fragment**
    (not JSON) and keys on the fund's **short name** (`#fund_name` hidden input,
    e.g. "ICICI Pru Large Cap Dir"), *not* the plan id — with peers optional.
    That route is fully Cloudflare-challenged (the `/api/` JSON routes aren't),
    and an in-page XHR straight from the fund page hangs behind the challenge; the
    working sequence is **navigate the risk URL once to clear CF for that path,
    then in-page XHR** (carries the `X-Requested-With` the route requires).
    Mapped via a throwaway-but-kept discovery spike (`scripts/vro_discover_endpoints.py`)
    that dumps the page bundle + `/api/` traffic + the risk fragment.
    - **Matched methodology:** VRO computes risk ratios trailing-3Y, *monthly*;
      `loaders.vro.trailing_risk_ratios` mirrors it via `metrics.*` with
      `periods_per_year=12`. VRO's assumed risk-free back-solves to ≈5.9% from a
      captured fragment (`VRO_RISK_FREE_ANNUAL`).
    - **Deliverables:** `parse_risk_ratios` (BeautifulSoup, fixture-tested),
      `vro_risk_ratios_api`, `fetch_vro_metrics` (one stealth session, both
      families), `VROMetrics`; fixture `tests/fixtures/vro_risk_ratios.html`;
      extended `tests/unit/test_vro.py` and `tests/integration/test_vro_parity.py`
      (asserts Mean/SD/Sharpe/Sortino) and `scripts/fetch_vro_metrics.py` snapshot.
    - **Beta / Alpha parity — ICICI Bluechip done (2026-06-19).** Reframed from
      "VRO parity" to "our CAPM correctness": source each fund's **stated
      benchmark TRI** and assert ours. Probe (`scripts/probe_niftyindices_*.py`)
      established niftyindices' free feed serves an index iff it's in the
      live-watch master (131 indices): **NIFTY 100 is fetchable** (TRI endpoint),
      but the tiered/hybrid debt benchmarks (Hybrid Composite Debt 65:35 & 15:85,
      Corporate Bond Index A-II) are **not on the feed at all** (endpoint-coverage
      gap, confirmed via a GS-index control), and Russell 3000 Growth (Franklin US
      FoF) is **out of scope by decision** (feeder fund, USD benchmark vs INR NAV,
      no stable free Russell TR source, VRO publishes neither). So niftyindices
      yields exactly 1 of 5 — and it's the only fund VRO publishes both Beta *and*
      Alpha for, and the cleanest equity-vs-equity CAPM case.
      - **Done:** `data/funds/vro_funds.csv` gains a `benchmark_index` column (the
        niftyindices name where fetchable; empty otherwise) → `VROFund.benchmark_index`;
        `loaders.vro.fetch_benchmark_tri(index_name)` wraps the stealth
        niftyindices TRI fetch as a Series; the parity test fetches the benchmark
        TRI (one hit, only for funds with `benchmark_index`) and asserts Beta/Alpha
        where VRO publishes them. **Live 2026-06-19** (NAV 2026-06-18): Beta ours
        0.909 vs VRO 0.92 (Δ−0.011); Alpha ours 3.320 vs VRO 3.33 (Δ−0.010pp).
        Tolerances `TOL_BETA=0.05`, `TOL_ALPHA_PP=0.40`. New unit tests: known-answer
        β=1/α=0 (benchmark≡fund) and the `benchmark_index` column load.
      - **Resolved (Thread 5, 2026-06-19):** the 3 hybrid/debt funds' stated
        benchmarks are **not freely sourceable** (niftyindices feed, product pages,
        factsheets-as-series, Morningstar, MoneyControl all checked — see the Thread 5
        record below) → **closed as out of scope**, same posture as Franklin
        (out of scope). VRO publishes Beta-only for the two HDFC hybrids and
        nothing for ICICI Corp Bond, so the remaining upside was thin anyway.

- [x] **Stale NIFTY benchmark is a hard blocker — add a bypass flag**
  (2026-06-17, cycle 10). Added `--skip-age-check` CLI flag. When
  active it bypasses both the benchmark and risk-free CSV staleness
  gates (the latter via auto-bumping `max_riskfree_delay` to 99999
  unless the user explicitly set it). main.py prints a one-line
  warning ("⚠️ --skip-age-check active …") so the bypass is never
  silent. 2 integration tests: --help lists the flag; strict-by-default
  still blocks with "outdated" on the stale CSV. Default remains
  strict so silent drift can't creep in.

- [x] **Auto-update benchmark + risk-free data** (2026-06-17). Built
  `loaders/data_update.py`: a DataSource registry + fetchers that normalize
  upstream feeds to the loader schema, write last-fetched stamps, and isolate
  per-source failures.
  - Risk-free: FRED `INDIRLTLT01STM` (no-auth CSV) — live-verified; now the
    default risk-free source (`data/reference/INDIRLTLT01STM.csv`).
  - Benchmark: NIFTY 50 TRI from niftyindices.com via a **stealth Chromium
    browser** (the only fetch path — no raw `requests`, which risks flagging
    the IP). Defeats the anti-scrape wall; live-verified (single stealth hit
    returns real TRI data). Steady-state-unattended reliability still to be
    confirmed via cron — see the open item below.
  - On-run: stale data is **force-refreshed by default** (off under
    `--as-of`/`--replay-from`/`--skip-age-check`/`--no-auto-update`); on
    upstream failure it warns and proceeds (early-warning, not a hard block).
  - Cron-able `portfolio-analyzer-update` console script. See
    `docs/DATA_REFRESH.md`. Goldens re-captured against the FRED risk-free.

- [x] **Data-freshness redesign — block-by-default invariant** (2026-06-18).
  Reworked freshness from a warn/skip/tune-the-age model into a correctness
  invariant (spec: `docs/ARCHITECTURE.md` → "Data freshness as a correctness
  invariant"). Reference data (benchmark + risk-free) is auto-refreshed when it
  has fallen off the feed's own publication **cadence** (business day for
  NIFTY, month for FRED) — no magic-number age thresholds. A source that can't
  be certified current **blocks** the run; the single override is the new
  `--allow-stale` (warns, naming the degraded metrics). niftyindices is
  contacted **≤ once/day** (every attempt stamped in `data/.last_fetched.json`;
  a second same-day attempt is suppressed); FRED refreshes whenever behind. The
  report prints per-source provenance (`last_date` + `fetched_at`). `--as-of` /
  `--replay-from` opt out cleanly (neither fetch nor block). **No cron** — the
  analyzer is self-sufficient; `portfolio-analyzer-update` remains an optional
  manual one-shot. Hard-removed `--skip-age-check`, `--no-auto-update`,
  `--max-riskfree-delay` (tombstoned → one-line error pointing to
  `--allow-stale`). `refresh_path_if_stale` replaced by
  `ensure_reference_data_fresh` / `ensure_source_current` / `cadence_frontier`.
  TDD: rewrote `test_data_update.py` freshness section + new
  `test_freshness_gate.py`; goldens stay green at 1e-9. Docs updated
  (ARCHITECTURE, DATA_REFRESH, README, QUICKSTART).

- [ ] **Confirm niftyindices TRI scrape works in steady state (unattended).**
  The fetch is now **stealth-browser-only** (Playwright + `playwright-stealth`),
  ported from the proven `~/Projects/mysore-spa-intelligence-engine` scraper:
  authentic fingerprint (real UA, `en-IN` / `Asia/Kolkata`, 1920×1080, hides
  `navigator.webdriver`), navigate the historical-data page to mint cookies /
  clear any JS challenge, brief human pause, then POST the TRI request from
  inside the page. The raw-`requests` path is **gone** — a `requests` hit
  risks flagging the IP, and the user wants zero trouble with niftyindices.
  Live-verified 2026-06-17: a single stealth hit returned real April-2026 TRI
  data (20 rows, last 36174.80). Needs the optional `browser` extra
  (`pip install '.[browser]' && playwright install chromium`); without it the
  fetch raises a clear install-guiding error.
  - **Still open — verify in steady state:** let a few normal runs (or the
    optional `portfolio-analyzer-update` one-shot) execute over several days
    and confirm `data/reference/NIFTY Total Returns Historical Data.csv` +
    `.last_fetched.json` actually advance via the on-run refresh. One verified
    hit is not proof of unattended reliability; the user (2026-06-17) is right
    that the earlier "it's just burst throttling" diagnosis could have been
    wrong, so treat this as open until the once/day refresh is demonstrably
    reliable.
  - **Further hardening avenues if the scrape ever stalls** (not needed unless
    it does): `launch_persistent_context(user_data_dir=…)` for a returning-user
    cookie jar across runs; incremental tail-fetch (only missing recent days)
    to shrink the footprint; a persisted "next-allowed" timestamp so the tool
    self-throttles across runs; an alternative NIFTY 50 TRI mirror as backup.

- [x] **Added `--as-of YYYY-MM-DD` flag** (2026-06-17, cycle 16).
  Pins the reference (as-of) date across the pipeline: every loaded
  NAV/CIV series (per-fund DataFrames, PPF/SCSS/REC/gold/SGB) is trimmed
  at `<= as_of` in `main.py`; the benchmark staleness gate
  (`loaders.benchmark.load_timeseries_csv`) accepts `as_of=`; the
  lookback computation (`utils.to_cutoff_date`) accepts `as_of=`;
  `fund_lifecycle.build_assets_meta` receives `settings['as_of']`
  for the DEFUNCT-status verdict. 5 new TDD tests
  (`tests/unit/test_as_of_flag.py`). Suite 193 → 198 pass. Tightening
  golden tolerances to 1e-9 is the follow-up step (deferred — current
  goldens still pass at the existing 5% relative tolerance).

- [x] **Built the replay path** (2026-06-17). CSV/HTML, not pickle (pickle
  was excised in be81873). `main.py --replay-from DIR` reads NAV history
  (`DIR/navs/<fund>.csv`) and SCSS (`DIR/scss_nsi.html`) from committed
  fixtures; `--save-replay DIR` captures them. Golden tests now run offline
  in the default suite (network marker dropped); the only two network
  sources were mfapi NAVs + SCSS HTML. See Phase C entry for detail.

#### C. Correctness bugs surfaced but not fixed

- [ ] **Maturing instruments are modelled as perpetual compounding** (surfaced
  2026-06-19 by the REC-bond drop). SCSS (and the dropped REC bond) are valued by
  `calculate_variable_bond_cumulative_gain`, which compounds a fixed coupon from
  2000→today with **no maturity event** — so a 5-year instrument keeps "earning"
  forever in a backtest, overstating its late-window contribution. A faithful
  model would stop compounding at the maturity (or pre-redemption) date and carry
  the principal as **cash** (flat) thereafter, optionally reinvested. This is the
  same gap as SGB maturity (see SGB items) — worth doing **generally** for all
  fixed-term instruments (SCSS, SGB, FDs), not per-asset. Not yet scheduled;
  related: [[project-sgb-htm-default]]. *(This is the real value the REC bond
  surfaced; REC itself was dropped — modelled identically to SCSS, no longer held.)*

- [x] **Fixed `PortfolioTimeseries.__init__` weight-sum check** (2026-06-17,
  cycle 1). Strict `!= 1` replaced with `abs(total_weight - 1) > 0.01`.
  All 6 previously-failing portfolios now render. Two new unit tests:
  one pinning that FP-rounding portfolios are accepted, one pinning that
  off-by-1% portfolios are still rejected. Suite went from 24/30 to 30/30
  rendered via `scripts/render-all.sh`.

- [ ] **SGB premature-redemption pricing — BACK BURNER (user, 2026-06-19).**
  **Decision: value all SGBs hold-to-maturity (HTM) for now** — i.e. keep the
  current IBJA-gold-spot-plus-coupons valuation everywhere SGBs appear (plots,
  charts, metric tables). Premature-redemption pricing is **not a big win right
  now**, so it is deferred. (Recorded in `docs/ARCHITECTURE.md` → *SGB valuation:
  hold-to-maturity by default*.) For our two held tranches the gold-spot proxy
  *is* the HTM value within the current window (neither has reached the 8-year
  maturity pin), so no code change is needed to honour this decision today.
  - *When revived:* every held SGB tranche is marked to IBJA gold spot; a real
    pre-redemption uses the RBI price announced ~3 business days before each
    coupon date past the 5-year window. Extend `data/funds/sgb_tranches.csv` with a
    sibling `data/sgb_redemptions.csv` (tranche_id, redemption_date, inr_per_gram,
    redemption_kind ∈ {PRE, MAT}); update `sgb_holding_civ` to use the actual
    redemption price on those dates instead of the IBJA-spot proxy. Reference:
    4 confirmed redemptions noted in `~/Downloads/sgb-master-ledger.md` (2017-18
    Series IV ₹12,704, Series XI ₹12,801, Series XIV ₹13,486, 2019-20 Series VII
    ₹15,275).

- [ ] **SGB hold-to-maturity vs redeemable subtypes — BACK BURNER** (user-raised
  2026-06-18; deferred 2026-06-19 with the redemption-pricing item it depends on).
  HTM is the default valuation in the meantime (above). Split the SGB asset type
  into two valuation subtypes so a portfolio can model a holder's actual intent
  per tranche:
  - **hold-to-maturity (HTM)** — valued through to the 8-year maturity, with the
    terminal value pinned to RBI's maturity redemption price (the average of the
    last week's gold) rather than the single-day IBJA-spot proxy.
  - **redeemable** — a tranche the holder intends to (or may) pre-redeem from the
    5-year window onward; valued at the RBI-announced premature-redemption price
    on the redemption date (the `sgb_redemptions.csv` mechanism in the item above).
  The scheme no longer accepts new purchases, so the universe is fixed. For our
  personal example we hold **2 tranches** (2019-20-IX, 2020-21-VII); crossed with
  the 2 subtypes that yields **4 possible SGB-related asset types** for hypothetical
  portfolios — letting us model, e.g., holding IX to maturity while pre-redeeming
  VII early and compare the portfolio metrics under each combination. Builds on the
  per-tranche "each tranche is a distinct investment" constraint (see Hygiene →
  SGB Phase 1) and depends on the premature-redemption pricing item above.

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
- [x] **Consolidated `*timeseries*.py` into `timeseries/` package**
  (2026-06-17, cycle 15). 4 files renamed: `timeseries.py` →
  `timeseries/returns.py`, `timeseries_civ.py` → `timeseries/civ.py`,
  `asset_timeseries.py` → `timeseries/asset.py`, `portfolio_timeseries.py`
  → `timeseries/portfolio.py`. `timeseries/__init__.py` re-exports the
  four public classes (TimeseriesReturn, TimeseriesCIV, AssetTimeseries,
  PortfolioTimeseries) plus the two factories (`from_civ`,
  `from_multiple_nav_series`). All consumers' imports updated to
  point at the qualified module path. Suite 193 pass + 7 network
  pass after the move.
- [x] **Walked `tests/TODO.md` checklist** (2026-06-18). The "25+ scenarios"
  framing was stale: the settings/config/CLI/freshness/failure sections were
  already `[x]` (covered by `test_settings_merge`, `test_data_update`,
  `test_freshness_gate`, and an e2e suite that had grown to 7 functions, not 3).
  Only the 3 items under "Remaining (not yet covered)" were genuinely open — all
  now in `test_main_e2e.py`:
  (1) garbage benchmark file → clean non-zero `Error:` naming the problem;
  (2) wrong benchmark date format → **clean hard error** (decided: fail-fast, not
  a warning — proceeding on mis-parsed dates would silently corrupt benchmark
  metrics; TODO wording updated with rationale);
  (3) drawdown-threshold `/100` end-to-end — which **surfaced a latent bug**:
  `main.py` built `settings["drawdown_threshold"]` but then hardcoded
  `max_drawdowns(threshold=0.05)`, so `--max-drawdown-threshold` and the config
  override were silently dead. Fixed to `settings["drawdown_threshold"] / 100`
  (default 5.0→0.05 unchanged ⇒ goldens green). Suite 340→343.
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
- [x] **Incremental type hints — light-touch pass** (2026-06-18). Investigated
  first: at the project's (sane) mypy config the *entire* real codebase had only
  **5 errors**, so there was no "vast slew" to annotate. Fixed the 3 real things
  they pointed at rather than carpet-bombing annotations:
  - **Latent bug found by mypy:** `loaders/benchmark.py`, `loaders/risk_free.py`,
    and `data_loader.py` did `from main import DEBUG` (try) / `DEBUG = False`
    (except). `main.py` never defines a module-level `DEBUG` (it sets
    `utils.DEBUG`), so the import *always* failed → `DEBUG` was permanently False
    → the `if DEBUG:` debug-logging blocks were **dead, even under `--debug`**.
    Replaced the broken dance with the already-imported `dbg()` (which gates live
    on `utils.DEBUG`), so debug logging now actually works; `data_loader`'s
    `DEBUG` was imported-but-unused and just deleted.
  - `metrics.py` drawdown loop: `assert peak_value is not None …` to document the
    in-drawdown invariant and satisfy the None-operand check (1 line).
  - `portfolio_calculator.py`: `allocations: dict[str, float]` annotation.
  mypy now clean at the project config; ruff clean; suite 343 green.
  **Deliberately NOT done** (over-annotation / noise): adopting `pandas-stubs` +
  flipping `ignore_missing_imports` off. That is the *only* way to make mypy
  verify Series-vs-DataFrame return contracts (the class behind the old gold
  loader bug), but it adds a dependency and a lot of `Any`-wrestling noise for
  modest gain. Left as an explicit future option, not a default. See the Hygiene
  entry below.

#### F. Optional features from the tmp4 attic

- [x] **Port the parked reporting helpers** to current `TimeseriesReturn`:
  `info_summary`, `describe_as_report`, `to_csv_report`, `to_latex_table`,
  `compare_to`, `as_rolling`. **DONE Thread 6 (2026-06-19)** — revived onto the
  live class, re-wired to the Series API; see the Thread 6 record in the in-flight
  section above.
- [ ] **Port the parked alignment helpers** if cross-asset analysis grows
  beyond `combined_civ_series`: `align_with`, `clip_to_overlap`,
  `aligned_to`, `interpolated` (also parked in
  `attic/timeseries_return_helpers.py`). *(Also on Phase E backlog.)*

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
  pickle-replay item). **Update (2026-06-17):** resolved — the 6 golden
  tests now use `--replay-from` and run offline in the default suite;
  only `test_main_e2e::test_full_run_produces_csv` remains `network`.

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

- [x] **Decided: keep `port/`** (2026-06-17). Renaming to `portfolios/` is
  purely cosmetic and would churn the Makefile, tests, README/QUICKSTART, and
  4 docs right after the v0.1 tag for marginal clarity. `port/` is
  unambiguous in context. (Will rename on request.)
- [x] **Decided: keep `outputs/` as a local cache** (2026-06-17). It is
  gitignored (0 tracked files) and the Makefile preserves it by default
  (commit 17eb010). Nothing to move to attic — these are reproducible
  renders, not source.
- [x] **`.env.example` not needed** (reconciled 2026-06-17). Repo-wide grep finds no `os.environ` / `getenv` / API-token / `.env` usage; all data sources are local CSVs or unauthenticated HTTP (mfapi.in, nsiindia.gov.in). Revisit if a token-gated source (e.g. FRED) is ever added.
- [x] **Relocated `docs/*.pdf` originals to `docs/sources/`** (2026-06-17). All 4 PDFs (5-ratios, RBI WSS guide, CRISIL ranking, OpenAI_Models) moved via `git mv`; their `.md` sidecars stay in `docs/` and now point at `sources/`.
- [x] **Deleted `defunct_feature_var_rate_bonds` and `defunct_main`**
  (local + origin, 2026-06-17). See the Hygiene-section entry for the
  recorded tip SHAs.

### Phase C — Golden-master safety net (all 3 portfolios DONE)
- [x] **Tightened golden-master tolerances 5% → 1e-9** (2026-06-17). Pin
  the test with `--as-of 2026-06-13` so series are trimmed to `<= AS_OF`
  and runs reproduce bit-for-bit; re-captured all 6 goldens at that date.
  Every numeric column now compared at 1e-9 (was 5% relative); drawdowns
  count also asserted. (Network dependency later removed by the replay
  path below.)
- [x] **Built CSV/HTML replay path** (2026-06-17). `main.py --replay-from DIR`
  reads NAV history (`DIR/navs/<fund>.csv`) + SCSS (`DIR/scss_nsi.html`)
  from committed fixtures under `tests/golden/replay/`; `--save-replay DIR`
  captures them from a live run. Golden tests dropped `@pytest.mark.network`
  and now run offline in the default suite (204 passed). CSV/HTML, not
  pickle (be81873). Threaded `replay_from`/`save_replay` through
  `fetch_portfolio_civs` + `load_scss_interest_rates`; also dropped a
  redundant second `fetch_portfolio_civs` call in main.py.
- [x] **Recorded coverage baseline** (2026-06-17). TOTAL 74% (3121 stmts,
  805 missed) at 204 passed; CI gate is `--cov-fail-under=65` (passing).
  Documented per-module in `docs/TESTING.md` → Coverage, including the
  `main.py` 1% subprocess-measurement caveat. Biggest genuine gap:
  `timeseries/returns.py` at 44%.
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
- [x] **Created the `loaders/` package** (commit 83849c0) — moved all `*_loader.py` files in; `data_loader.py` re-exports for compat. (Used top-level `loaders/`, not `portfolio_analyzer/loaders/`.)
- [x] Extract `fetch_navs_of_mutual_fund` → `mutual_fund_loader.fetch_navs` with 7 unit tests (DataFrame contract, nav float, sorted DatetimeIndex, dayfirst parsing, retry-on-transient, exhausted-retries error, missing-data error). `data_loader` re-exports for compat.
- [x] Extract `load_ppf_interest_rates` + `load_ppf_civ` → `ppf_loader.py` with 3 added rate-CSV contract tests. Total PPF tests: 8.
- [x] Extract `load_timeseries_csv` → `benchmark_loader.py` with 6 tests. Deleted dead `load_index_data` and its unused main.py import.
- [x] Extract `fetch_and_standardize_risk_free_rates` + `align_dynamic_risk_free_rates` → `risk_free_loader.py` with 5 tests (Series return, percent→decimal, staleness, weekend-gap interpolation, alignment).
- [x] Extract NSI SCSS scraper → `scss_loader.py`, decomposed into pure `parse_scss_html` + network `fetch_scss_html` + composed `load_scss_interest_rates`. 8 tests using static HTML fixture, all in 0.5s.
- [x] Extract REC bond from `main.py` → `rec_bond_loader.py` honoring the TOML `coupon` field (was silently hardcoded 5.25%). 5 tests.
- [x] Unit tests for `sgb_loader.py` (added in commit 791d955; KANBAN was stale).
- [x] **Consolidated all `*_loader.py` files into the `loaders/` package** (commit 83849c0; same work as the item above). No top-level `*_loader.py` remain.
- [x] **Consolidated the four `*timeseries*.py` files into the `timeseries/` package** (commit 0f440fc): `timeseries/{civ,asset,portfolio,returns}.py`. No top-level `*timeseries*.py` remain.
- [x] Test `TimeseriesCIV`, `TimeseriesReturn`, `AssetTimeseries`, `PortfolioTimeseries` independently — 11 tests in `tests/unit/test_timeseries_classes.py` covering class-level surface (constructors, validation, `combined_daily_returns`, hand-rolled `TimeseriesCIV.max_drawdowns`, `summary`). Coverage of the four files now 71–98% (was 23–81%).
- [x] Unit tests for `synthetic_civ.py` — 6 tests pin PPF monthly accrual, March yearly credit, mid-year rate changes, and Series/DataFrame input flexibility.
- [x] Unit tests for `civ_to_returns.py` — 6 tests including the round-trip identity (CIV → returns → cumprod ≈ normalized CIV).
- [x] **Replace dead/scattered metrics code with tested `metrics.py`** (2026-06-18).
  The risky 80% (cagr/vol/Sharpe/Sortino/max_drawdown/max_drawdowns) was already
  consolidated in Phase C/F; the only metric math still living inline in
  `timeseries/returns.py` was the CAPM/regression alpha & beta. Moved all four
  (`alpha_capm`, `beta_capm`, `alpha_regression`, `beta_regression`) into
  `metrics.py` as pure functions taking two return Series; the class methods now
  delegate (and the lazy per-method `import metrics` + unused `warnings`/`dbg`
  imports were cleaned up — single top-level `import metrics`). 10 new pure-function
  tests in `test_metrics.py` (beta=2× / 3× known answers, zero-intercept, identical-
  series-alpha=0, empty/too-few raises, fallback-to-regression-beta via monkeypatch).
  The class-level fallback test was repointed at `metrics.beta_capm` since the
  instance method is no longer consulted. Behavior byte-identical: all 6 goldens
  green at 1e-9; suite 337 pass (340 incl. network). `returns.py` delegators 100%
  covered, `metrics.py` 95%, TOTAL 88%.
- [x] **Reference `attic/tmp-mar2025/test_metrics.py`** (2026-06-18). Inspected —
  it only asserts dict keys exist (`"Annualized Return" in metrics`); no golden
  formulas to salvage. The live `tests/unit/test_metrics.py` is already richer.
  Nothing to port.
- [x] Decided: `portfolio_calculator.py` — KEEP. Two functions live (`calculate_portfolio_allocations`, `calculate_gains_cumulative`); dropped dead `calculate_gain_daily_portfolio_series` and its unused import in main.py, plus the commented-out duplicate header.
- [x] **Audited `bond_calculators.py`** (already done in post-v0.1 cleanup cycle 5; reconciled 2026-06-17). The 4 unused funcs (`calculate_bond_cumulative_gain` + 3 pre-refactor SGB approximations) were deleted — superseded by `sgb_holdings.sgb_holding_civ`. Only the live `calculate_variable_bond_cumulative_gain` remains (used by `loaders/rec_bond.py` + `main.py` SCSS path); file is now ~52 lines at 100% coverage. Their removal is preserved in git history (no future rationale — superseded, not parked).
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
- [x] Walked `tests/TODO.md` checklist (2026-06-18). See the detailed entry under
  *Structural improvements* above — bulk was already covered; the 3 genuinely-open
  "Remaining" items are now in `test_main_e2e.py`, and the exercise surfaced + fixed
  the dead `--max-drawdown-threshold` flag (hardcoded 0.05 in `main.py`).

### Phase E — Salvage from old checkpoints
- [x] **Live-gold via yfinance: DROPPED.** User: "yfinance probably cannot be depended upon by a stable program." yfinance scrapes an undocumented Yahoo endpoint that breaks without warning. Keep the static monthly `data/reference/gold_monthly_inr.csv` path; document the manual refresh procedure in `docs/DATA_REFRESH.md` (see Data freshness section). If a stable public gold-price API surfaces later, port it then — but not yfinance.
- [x] **Audit of `attic/tmp4-apr2025` complete.** `TimeseriesFrame` in tmp4 ≈ current `TimeseriesReturn` (the rename happened during the OOP rewrite). Substantively additional surface in tmp4 is *reporting utilities*, not math: `info_summary`, `describe_as_report`, `to_csv_report`, `to_latex_table`, `compare_to`, `as_rolling`, `align_with`, `clip_to_overlap`, `aligned_to`, `interpolated`, `plot_with`. The math (cagr/vol/sharpe/sortino/max_drawdown/alpha/beta) is equivalent to current `metrics.py`. One semantic difference in tmp4: it divides annual `risk_free_rate` by `periods_per_year` internally; current pipeline does the conversion correctly upstream in `main.py` (geometric per-period rate), so no port needed. **Nothing blocks v0.1-salvage.** Reporting utilities backlogged below.
- [x] **Audit of `attic/tmp4-apr2025/bonus/` complete.** 4 shell helpers (`plot_all.sh`, `run_all_configs.sh`, `run_all_metrics_to_csv.sh`, `single-asset-type.sh`) and one diagnostic script (`ppf_annualized_interest_rate.py`). Useful as references but non-blocking; backlogged for post-v0.1.

### Phase E — backlog (post-v0.1 only)
- [x] Port the reporting helpers to current `TimeseriesReturn`: `info_summary`, `describe_as_report`, `to_csv_report`, `to_latex_table`, `compare_to`, `as_rolling`. **DONE Thread 6 (2026-06-19).**
- [ ] Port the series-alignment helpers if cross-asset analysis grows beyond `combined_civ_series`: `align_with`, `clip_to_overlap`, `aligned_to`, `interpolated`. Now parked in `attic/timeseries_return_helpers.py` (2026-06-17).
- [ ] Port tmp4 `bonus/` shell helpers (plot_all / run_all_configs / run_all_metrics_to_csv / single-asset-type) into a `scripts/` directory if the user wants CLI orchestration.
- [ ] Port tmp4 `bonus/ppf_annualized_interest_rate.py` as an analysis tool under `scripts/`.

### Phase F — Integration + docs
- [x] `tests/integration/test_main_e2e.py` — 3 subprocess tests: --help exits clean; missing TOML → non-zero exit; full-run smoke produces CSV. Integration-marked; one is `network`-marked.
- [x] `docs/ARCHITECTURE.md` — module map + pipeline diagram + design decisions (pure-function math layer; unit-free daily-calendar CIV; synthetic CIVs; risk-free rate convention).
- [x] `docs/TESTING.md` — three test tiers, marker matrix, golden tolerances, regeneration recipe, why-CSV-not-pickle.
- [x] `docs/CONTRIBUTING.md` — TDD-first rule, decomposition guidance, pre-commit, KANBAN expectation, anti-patterns (yfinance, hypothetical abstractions, what-comments).
- [x] **Tagged + pushed `v0.1-salvage`** (annotated tag at `1b53d5c`, "salvage: merge Phase B–F", 2026-06-16; confirmed on origin 2026-06-17). The 38 commits since are post-v0.1 work and intentionally not under this tag.
- [x] **Decided: keep the repo public** (user, 2026-06-17). No reason to make it private during salvage.

### Data freshness (separate from salvage; user responsibility)
- [x] **Benchmark + risk-free refresh is now automated** (2026-06-17) — superseded the manual-download burden. See the "Auto-update benchmark + risk-free data" item under *Determinism / trustworthiness* and `docs/DATA_REFRESH.md`:
  - Risk-free now FRED `INDIRLTLT01STM` (auto-fetched); benchmark NIFTY 50 TRI auto-scraped from niftyindices.
  - Procedure documented in `docs/DATA_REFRESH.md`; on-run force-refresh + cron-able `portfolio-analyzer-update`; staleness gate is now early-warning, not a hard blocker.
  - Goldens re-captured against the FRED risk-free (pinned via `--as-of`/`--replay-from`, so they stay deterministic regardless of live data).
- [ ] **Audit the remaining under-watched data sources** not yet auto-updated: `ppf_interest_rates.csv`, gold (`data/reference/gold_monthly_inr.csv`). SCSS is already fetched live. These change rarely; add registry entries if/when stable feeds are identified.

### Hygiene / tech debt
- [x] **Plot ↔ metrics consistency** (Phase F follow-up). `main.py` fed the plot with `cumprod(1 + combined_daily_returns)` while the metrics box used `combined_civ_series`. For mixed-frequency portfolios (daily MFs + monthly gold), the two diverged because weighted-sum-of-asset-returns ≠ return-of-weighted-sum, and the legacy `combined_daily_returns` inner-joined to the monthly intersection. Fixed `main.py` to feed `portfolio_civ_series.series` directly to the plotter. Three TDD tests pin the contract (`tests/unit/test_plot_metric_consistency.py`); the third test documents the *reason* the old path was wrong and will fail loudly if `combined_daily_returns` is ever independently re-aligned.
- [x] **Portfolio CIV truncates at the earliest-ending asset's last date** — surfaced 2026-06-16 while verifying the plot fix. `port-everything` ended 2024-02-21 because the old SGB data stopped at the final tranche issue date (scheme discontinued). **Fixed indirectly by Phase 2 of the SGB modeling refactor:** SGB is now modeled per-tranche driven by IBJA gold spot, so the SGB CIV extends as long as the gold data does, and the portfolio CIV is no longer dragged backward by the SGB end-date.
- [x] **SGB tranche reference + lookup API** (2026-06-16). `data/funds/sgb_tranches.csv` covers all 33 tranches purchasable from Feb 2020 onward (the user's window); `sgb_tranches.py` exposes `load_tranches()`, `lookup_tranche(id)`, `tranche_status(id, as_of)`, `list_tranches(fiscal_year=, status=, as_of=)`, and `describe_tranche(id)` for the CLI/REPL. Two of the user's actual holdings are anchor-pinned to their RBI certificates: FY 2019-20-IX (11 Feb 2020 @ ₹4,070, Hutoxi 12g) and FY 2020-21-VII (20 Oct 2020 @ ₹5,051, Tom 6g across 2 certs). 11 unit tests.
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
- [x] **`SIM` (flake8-simplify) lint family re-enabled** (reconciled 2026-06-17). `SIM` is in `[tool.ruff.lint] select` and `ruff check .` is clean — no outstanding hints.
- [x] **No deprecated `.fillna(method='ffill')` remain** (reconciled 2026-06-17). Repo-wide grep is clean; the `synthetic_civ.py` cases were fixed earlier (see Phase-C bug-fix log).
- [x] Add type hints incrementally (2026-06-18). Light-touch pass done — see
  *Quality gates → Incremental type hints* above (fixed the dead `main.DEBUG`
  import bug + 2 trivial annotations; codebase was already cleanly typed at the
  signature level). Switching `ignore_missing_imports` off per-module is
  **deferred by decision** — it only pays off with `pandas-stubs`, which adds a
  dependency and `Any`-noise disproportionate to the benefit; revisit only if a
  Series/DataFrame contract bug recurs.
- [x] **CI coverage gate raised to 85%** (2026-06-17; `--cov-fail-under=85`). Baseline TOTAL is 87% (287 passed) after adding `data_loader`, `data_update`, and `visualizer` (smoke) tests, and parking the dead `ppf_calculator.py` / `extract_gold_inr_from_excel.py` to `attic/`. The only large remaining "gap" is `main.py` at 1%, which is a subprocess-measurement artifact (it's covered end-to-end by the golden + e2e tests), not untested behavior.
- [x] **Deleted `defunct_feature_var_rate_bonds` and `defunct_main`** (local + origin, 2026-06-17), now that `v0.1-salvage` is confirmed tagged + pushed. Both superseded (defunct_main = pre-salvage main, fully reachable from current main; defunct_feature_var_rate_bonds = old Yahoo-Finance-replacement WIP, functionality reimplemented live in `bond_calculators.calculate_variable_bond_cumulative_gain`). Tip SHAs recorded for recovery (defunct branch had 8 commits not on main): `defunct_main` = `cea0300`, `defunct_feature_var_rate_bonds` = `2fe272d`. Recoverable from local reflog short-term; re-create with `git branch <name> <sha>` if ever needed.
- [x] **Decided: keep `port/`** (2026-06-17) — cosmetic rename not worth the cross-repo churn post-tag; see Housekeeping section.
- [x] **Decided: keep `outputs/` as a gitignored local cache** (2026-06-17) — see Housekeeping section.
- [x] **`Makefile~` editor backup** — already absent (reconciled 2026-06-17); not tracked, not on disk.
- [x] **`.env.example` not needed** (2026-06-17) — no env-var/token usage anywhere; revisit if a token-gated data source is added.
- [ ] Consider relocating `docs/*.pdf` originals to a `docs/sources/` subdirectory once `.md` sidecars exist

## In Progress

**Big round of cleanup / audits / improvements (user, 2026-06-19).** Scope = all
four areas (code+coverage, docs, data+benchmarks, reporting); sequence = tidy
first, then big. Per-thread branch + full network-gated merge. Progress below.

### Big round — thread log

- [x] **Thread 1 — dead-code sweep** (merged 2026-06-19, `--no-ff`). Vulture-guided
  removal of ~342 lines reachable from nothing + tests of dead code: superseded
  `vro.fetch_vro_trailing_returns`, never-read `DataSource.description`, 4 legacy
  `data_loader` funcs, 2 whole dead test files, 5 unused conftest fixtures.
- [x] **Thread 2 — coverage gaps** (2026-06-19). The stale premise (returns.py 44%)
  was obsolete — returns.py is now 100% (metrics consolidation). Real gaps closed:
  `portfolio_calculator` 62→100, `sgb_tranches` 86→100 (incl. the thread-1-deferred
  `describe_tranche` test), `timeseries/portfolio` 70→98. **Bug found+fixed:**
  `combined_civ_series` returned an *unnamed* empty Series on non-overlapping
  assets → `TimeseriesCIV` ctor rejected it → non-overlapping portfolios crashed;
  now names it `"value"`. Also removed 2 provably-unreachable `if not self.assets:`
  guards (ctor already forbids empty). TOTAL 89→90%.
- [x] **Thread 3 — data-source freshness & quality** (2026-06-19). Scoped to the
  bounded data-quality work; the open-ended hybrid/debt benchmark-TRI *sourcing*
  was deferred into Thread 5 (big) per tidy-first. Delivered:
  - **Gold staleness now surfaced.** `data/reference/gold_monthly_inr.csv` was ~15 months
    stale (ended 2025-03-31) with nothing catching it. New
    `data_update.manual_staleness_warning` (reuses `cadence_frontier`); main.py's
    `_warn_manual_sources` prints a one-line stderr warning naming the file +
    affected metrics (gold/SGB) on non-deterministic runs. Warn-only (feedless →
    can't block/auto-fetch, per ARCHITECTURE).
  - **PPF data bug fixed:** `data/funds/ppf_interest_rates.csv` had `2025=01-01` (typo)
    → the row was silently dropped → PPF stopped accruing at 2024-10 instead of
    2025-01. Fixed the date; made `load_ppf_interest_rates` **fail-fast** on
    unparseable dates / non-numeric rates (was silent-drop). Re-captured the 4
    PPF-bearing goldens (CAGR +0.03–0.05pp; anchored cols unchanged).
  - **Gold loader** also fails fast on unparseable dates (was silent NaT-coerce).
  - PPF deliberately *not* cadence-warned (sparse-by-design: rows only on rate
    *changes*). Docs: `DATA_REFRESH.md` → *Manual, feedless sources*.
  - **REC bond DROPPED as an asset class** (user, 2026-06-19): the user's REC
    54EC bond matured and was redeemed to cash. REC was modelled *identically* to
    SCSS (same `calculate_variable_bond_cumulative_gain` perpetual-compounding
    machinery), so it demonstrated nothing SCSS doesn't, and the user no longer
    holds it. Removed `loaders/rec_bond.py`, `port/port-rec-bond.toml`,
    `tests/unit/loaders/test_rec_bond.py`, all `main.py`/`data_loader.py`/
    `fund_lifecycle.py` handling, and doc/pyproject refs. `port-everything`'s 5%
    REC folded into SCSS (5→10%, both fixed-coupon bonds); 2 goldens re-captured
    (CAGR 13.21→13.32%, label updated). Surfaced a real limitation → backlog item
    below.
  - **niftyindices steady-state** = observational (needs real runs over days);
    nothing to code — stays open for the user to confirm.
- [x] **Thread 4 — docs staleness sweep** (2026-06-19). Reconciled the canonical
  user-facing docs against current code:
  - **README**: project-structure tree + Modular-Design bullet rewritten — flat
    `timeseries*.py` → the `timeseries/` package (`returns`/`civ`/`asset`/
    `portfolio`); top-level `*_loader.py` → the `loaders/` package.
  - **ARCHITECTURE**: module-map + pipeline diagram names fixed (`*_loader.py` →
    `loaders/*.py`, `gold_loader.py` → `loaders/gold.py`, `asset_timeseries.from_civ`
    → `timeseries.asset.from_civ`).
  - **TESTING**: coverage baseline refreshed (287→355 passed, TOTAL 87→89%,
    returns.py 99→100%, main.py 1→14%).
  - OUTPUTS / CONTRIBUTING / QUICKSTART: clean (no stale module refs).
  - **Follow-up DONE (user, 2026-06-19):** archived the 8 orphaned pre-refactor
    ChatGPT-era planning docs (STRUCTURE, TODO, Variables, Calculations,
    Calculation_styles, Class_Roles_Summary, RefactorTable, full_global_todos) —
    0 references from canonical docs, described a superseded design (TimeseriesFrame
    / metrics_calculator) — to `attic/legacy-planning-docs/` (with a provenance
    README). External reference material (CRISIL, RBI WSS, 5-ratios, etc.) stays in
    `docs/`.
  - **Follow-up DONE (user, 2026-06-19):** removed app-dead `utils.warn_if_stale`
    (+ its only consumer `tests/test_staleness.py` + the now-unused `import datetime`)
    — a thread-1 dead-code miss, superseded by the freshness invariant. README +
    ARCHITECTURE `utils.py` role lines updated.
- [x] **Thread 6 — reporting helpers revived** (2026-06-19). Ported the six
  parked reporting helpers from `attic/timeseries_return_helpers.py` onto the
  live `TimeseriesReturn`, re-wired to today's Series-based API: `info_summary`,
  `describe_as_report`, `to_csv_report`, `to_latex_table`, `compare_to`,
  `as_rolling`.
  - The parked code referenced an older surface that no longer exists
    (`self.columns` / `self.shape` / `self['value']` / `self.annualized()`);
    rewired to `value_series()` / `index` and the current metric methods.
  - New private `_summary_metrics(ts, …)` replaces the removed `annualized()`
    dict: annualized-return → `cagr()`, annualized-vol → `volatility()`. Carries
    an `is_percent` flag so CAGR/Max DD/Vol render as percents and Sharpe/Sortino
    as plain ratios across both `to_latex_table` and `compare_to`.
  - `compare_to` now raises `TypeError`/`ValueError` (was `assert`) and keeps the
    <30-overlapping-dates guard. `info`/`describe`/`compare` write to **stderr**
    via `utils.info` (per self-documenting-outputs preference).
  - 14 new tests in `tests/unit/test_reporting_helpers.py`; parked copies removed
    from the attic (alignment helpers `align_with`/`clip_to_overlap`/`aligned_to`/
    `interpolated` stay parked). Full unit suite green; ruff + mypy clean.
- [x] **Thread 5 (big) — benchmark TRI sourcing** (2026-06-19). Investigated every
  free source for the 3 hybrid/debt funds' stated benchmarks (HDFC Balanced Advantage
  → 65:35, HDFC Hybrid Debt → 15:85, ICICI Corp Bond → A-II); **none yields a usable
  time series → closed as out of scope (not freely sourceable)**, same posture as
  Franklin. Evidence (all reproducible):
  - **niftyindices Backpage endpoints — definitive dead end.** New
    `probe_niftyindices_hybrid_exact.py` POSTs the *exact* registered names (+ variants)
    to **both** `getTotalReturnIndexString` (TRI) and `getHistoricaldatatabletoString`
    (HIST) in one session with a NIFTY 100 control: all 3 → **0 rows on both**, control
    → 136 rows on both. Upgrades the earlier live-watch-master finding to airtight (not
    a naming/session issue — true endpoint-coverage gap).
  - **niftyindices product pages** (`probe_niftyindices_productpage.py`) — no chart/
    time-series XHR, only factsheet metadata. **Factsheets** give a monthly snapshot,
    overwritten each month (no backfillable series).
  - **Morningstar & MoneyControl** (user-requested) — risk ratios computed vs
    Morningstar **category/standard indices** ("Nifty 50 TR INR" for the hybrids), not
    the stated benchmark; β clusters ~1 regardless of asset mix (fund-vs-category); **no
    stated-benchmark series exposed**. MoneyControl's risk JSON is curl-scrapeable but
    benchmark-mismatched, so not a clean cross-check.
  - **No production-code change** — `data/funds/vro_funds.csv` already leaves `benchmark_index`
    empty for these 3, so `loaders/vro.fetch_benchmark_tri` + the parity test correctly
    skip them. Documented in `docs/ARCHITECTURE.md` (External-metric parity) and
    `scripts/README.md` (2 new probes). **Big round complete** — all of Threads 1–6 done.

### Fund-benchmark catalog (post-big-round, 2026-06-20)

Goal: catalog popular/prominent funds of different kinds, tracking each fund's
stated benchmark and whether that benchmark has a **free data source** — to (a) add
per-fund α/β validation cases where sourceable and (b) build realistic test
portfolios. Catalog lives at **`data/funds/fund_catalog.csv`** (superset of
`data/funds/vro_funds.csv`); integrity guarded by `tests/unit/test_fund_catalog.py`.

- [x] **Catalog built — 19 funds across 14 categories** (2026-06-20). Fetchability
  determined authoritatively against the niftyindices live-watch master
  (`LiveIndicesWatch_new.json`, 131 indices); exact TRI-fetch spellings confirmed in
  one stealth session (`scripts/probe_niftyindices_catalog_names.py`).
  - **10 fetchable** (stated benchmark on niftyindices' free feed) — all equity
    broad/sectoral: large-cap (NIFTY 100), flexi/ELSS (NIFTY 500), large&mid
    (NIFTY LargeMidcap 250), mid (NIFTY Midcap 150), small (NIFTY Smallcap 250),
    index (NIFTY 50), IT (NIFTY IT), pharma (NIFTY Pharma).
  - **9 not free** — every non-equity-broad kind: balanced-advantage & hybrid-debt
    (NIFTY hybrid composite debt 65:35 / 15:85), corporate-bond (A-II), short-duration
    debt, gilt (NIFTY All Duration G-Sec), aggressive-hybrid (CRISIL), gold FoF
    (priced separately), and 2 international FoFs (Russell 3000 Growth, NASDAQ 100 —
    USD benchmarks). Confirms the Thread-5 pattern: only mainstream NIFTY equity
    benchmarks are on the free feed.
- **Conceptual finding recorded** (ARCHITECTURE → External-metric parity): portfolio
  α/β is computed vs **one global benchmark**, not aggregated from per-asset α/β
  (which is only meaningful when all assets share one benchmark). So a portfolio's
  α/β is lost only when the *global* benchmark is absent — not when an individual
  fund's own benchmark is unsourceable.
- [ ] **Follow-on A (gated — ASK):** add the 10 fetchable equity funds to
  `data/funds/vro_funds.csv` (need VRO plan ids) so `tests/integration/test_vro_parity.py`
  validates their Beta/Alpha against VRO.
- [ ] **Follow-on B (gated — ASK):** build new `port/*.toml` test portfolios from
  catalogued funds (portfolio α/β needs only mfapi NAVs + the shipped NIFTY TRI).

### Remaining candidate detail (unprioritized)

- **Audits (read-then-fix).**
  - *Docs staleness sweep* — README / QUICKSTART / ARCHITECTURE / DATA_REFRESH /
    OUTPUTS / TESTING / CONTRIBUTING vs. current code (several touched this round).
  - *Test-suite hygiene* — marker correctness, network-tier cost/runtime, any
    flakiness; the live VRO/niftyindices tier now takes ~2.5 min.
- **Under-watched data sources** (existing backlog item): add freshness/registry
  handling for `ppf_interest_rates.csv`, the REC coupon table, and
  `data/reference/gold_monthly_inr.csv` if stable feeds exist; else document the manual cadence.
- **niftyindices steady-state confirmation** (existing open item): verify the
  ≤once/day on-run refresh advances `NIFTY ... CSV` + `.last_fetched.json` over
  several real runs — one verified hit is not proof of unattended reliability.
- **Reporting helpers from `attic/`** — DONE Thread 6 (2026-06-19). The six
  reporting helpers are now on the live `TimeseriesReturn`. The alignment helpers
  (`align_with`/`clip_to_overlap`/`aligned_to`/`interpolated`) remain parked until
  cross-asset analysis needs them.
- **Typing decision** (deferred): whether to adopt `pandas-stubs` +
  `ignore_missing_imports=off` to verify Series/DataFrame contracts, vs. the
  current light-touch mypy config.
- **Deferred / gated, not part of this round unless the user says so:** SGB
  premature-redemption + HTM-vs-redeemable (back burner); money-vault integration
  (needs explicit go-ahead). *(The 3 hybrid/debt + Franklin Beta/Alpha are now
  CLOSED as out of scope — no free benchmark source; see Thread 5.)*

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
- [x] **Phase F: Integration + docs + release tag (2026-06-17).** Closed. `test_main_e2e.py`; ARCHITECTURE/TESTING/CONTRIBUTING docs; `v0.1-salvage` tagged + pushed (`1b53d5c`); decided to keep the repo public.
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
