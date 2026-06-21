# Architecture

PortfolioAnalyzer is a Python CLI that ingests a portfolio TOML, loads each
asset's price/return history from heterogeneous sources, aligns everything onto
a common daily calendar, and reports CAGR / volatility / Sharpe / Sortino /
alpha / beta / drawdowns against a benchmark and risk-free rate.

## Pipeline at a glance

```
┌─────────────┐
│ portfolio   │  examples/port/port-*.toml  (asset list + weights + asset_allocation)
│   TOML      │
└─────┬───────┘
      │
      ▼
┌──────────────────────────────┐
│  Loaders (per-asset)         │
│  - mutual_fund_loader        │  mfapi.in JSON (live)
│  - ppf_loader                │  static rate CSV → synthetic CIV
│  - scss_loader               │  NSI HTML scrape → synthetic CIV
│  - sgb_holdings + sgb_tranches│ per-tranche CIV: LBMA gold spot + USD coupons (DEXINUS FX)
│  - gold_loader               │  daily LBMA USD/oz CSV (auto-refreshed)
│  - benchmark_loader          │  NIFTY Total Return CSV
│  - risk_free_loader          │  India 10Y bond CSV → daily series
└──────────────┬───────────────┘
               │  per-asset pd.Series (NAV / CIV) or DataFrame
               ▼
┌──────────────────────────────┐
│  timeseries.asset.from_civ   │  NAV → (civ, ret, cumret) views
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  PortfolioTimeseries         │
│  - combined_civ_series       │  normalized weighted CIV on common bday cal
│  - effective_window          │  start/end + which asset set each bound
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  metrics.py (pure functions) │  cagr / volatility / sharpe / sortino /
│                              │  max_drawdown / max_drawdowns
│  TimeseriesReturn methods    │  alpha_capm / beta_capm /
│                              │  alpha_regression / beta_regression
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  visualizer + main.py        │  CSV / snapshot / matplotlib + stdout report
└──────────────────────────────┘
```

## Module map

All application code lives in one flat package, `portfolioanalyzer/`; the module
names below are relative to it (e.g. `main.py` → `portfolioanalyzer/main.py`,
`loaders/gold.py` → `portfolioanalyzer/loaders/gold.py`). The canonical entry
points are `./pa` and `python -m portfolioanalyzer.main`.

| Module | Role |
|---|---|
| `main.py` | CLI orchestration: argparse, config merge, pipeline driver, reporting |
| `loaders/*.py` | One per asset class; ingest source data → standardized Series/DataFrame |
| `synthetic_civ.py` | PPF / SCSS interest-rate → daily-equivalent CIV |
| `civ_to_returns.py` | CIV → daily/monthly returns (with `pct_change`) |
| `timeseries/returns.py` | `TimeseriesReturn` class: the alpha/beta methods + thin delegates to `metrics.py` + the reporting helpers (see *Reporting helpers* below) |
| `timeseries/civ.py` | `TimeseriesCIV` class: validates name='value'; `to_returns` + hand-rolled `max_drawdowns` |
| `timeseries/asset.py` | `AssetTimeseries` dataclass holding `civ` / `ret` / `cumret` views |
| `timeseries/portfolio.py` | `PortfolioTimeseries`: weighted aggregation; the two CIV-bug fixes live here |
| `metrics.py` | **Pure-function math layer.** All Sharpe/Sortino/Vol/CAGR/drawdown logic |
| `portfolio_calculator.py` | `calculate_portfolio_allocations` + `calculate_gains_cumulative` |
| `bond_calculators.py` | `calculate_variable_bond_cumulative_gain` + `term_locked_rate_series` (SCSS term-locked rollover) |
| `fund_lifecycle.py` | Inauguration + DEFUNCT-status detection; writes the per-asset sibling CSV |
| `drawdowns_csv.py` | Per-drawdown sibling CSV writer |
| `sgb_holdings.py` | Per-tranche SGB valuation: `sgb_holding_civ(tranche, grams, gold, fx)` — USD CIV, coupons converted via USD/INR |
| `sgb_tranches.py` | SGB tranche reference data + lookup API |
| `loaders/sgb_redemptions.py` | Ingest RBI press-release redemption prices → `data/funds/sgb_redemptions.csv` (event cadence; B2 data layer) |
| `loaders/gold.py` | Daily LBMA USD/troy-ounce CSV → per-gram price series (auto-refreshed) |
| `visualizer.py` | Matplotlib plotting + drawdown printout; embeds PNG `tEXt` metadata + provenance footnote |
| `output_metadata.py` | Pure formatters for the metrics block / provenance / PNG `tEXt` payload |
| `utils.py` | `info` / `dbg` / `to_cutoff_date` |
| `data_loader.py` | Legacy aggregator; re-exports loaders for back-compat |

## Key design decisions

### Pure-function math layer

`metrics.py` is the canonical home for CAGR, volatility, Sharpe, Sortino,
`max_drawdown`, and `max_drawdowns`. The class methods on `TimeseriesReturn`
delegate to it. Pure functions are trivially unit-tested with hand-computed
inputs (see `tests/unit/test_metrics.py`, 20 tests).

### Reporting helpers on `TimeseriesReturn`

Six optional, ad-hoc reporting/export methods sit alongside the metric
delegates. They are *not* wired into the CLI pipeline — they exist for
interactive/library use (notebooks, scratch scripts, comparison runs). The
console-printing ones write to **stderr** via `utils.info` so they never
pollute a piped CSV/LaTeX payload on stdout.

| Method | Purpose |
|---|---|
| `info_summary(name)` | Print a compact structural summary — shape, date range, NaN count, non-zero count. |
| `describe_as_report(name)` | Print descriptive stats — observations, missing/non-zero, mean, std, min, max. |
| `to_csv_report(path, name)` | Write a one-row summary CSV (structure + descriptive stats). |
| `to_latex_table(compare_to, name, title, label)` | Return a LaTeX metrics table (CAGR / Max DD / Vol / Sharpe / Sortino); optional second column for a comparison series. |
| `compare_to(other, …)` | Print a side-by-side metrics comparison over the two series' common dates (needs ≥30 overlapping dates). |
| `as_rolling(window, method)` | Return a new `TimeseriesReturn` of a rolling `mean`/`std`/`median`/`min`/`max` over the value series. |

A private `_summary_metrics(ts, …)` builds the ordered `(label, value,
is_percent)` rows shared by `to_latex_table` and `compare_to`. Its
annualized-return notion is `cagr()` and its annualized-vol notion is
`volatility()` (the older parked code's removed `annualized()` dict). The
`is_percent` flag drives formatting: CAGR / Max Drawdown / Volatility render as
percents, Sharpe / Sortino as plain ratios. These were revived from
`attic/timeseries_return_helpers.py` (Thread 6, 2026-06-19); see
`tests/unit/test_reporting_helpers.py`.

### Portfolio CIV is unit-free and daily

`PortfolioTimeseries.combined_civ_series` does two non-obvious things that
together fix the two CIV bugs Phase D shook out:

1. **Normalize each asset's CIV to 1.0 at the common start before weighting.**
   Otherwise an asset with raw NAV ~₹2500 (PPF) dominates one with NAV ~₹18
   (a mutual fund) regardless of intended weight.
2. **Reindex every asset onto a common business-day calendar with ffill before
   joining.** A plain `pd.concat(join="inner")` collapses the portfolio CIV
   to the *intersection* of dates — sparse when a coarse-grained asset like PPF
   is present (gold is now daily) — and the daily `sqrt(252)` annualization
   downstream then over-states volatility ~10×.

Two TDD test files pin these contracts:
`tests/unit/test_portfolio_civ_normalization.py` and
`tests/unit/test_portfolio_civ_frequency.py`.

### Synthetic CIVs

PPF / SCSS don't have NAV histories. `synthetic_civ.py` rebuilds
an equivalent CIV from declared/historical interest rates. PPF specifically:
monthly accrual on the year's opening principal, with yearly credit in March
that compounds for the following year.

### SCSS valuation: term-locked rollover

SCSS is valued by `bond_calculators.calculate_variable_bond_cumulative_gain`
(`term_years` set). Two facts drive the model:

1. **Reinvestment must be assumed for metrics.** SCSS is a fixed-term scheme,
   but to characterise it *as an asset* over a backtest window the principal has
   to roll into a fresh SCSS at each maturity. Letting a matured sleeve fall to
   flat cash would make the metrics describe "SCSS then idle cash," not SCSS —
   distorting CAGR/vol/Sharpe. So the CIV compounds continuously to the window
   end; there is no cash tail.
2. **The rate locks per term.** A real SCSS account fixes its rate at opening
   for the whole term; later quarterly government revisions do *not* touch an
   open account. So the applicable rate is **locked per `term_years`-year term
   and re-looked-up only at each rollover boundary** (`term_locked_rate_series`)
   — a step function, not the published rate floated daily. Rollover boundaries
   are anchored at the holding's `purchase_date` (TOML), or the analysis-window
   start when omitted; `term_years` defaults to 5.

The generic `term_years=None` path keeps the old continuous (daily-floated) rate
behaviour for any non-SCSS variable-rate use. This is distinct from SGB, which
is a one-time bond redeemed at maturity (see *SGB valuation* below), not a
rolled-over deposit.

### Portfolio effective window

`combined_civ_series` only spans `[max(asset.start), min(asset.end)]` because
every asset has to have data for every point in the joined series. The
companion `effective_window()` method exposes those bounds plus the *names*
of the assets that set each cutoff, so `main.py` can print a banner like
"End set by 'ICICI Prudential Bluechip Fund'" — useful when a portfolio
carries one defunct fund and the user would otherwise wonder why metrics
end months in the past. `main.py` also trims `benchmark_returns_series` at
`<= end` so alpha/beta are computed over the same window the headline
metrics use.

### Risk-free rate

`risk_free_loader` reads India 10Y bond yields, converts percent → decimal,
and aligns onto the portfolio's date index. `main.py` converts the annual rate
to a per-period geometric rate before passing into Sharpe/Sortino, so
`metrics.sharpe(returns, risk_free_rate=...)` expects a pre-normalized
per-period rate.

### Data freshness as a correctness invariant

The program's job is correct metrics. Stale **reference** data (the benchmark,
the risk-free rate, for gold/SGB-bearing portfolios the gold price, and for
SGB-bearing portfolios the USD/INR rate)
silently corrupts results — a stale benchmark skews alpha/beta, a stale
risk-free skews Sharpe/Sortino/alpha, a stale gold price freezes gold/SGB
valuation — so freshness is not a user-tunable policy; it is an invariant the
program defends. The model:

- **Scope = reference data only.** Benchmark (NIFTY TRI), risk-free (FRED),
  gold (LBMA Gold Price), and USD/INR (FRED DEXINUS) are the auto-refreshable
  upstream feeds; the gold feed is only gated for portfolios that hold gold or
  SGBs, and the FX feed only for portfolios that hold SGBs. Mutual-fund NAVs
  are fetched live every run, so they are current by construction. PPF is a
  manual CSV with no feed to pull (and sparse by design — one row per rate
  change), so it can only warn, never block.
- **Block by default.** If the program cannot certify the reference data is
  current, it stops rather than print degraded metrics. The single override is
  `--allow-stale`, which proceeds after printing a warning that names the
  affected metrics (so the user knows *what* is degraded, not just *that*
  something is).
- **Auto-refresh is built in, not a flag.** When a reference source is behind,
  the program refreshes it before computing. Refreshing *is* the remedy; an
  age-check that doesn't refresh would just nag.
- **No magic-number thresholds.** "Behind" is not an arbitrary age tolerance;
  it is *having fallen off the feed's own publication cadence* — older than the
  most recent **business day** (NIFTY) or **month** (FRED). A successful fetch
  yields the latest the source offers, so it is *current by definition*,
  whatever the calendar says. (Accepted edge case: on an exchange holiday the
  benchmark can look "behind" and trigger one harmless refresh that finds
  nothing new — cheaper than maintaining a holiday calendar.)
- **niftyindices is contacted at most once per day.** It is an anti-scrape host
  that blocks the source IP, so this is the one genuine guard in the system:
  every *attempt* (success or failure) is stamped in `data/.last_fetched.json`,
  and a second attempt the same day is suppressed. FRED is a clean public CSV
  with no such risk, so it refreshes whenever it is behind.
- **Provenance in the output.** For each reference source the run reports
  `last_date` (the latest data point — what bears on correctness), `fetched_at`
  (when our copy was pulled), and `attempted_at` (the last refresh attempt),
  read from the stamp file. It rides three channels: echoed to **stderr** every
  run, drawn as a small **footnote** on the PNG, and embedded in the PNG's
  `tEXt` **metadata** (alongside a human-readable metrics block — see
  `output_metadata.py` and `docs/OUTPUTS.md`). This is documentation of result
  quality, not decoration.
- **Deterministic modes opt out cleanly.** `--as-of DATE` and `--replay-from`
  neither fetch nor block: data pinned through the as-of date is current *as of
  that date*.

Deliberately **not** done: FRED publishes a release calendar, so risk-free
refreshes could be scheduled to release days only. Wiring up a release-calendar
dependency to save a handful of harmless FRED hits is not worth the coupling;
once-a-day "check if behind" is the pragmatic floor. Revisit only if a cheap,
reliable publication oracle for *both* feeds becomes available.

Implementation lives in `loaders/data_update.py` (registry, fetchers,
cadence-gated refresh) and `main.py` (the block/allow-stale gate + provenance
line). Operational details — the `browser` extra for the
niftyindices stealth fetch, manual refresh — are in
[`DATA_REFRESH.md`](DATA_REFRESH.md).

### External-metric parity (Value Research Online)

We validate our own metrics against an external authority (Value Research
Online) so a methodology drift can't pass silently. `loaders/vro.py` fetches
VRO's published figures for the mapped funds (`data/funds/vro_funds.csv`) and the
live wire test (`tests/integration/test_vro_parity.py`) asserts ours match.
Reconciled methodology:

- **Trailing returns** — VRO uses **point-to-point daily** NAVs (not month-end),
  so the analog is `metrics.cagr` over the trailing window (`trailing_cagr_pct`).
- **Risk ratios (Mean / Std Dev / Sharpe / Sortino)** — trailing-3Y, **monthly**;
  mirrored by `trailing_risk_ratios` (`metrics.*` with `periods_per_year=12`).
- **CAPM Beta / Alpha** — measured against each fund's **stated benchmark**, so
  they need that benchmark's own return series. The framing is *our* correctness,
  not parity for its own sake: where we can source the benchmark we compute
  correct Beta/Alpha for the fund, and VRO parity is a free cross-check on top.

**niftyindices benchmark-feed coverage (probe finding, 2026-06-19).** Sourcing
those benchmark TRIs runs into a hard limit. The niftyindices free historical
endpoint serves an index **iff it is in the live-watch master**
(`iislliveblob.niftyindices.com/jsonfiles/LiveIndicesWatch_new.json`, ~131
indices). Two endpoint shapes: equity indices answer on
`getTotalReturnIndexString` (a real TRI), plain debt/gilt indices answer on
`getHistoricaldatatabletoString` (OHLC, `CLOSE` column) — confirmed with
`NIFTY GS 10YR`/`NIFTY GS COMPSITE` controls. Consequence for our five funds:

| Fund benchmark | On the feed? |
|---|---|
| NIFTY 100 (ICICI Bluechip) | **yes** — TRI endpoint |
| NIFTY 50 Hybrid Composite Debt 65:35 / 15:85 | no — absent from the master |
| NIFTY Corporate Bond Index A-II | no — absent from the master |
| Russell 3000 Growth (Franklin US FoF) | n/a — not a NIFTY index |

So niftyindices yields exactly **one** benchmark, NIFTY 100, which is also the
only fund VRO publishes both Beta *and* Alpha for. `VROFund.benchmark_index`
records the fetchable niftyindices name (empty where none); `fetch_benchmark_tri`
pulls it via the same stealth path; only funds with a `benchmark_index` get
Beta/Alpha asserted. The reusable probe spikes that established this live under
`scripts/probe_niftyindices_*.py` (see [scripts/README.md](../scripts/README.md)).

**The 3 hybrid/debt benchmark series are not freely sourceable — out of scope
(Thread 5, 2026-06-19).** We exhaustively checked every free avenue for the
stated benchmarks of HDFC Balanced Advantage (65:35), HDFC Hybrid Debt (15:85),
and ICICI Corporate Bond (A-II); none yields a usable time series:

| Avenue | Result |
|---|---|
| niftyindices `getTotalReturnIndexString` (TRI) + `getHistoricaldatatabletoString` (HIST) | 0 rows for all 3 with **exact** registered names + variants; NIFTY 100 control returns 136 rows on both. Definitive endpoint-coverage gap (`probe_niftyindices_hybrid_exact.py`). |
| niftyindices index **product pages** | No chart-data/time-series XHR — only factsheet metadata (`probe_niftyindices_productpage.py`). |
| niftyindices **factsheets** (monthly PDF) | One monthly snapshot (level + trailing returns); overwritten each month, not archived ⇒ no backfillable series. |
| Morningstar / MoneyControl fund pages | Risk ratios computed vs Morningstar **category/standard indices** ("Nifty 50 TR INR" for the hybrids), not the stated benchmark; β clusters ~1 regardless of asset mix (fund-vs-category). **No stated-benchmark series exposed.** |

These indices are total-return-by-construction (the 65:35/15:85 blends NIFTY 50
*TR* with a debt TR index; A-II is a bond TR index), so the problem is purely
retrievability, not a price-vs-TRI gap. Payoff was thin anyway — VRO publishes
Beta-only for the 2 HDFC hybrids and nothing for ICICI Corp Bond. We therefore
treat these 3 funds' stated-benchmark Beta/Alpha as **out of scope (not freely
sourceable)**, the same posture as Franklin below: `benchmark_index` stays empty
for them in `data/funds/vro_funds.csv`, so the parity test simply skips them. For the
reader-facing summary of *which metrics survive* without a benchmark (everything
except Alpha/Beta), see [README.md → Metrics](../README.md#metrics).

**Franklin US FoF is a deliberate Beta/Alpha omission, not a gap.** It is a
USD-denominated feeder fund benchmarked to Russell 3000 Growth (a USD index),
while its NAV is INR — so a CAPM regression would conflate equity beta with
USD/INR currency moves. There is no single correct convention, no stable free
Russell TR source, and VRO itself publishes neither Beta nor Alpha for it. We
omit them rather than manufacture false precision; revisiting would mean an
explicitly currency-adjusted variant, a deliberate modelling choice.

**Portfolio Alpha/Beta is measured against one global benchmark — not aggregated
from per-asset benchmarks.** The portfolio's Beta/Alpha (`main.py`) is computed by
regressing the *aggregate* portfolio return series on a **single** benchmark
(the configured NIFTY TRI) via `metrics.beta_capm` / `metrics.alpha_capm`. It does
not use each fund's own stated benchmark and does not combine per-asset α/β. So a
portfolio's α/β is unavailable only when the *global* benchmark itself is absent
(`use_benchmark=false`, or too few aligned points) — never because an individual
asset's own benchmark can't be sourced. Why not aggregate per-asset α/β? It is
exact (β_p = Σ wᵢβᵢ, α_p = Σ wᵢαᵢ) **only when every asset shares one benchmark,
window, and risk-free** — and in that case it is *identical* to the aggregate
regression we already run (the per-asset form is just an attribution of it). With
*different* per-asset benchmarks (equity vs NIFTY, debt vs a bond index, gold/PPF
vs none) the betas measure sensitivity to different market factors and cannot be
summed. Per-fund benchmarks (`data/funds/vro_funds.csv`, `data/funds/fund_catalog.csv`) are
therefore used only for **per-fund CAPM validation** against external authorities,
not for the portfolio number. See the fund catalog in `data/funds/fund_catalog.csv`
(which benchmarks are freely sourceable) and the KANBAN fund-catalog thread.

### SGB valuation: hold-to-maturity by default

Sovereign Gold Bonds are modelled per tranche (each tranche is a distinct
investment) as `units × gold_per_gram(t) + Σ(coupon_inr / fx_at_coupon_date)` —
i.e. marked to gold spot (the LBMA Gold Price, USD/gram) plus accrued coupons
(`sgb_holdings.sgb_holding_civ`). For the holdings and analysis windows we care
about, this **is** the hold-to-maturity (HTM) valuation, and HTM is the default
everywhere SGBs appear (plots, charts, metric tables).

The whole CIV is in **USD**. The capital leg is USD/gram LBMA gold; the 2.5%
coupon is contractually a **rupee** cash amount, so each coupon is converted to
USD at **its own payment date's** USD/INR rate (FRED `DEXINUS`, INR per USD:
`usd = inr / DEXINUS`). This keeps the series a single consistent USD number
rather than summing rupees into dollars — the bug Part B1 fixed, where a ₹63
coupon added to a ~$133/gram capital number turned each semiannual coupon into
~47% of capital. FX touches only the discrete coupon cash amounts, never the
gold price path (the sanctioned "contractual cash value" FX exception). So a
gold-only run depends on LBMA alone; an SGB run depends on LBMA gold **and** the
`DEXINUS` FX feed — the honest decoupling of gold and SGB.

**Early-redeemed holdings (B2).** An early-redeemed SGB is modelled as a
*distinct holding type*, not a global view toggle: a `[[sgb]]` entry carries
`valuation = "redeemed"` (default `"htm"`) plus an optional `redemption_date`.
Such a holding behaves as HTM up to its redemption date; on that date it exits at
RBI's announced premature-redemption price (an **INR**, IBJA-3-day-average figure
— the contractual cash value, for which the Indian source is correct) converted
to USD at the redemption date's `DEXINUS`, and from then on the CIV is **carried
flat as cash** (coupon accrual freezes — there is no holding left to pay coupons
on). Because each `[[sgb]]` holding is already its own portfolio asset with its
own weight and label (`SGB <tranche> (redeemed <date>)` vs the bare `SGB
<tranche>` for HTM), a portfolio can hold the same tranche **both** ways and the
report compares them side by side — no `-htm`/`-redemption` flag or second output
pipeline. HTM stays the default everywhere it is not overridden. (The terminal
**maturity pin** is a separate, deferred refinement — see below.)

**B2 data layer.** The redemption-price *data* is ingested by
`loaders/sgb_redemptions.py` into `data/funds/sgb_redemptions.csv` (`tranche_id,
redemption_date, kind ∈ {PRE,MAT}, inr_per_gram, source_prid_or_url, tier`);
`lookup_redemption(tranche_id, redemption_date=None)` resolves one row for the
valuation layer (fail-loud on a missing or — when the date is omitted —
ambiguous match). The fetchable source was confirmed by a **fail-loud Step 0**
gate (`scripts/probe_rbi_sgb_redemption.py`): RBI's mobile directory
(`BS_SwarnaBharat.aspx`) is CAPTCHA-walled, but the press releases themselves are
reachable over plain `requests` — enumerate via the open `SearchResults.aspx`
endpoint, then parse each `BS_PressReleaseDisplay.aspx?prid=<N>`. This is an
*event*-cadence table (irregular redemption dates, no calendar frontier), so it
is a standalone loader rather than a cadence-gated `data_update` `DataSource`.
See `docs/DATA_REFRESH.md` → *SGB redemption prices*.

**Maturity pin (deferred, dormant).** At its **8-year maturity** an SGB is
redeemed by RBI at a *contractual* price — the simple average of the closing
999-gold price over the week (Mon–Fri) preceding maturity — **not** that single
day's spot. The maturity pin would, at `maturity_date`, replace the daily
gold-spot capital leg with this RBI **maturity** redemption price (`kind = MAT`
in `sgb_redemptions.csv`), INR→USD via `DEXINUS`, and carry it flat as cash
thereafter. It is the at-maturity twin of the early-redemption (`PRE`) path
already built — same `sgb_holding_civ` mechanism, but triggered by the tranche's
own `maturity_date` and a `MAT` row rather than a user-supplied `redemption_date`
and a `PRE` row. Its only effect is *terminal-day accuracy*: the week-average
contractual figure instead of one day's LBMA spot, on the very last day of an
8-year series.

It is **dormant on two counts** and so left unimplemented:

1. *Out of window.* Every tranche in `data/funds/sgb_tranches.csv` is scoped to
   Feb 2020+, so the earliest maturity is **2028**, beyond the end of the daily
   gold/FX data (~2026). `sgb_holding_civ` clips the series at
   `min(maturity_date, gold_end)` = `gold_end`, so the maturity date never falls
   inside the analysis window and HTM coincides with the gold-spot proxy there.
2. *No data yet.* The `MAT` rows that exist (from the redemption backfill) are all
   for **pre-2020** tranches that have already matured — none are in the tranche
   reference. The in-scope 2020+ tranches have no `MAT` row yet.

**Activation** needs *both*: the daily gold series to extend past a tranche's
2028+ maturity, and a `MAT` redemption row for that tranche. Until then there is
nothing to test and nothing it would change. See the deferred backlog item in
`docs/KANBAN.md` and the plan `~/.claude/plans/sgb-maturity-pin.md`.
