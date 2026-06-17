# Architecture

PortfolioAnalyzer is a Python CLI that ingests a portfolio TOML, loads each
asset's price/return history from heterogeneous sources, aligns everything onto
a common daily calendar, and reports CAGR / volatility / Sharpe / Sortino /
alpha / beta / drawdowns against a benchmark and risk-free rate.

## Pipeline at a glance

```
┌─────────────┐
│ portfolio   │  port/port-*.toml  (asset list + weights + asset_allocation)
│   TOML      │
└─────┬───────┘
      │
      ▼
┌──────────────────────────────┐
│  Loaders (per-asset)         │
│  - mutual_fund_loader        │  mfapi.in JSON (live)
│  - ppf_loader                │  static rate CSV → synthetic CIV
│  - scss_loader               │  NSI HTML scrape → synthetic CIV
│  - rec_bond_loader           │  TOML coupon → synthetic CIV
│  - sgb_holdings + sgb_tranches│ per-tranche CIV from IBJA gold spot
│  - gold_loader               │  monthly INR CSV
│  - benchmark_loader          │  NIFTY Total Return CSV
│  - risk_free_loader          │  India 10Y bond CSV → daily series
└──────────────┬───────────────┘
               │  per-asset pd.Series (NAV / CIV) or DataFrame
               ▼
┌──────────────────────────────┐
│  asset_timeseries.from_civ   │  NAV → (civ, ret, cumret) views
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

| Module | Role |
|---|---|
| `main.py` | CLI orchestration: argparse, config merge, pipeline driver, reporting |
| `*_loader.py` | One per asset class; ingest source data → standardized Series/DataFrame |
| `synthetic_civ.py` | PPF / SCSS / REC interest-rate → daily-equivalent CIV |
| `civ_to_returns.py` | CIV → daily/monthly returns (with `pct_change`) |
| `timeseries/returns.py` | `TimeseriesReturn` class: the alpha/beta methods + thin delegates to `metrics.py` |
| `timeseries/civ.py` | `TimeseriesCIV` class: validates name='value'; `to_returns` + hand-rolled `max_drawdowns` |
| `timeseries/asset.py` | `AssetTimeseries` dataclass holding `civ` / `ret` / `cumret` views |
| `timeseries/portfolio.py` | `PortfolioTimeseries`: weighted aggregation; the two CIV-bug fixes live here |
| `metrics.py` | **Pure-function math layer.** All Sharpe/Sortino/Vol/CAGR/drawdown logic |
| `portfolio_calculator.py` | `calculate_portfolio_allocations` + `calculate_gains_cumulative` |
| `bond_calculators.py` | `calculate_variable_bond_cumulative_gain` (used by `rec_bond_loader`) |
| `fund_lifecycle.py` | Inauguration + DEFUNCT-status detection; writes the per-asset sibling CSV |
| `drawdowns_csv.py` | Per-drawdown sibling CSV writer |
| `sgb_holdings.py` | Per-tranche SGB valuation: `sgb_holding_civ(tranche, grams, gold)` |
| `sgb_tranches.py` | SGB tranche reference data + lookup API |
| `gold_loader.py` | Monthly INR/troy-ounce CSV → per-gram price series |
| `visualizer.py` | Matplotlib plotting + drawdown printout; embeds PNG `tEXt` metadata + provenance footnote |
| `output_metadata.py` | Pure formatters for the metrics block / provenance / PNG `tEXt` payload |
| `utils.py` | `info` / `dbg` / `warn_if_stale` / `to_cutoff_date` |
| `data_loader.py` | Legacy aggregator; re-exports loaders for back-compat |

## Key design decisions

### Pure-function math layer

`metrics.py` is the canonical home for CAGR, volatility, Sharpe, Sortino,
`max_drawdown`, and `max_drawdowns`. The class methods on `TimeseriesReturn`
delegate to it. Pure functions are trivially unit-tested with hand-computed
inputs (see `tests/unit/test_metrics.py`, 20 tests).

### Portfolio CIV is unit-free and daily

`PortfolioTimeseries.combined_civ_series` does two non-obvious things that
together fix the two CIV bugs Phase D shook out:

1. **Normalize each asset's CIV to 1.0 at the common start before weighting.**
   Otherwise an asset with raw NAV ~₹2500 (PPF) dominates one with NAV ~₹18
   (a mutual fund) regardless of intended weight.
2. **Reindex every asset onto a common business-day calendar with ffill before
   joining.** A plain `pd.concat(join="inner")` collapses the portfolio CIV
   to the *intersection* of dates — effectively monthly when gold or PPF is
   present — and the daily `sqrt(252)` annualization downstream then
   over-states volatility ~10×.

Two TDD test files pin these contracts:
`tests/unit/test_portfolio_civ_normalization.py` and
`tests/unit/test_portfolio_civ_frequency.py`.

### Synthetic CIVs

PPF / SCSS / REC bonds don't have NAV histories. `synthetic_civ.py` rebuilds
an equivalent CIV from declared/historical interest rates. PPF specifically:
monthly accrual on the year's opening principal, with yearly credit in March
that compounds for the following year.

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

The program's job is correct metrics. Stale **reference** data (the benchmark
and the risk-free rate) silently corrupts results — a stale benchmark skews
alpha/beta, a stale risk-free skews Sharpe/Sortino/alpha — so freshness is not
a user-tunable policy; it is an invariant the program defends. The model:

- **Scope = reference data only.** Benchmark (NIFTY TRI) and risk-free (FRED)
  are the auto-refreshable upstream feeds. Mutual-fund NAVs are fetched live
  every run, so they are current by construction. Gold/PPF are manual CSVs
  with no feed to pull — they can only warn, never block.
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
