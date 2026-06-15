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
│  - sgb_loader                │  Wikipedia / static CSV
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
│  - combined_daily_returns    │  weighted sum of asset daily returns
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
| `timeseries.py` | `TimeseriesReturn` class: the alpha/beta methods + thin delegates to `metrics.py` |
| `timeseries_civ.py` | `TimeseriesCIV` class: validates name='value'; `to_returns` + hand-rolled `max_drawdowns` |
| `asset_timeseries.py` | `AssetTimeseries` dataclass holding `civ` / `ret` / `cumret` views |
| `portfolio_timeseries.py` | `PortfolioTimeseries`: weighted aggregation; the two CIV-bug fixes live here |
| `metrics.py` | **Pure-function math layer.** All Sharpe/Sortino/Vol/CAGR/drawdown logic |
| `portfolio_calculator.py` | `calculate_portfolio_allocations` + `calculate_gains_cumulative` |
| `bond_calculators.py` | `calculate_variable_bond_cumulative_gain` (used by `rec_bond_loader`) |
| `visualizer.py` | Matplotlib plotting + drawdown printout |
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

### Risk-free rate

`risk_free_loader` reads India 10Y bond yields, converts percent → decimal,
and aligns onto the portfolio's date index. `main.py` converts the annual rate
to a per-period geometric rate before passing into Sharpe/Sortino, so
`metrics.sharpe(returns, risk_free_rate=...)` expects a pre-normalized
per-period rate.
