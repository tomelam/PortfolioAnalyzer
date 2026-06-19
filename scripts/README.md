# scripts/

Helper scripts grouped by purpose. None are imported by the application; they
are operator conveniences and one-off investigations. App invocation itself is
the `portfolio-analyzer` entry point (see `QUICKSTART.md`).

## Orchestration / rendering

| Script | What it does |
|---|---|
| `render-all.sh` | Render every `port/*.toml` (or named ones) to a PNG + CSV under `outputs/<name>/`. |
| `run_metrics_and_save_plot.sh` | Render a single portfolio to `outputs/<name>/{<name>.png,<name>.csv}`. |
| `plot_all.sh` | PNG-only render of every `port/*.toml` into a chosen output dir. |
| `run_all_metrics_to_csv.sh` | One combined headline-metrics CSV across all portfolios. |
| `run_all_configs.sh` | Run the analyzer against every `config/*.toml` (CLI-flag hand-test fixtures). |
| `single-asset-type.sh` | Run the single-asset sanity portfolios in sequence (`-H` for headless). |

## Analysis tools

| Script | What it does |
|---|---|
| `fetch_vro_metrics.py` | Collector CLI: fetch Value Research Online's published figures for the mapped funds and print VRO vs ours vs Δ (returns, Mean/SD/Sharpe/Sortino, and Beta/Alpha where the benchmark is sourceable). `--json` writes a snapshot. Needs the `[browser]` extra. |
| `ppf_annualized_interest_rate.py` | Diagnostic: compute the overall annualized PPF rate from the rate CSV. |

## Discovery spikes (throwaway diagnostics — kept for the record)

These are **throwaway diagnostics**: one-off scripts written to map an external
site's behaviour. They are not run by CI or the app and may rot as those sites
change. We keep them deliberately — they document *how* a non-obvious finding
was established, so the next person can re-derive or re-verify it rather than
re-discover it from scratch. Each carries its own conclusion in its docstring.

| Script | What it established |
|---|---|
| `vro_discover_endpoints.py` | VRO's per-fund API surface — that the risk-ratio family (SD/Sharpe/Sortino/Beta/Alpha/Mean) loads from `/funds/risk-ratios-tab-data/` as an HTML fragment keyed on the fund short name, not the `/api/` JSON routes. |
| `probe_niftyindices_benchmarks.py` | Whether each VRO fund's stated benchmark is fetchable from niftyindices. Result: only **NIFTY 100** (ICICI Bluechip) is; the tiered/hybrid debt benchmarks are not. |
| `probe_niftyindices_indexmaster.py` | *Why* — niftyindices' free endpoint serves an index iff it's in the live-watch master (`LiveIndicesWatch_new.json`); the hybrid/debt benchmarks are absent from it (endpoint-coverage gap, not a naming problem). |

See `docs/ARCHITECTURE.md` → *External-metric parity (Value Research Online)*
for the distilled conclusions these spikes produced.
