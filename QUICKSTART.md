# PortfolioAnalyzer — Quickstart

A practical guide to running analyses, saving plots and CSVs, and sweeping
many portfolios at once. For the long-form description of what the
project does and where the data comes from, see [`README.md`](README.md).
For architecture and design, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Setup (once)

```bash
git clone git@github.com:tomelam/PortfolioAnalyzer.git
cd PortfolioAnalyzer

python3.12 -m venv venv
./venv/bin/python -m pip install -e ".[dev]"
```

The editable install puts a `portfolio-analyzer` console entry point
into `./venv/bin/`. All examples below use `./venv/bin/portfolio-analyzer`
so they work without an activated virtualenv. If you activate the venv
(`source venv/bin/activate`), you can shorten that to just
`portfolio-analyzer`.

> The old `./venv/bin/portfolio-analyzer …` form still works for back-compat,
> but the entry-point form is preferred — shorter, no interpreter path.

---

## Run one portfolio (the most common case)

```bash
./venv/bin/portfolio-analyzer port/port-1.toml
```

That prints metrics to the terminal **and** opens a matplotlib window
with the cumulative-returns plot. The window is interactive — pan, zoom,
press keys (see plot title for hints).

Common variations:

```bash
# Use monthly metrics instead of the default daily.
./venv/bin/portfolio-analyzer --metrics-method monthly port/port-1.toml

# Last 5 years only.
./venv/bin/portfolio-analyzer --lookback 5Y port/port-1.toml

# Save the plot as PNG (no window pops up) and dump metrics as CSV.
./venv/bin/portfolio-analyzer \
    --disable-plot-display --output-snapshot --output-csv \
    --output-dir outputs/port-1 \
    port/port-1.toml
```

After the third invocation you'll find:

```
outputs/port-1/port-1.png    ← the plot
outputs/port-1/port-1.csv    ← the metrics as one CSV row
```

---

## CLI flag reference

| Flag | Short | What it does |
|---|---|---|
| *(positional)* `toml_file` |     | The portfolio TOML to analyze. **Required.** |
| `--config FILE` | `-c` | Use this config TOML for runtime defaults (data paths, etc.). |
| `--metrics-method daily\|monthly` |     | Frequency for return/risk calculations. Default `daily`. |
| `--lookback YTD\|1M\|3M\|6M\|1Y\|3Y\|5Y\|10Y` | `-lb` | Trim all series to this trailing period before computing metrics. |
| `--max-drawdown-threshold PCT` | `-dt` | Drawdown threshold in percent. Default 5. |
| `--max-riskfree-delay DAYS` | `-mrd` | Maximum allowed staleness for the risk-free-rate CSV. |
| `--disable-plot-display` | `-dpd` | Don't open the matplotlib window. (Essential for shell scripts and headless runs.) |
| `--output-snapshot` | `-os` | Save the plot as a PNG to the output directory. |
| `--output-csv` | `-co` | Write the metrics as a single-row CSV instead of human-readable stdout. |
| `--output-dir DIR` | `-od` | Where to put `*.png` / `*.csv` output. Default `outputs/`. |
| `--save-golden-data` | `-sgd` | Pickle the intermediate data sets for regression tests. |
| `--quiet` | `-q` | Don't prompt on stale data; proceed as if you said yes. |
| `--debug` | `-d` | Show full Python tracebacks on error. |

---

## Recipes

### See the familiar plot window + printed metrics

```bash
./venv/bin/portfolio-analyzer port/port-everything.toml
```

### Save plot + CSV without opening any window

```bash
./venv/bin/portfolio-analyzer \
    --disable-plot-display --output-snapshot --output-csv \
    --output-dir outputs/port-everything \
    port/port-everything.toml
```

### Compare two portfolios side-by-side

```bash
for p in port-1 port-mf-ppf-gold; do
    ./venv/bin/portfolio-analyzer \
        --disable-plot-display --output-snapshot \
        --output-dir outputs/$p \
        port/$p.toml
done
open outputs/port-1/port-1.png outputs/port-mf-ppf-gold/port-mf-ppf-gold.png
```

### Sweep every portfolio in `port/` with one command

```bash
make all                 # incremental: render only stale/missing outputs
make rerender            # force-rebuild every PNG + CSV (preserves outputs/)
scripts/render-all.sh    # same as `make rerender`, as pure bash
```

Both write to `outputs/<portfolio-name>/` per portfolio. **`outputs/` is
preserved by default** — `make clean` does **not** delete it; only
`make distclean` (with a confirmation prompt) does. See
[`docs/OUTPUTS.md`](docs/OUTPUTS.md) for the full policy.

### Single asset class only

The `port/` directory has tiny single-asset TOMLs for sanity checks:

```bash
./venv/bin/portfolio-analyzer port/port-ppf.toml
./venv/bin/portfolio-analyzer port/port-scss.toml
./venv/bin/portfolio-analyzer port/port-sgb.toml
./venv/bin/portfolio-analyzer port/port-gold.toml
./venv/bin/portfolio-analyzer port/port-rec-bond.toml
```

Or all six in one go via `scripts/single-asset-type.sh`.

### Quiet, batch-friendly run (no prompts)

The staleness gate prompts interactively by default; in scripts use
`--quiet` so it auto-proceeds when data is older than the freshness
threshold:

```bash
./venv/bin/portfolio-analyzer --quiet --disable-plot-display \
    --output-snapshot --output-csv \
    --output-dir outputs/port-1 \
    port/port-1.toml
```

---

## Scripts (in `scripts/`)

| Script | What it does |
|---|---|
| `scripts/render-all.sh` | Loop over every `port/*.toml`. For each: save PNG + CSV under `outputs/<name>/`. |
| `scripts/single-asset-type.sh` | Run the six single-asset sanity portfolios with stdout-only output (matplotlib windows pop up). |
| `scripts/run_metrics_and_save_plot.sh PORT.toml` | One portfolio → `outputs/<name>/` with PNG + CSV. |
| `scripts/ppf_annualized_interest_rate.py CSV` | Standalone PPF rate analysis tool. |

Run any script with `--help` (or read the first few lines) for usage.

---

## Make targets

```bash
make all                  # incremental render of every port/*.toml
make rerender             # force-rebuild every PNG + CSV
make outputs/port-1.png   # render just port-1
make clean                # remove only portfolio_metrics.csv (outputs/ preserved)
make distclean            # rm -rf outputs/ (asks for confirmation)
```

The `Makefile` is parallel-safe: `make -j 4 all` runs four portfolios at
once (mfapi.in is happy to serve concurrent requests).

---

## Portfolio TOML schema (cheat sheet)

```toml
label = "My portfolio"

# Mutual funds — one block per fund. allocation = fraction of portfolio.
[[funds]]
name = "ICICI Prudential Bluechip Fund"
url = "https://api.mfapi.in/mf/120586"
allocation = 0.30
asset_allocation = {equity = 92.69, debt = 0.35, real_estate = 0.0, commodities = 0.0, cash = 6.96}

# Single-block assets — one [section]:
[ppf]
allocation = 0.15
ppf_interest_rates_file = "ppf_interest_rates.csv"

[gold]
allocation = 0.10

[scss]
allocation = 0.05

[rec_bond]
allocation = 0.05
coupon = 5.0

# SGB — one [[sgb]] entry per tranche (different tranches = different
# investments). Same tranche bought multiple times the same day lumps
# into one entry by summing units_grams.
[[sgb]]
tranche_id = "2019-20-IX"     # see data/sgb_tranches.csv for the list
units_grams = 12
allocation = 0.05

[[sgb]]
tranche_id = "2020-21-VII"
units_grams = 6
allocation = 0.05
```

All `allocation` values must sum to 1.0 (±0.01).

---

## Where things land

```
outputs/<portfolio-name>/  ← PNG + CSV when --output-snapshot / --output-csv
tests/golden/              ← reference outputs for regression tests
data/                      ← raw input data (NIFTY, gold, PPF rates, etc.)
KANBAN.md                  ← what's planned / done / in flight
```

---

## Troubleshooting

- **"⚠️ NIFTY data is N days old"** — the bundled CSVs in `data/` are
  refreshed manually; the gate prompts you to proceed. Use `--quiet` in
  scripts or update the file from
  [investing.com](https://in.investing.com/indices/nifty-total-returns-historical-data).
- **`KeyError: 'sgb'`** — the portfolio TOML uses the legacy `[sgb]`
  dict form. Migrate to one or more `[[sgb]]` entries (see schema above).
- **"unknown tranche"** — the `tranche_id` doesn't exist in
  `data/sgb_tranches.csv`. Currently only Feb 2020+ tranches are
  registered. Use `./venv/bin/python -c "from sgb_tranches import
  list_tranches; print(list_tranches()[['tranche_id','issue_date']]
  .to_string(index=False))"` to see all valid IDs.
