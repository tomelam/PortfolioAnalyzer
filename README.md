# Portfolio Analyzer

> **In a hurry?** See [`docs/QUICKSTART.md`](docs/QUICKSTART.md) for the most common
> invocations, the full CLI flag table, and `make all` / `scripts/render-all.sh`
> for batch sweeps. The outputs directory is preserved by default —
> see [`docs/OUTPUTS.md`](docs/OUTPUTS.md) for the policy.

This repository contains the Portfolio Analyzer application. It fetches historical NAV data for Indian mutual funds from [mfapi.in](https://mfapi.in), uses benchmark data from [investing.com](https://investing.com) or [niftyindices.com](https://in.investing.com/indices/nifty-total-returns-historical-data), uses risk-free rate data from [FRED](https://fred.stlouisfed.org), [RBI](https://rbi.org.in), or [DBIE (by advanced-searching "bill" while selecting "Weekly" Report Frequency and "Publication" Function)](https://data.rbi.org.in), and uses monthly-average gold spot prices from the link "Gold price averages in a range of currencies since 1978" on [gold.org](https://www.gold.org/goldhub/data/gold-prices), computes key portfolio performance metrics (such as annualized return, volatility, Sharpe/Sortino ratios, Alpha, Beta, and maximum drawdowns), and visualizes historical returns along with benchmark data.

---

## Features

- **Data Loading & Alignment:**  
  - Fetches NAV data for each fund via API calls.
  - Loads risk-free rate data from CSV. The included file INDIRLTLT01STM.csv is one such file manually downloaded from [fred.stlouisfed.org](https://fred.stlouisfed.org).
  - Uses PPF interest rates manually encoded as CSV in the file `data/funds/ppf_interest_rates.csv`.
  - Scrapes the SCSS interest rates from [The National Savings Institute's table "Senior Citizens' Savings Scheme--Interest Rate Since Inception"](https://www.nsiindia.gov.in/(S(2xgxs555qwdlfb2p4ub03n3n))/InternalPage.aspx?Id_Pk=181).
  - Loads the SGB issue price/unit and redemption price/unit data manually copied from the Wikipedia page [Sovereign Gold Bond](https://en.wikipedia.org/wiki/Sovereign_Gold_Bond).
  - Uses gold futures (GCJ5) manually downloaded as CSV from [https://www.investing.com/commodities/gold-historical-data](https://www.investing.com/commodities/gold-historical-data) and stored in the file `"data/reference/Gold Futures Historical Data.csv"`. It is difficult to source the gold spot price for free, but gold futures front-month contracts closely approximate the gold spot price, especially as the contract nears expiration. This is why PortfolioAnalyzer uses the gold futures front-month contract price as a proxy for the gold spot price.
  - Uses benchmark historical data from [niftyindices.com](https://www.niftyindices.com) or [investing.com](https://investing.com) (deprecated).
  - Aligns data to a common date range across all data sources.
  - **Note — REC 54EC capital-gains bonds are no longer supported.** They were
    removed because the user has no further use for a REC bond (the holding
    matured and was redeemed to cash). They had been modelled identically to
    SCSS — the same fixed-coupon-bond machinery — so SCSS now covers that sleeve
    and nothing is lost. Do not re-add REC support.

- **Portfolio Metrics Calculation:**  
  - Computes annualized return, volatility, Sharpe ratio, Sortino ratio.
  - Calculates Alpha and Beta relative to the benchmark.
  - Identifies significant drawdowns.

- **Visualization:**  
  - Plots historical cumulative returns for the portfolio.
  - Overlays benchmark data and highlights significant drawdown periods.

- **Modular Design:** all application code lives in one flat package,
  `portfolioanalyzer/`.
  - Loaders (one per asset class) in the `portfolioanalyzer/loaders/` package:
    `mutual_fund.py`, `ppf.py`, `scss.py`, `gold.py`, `benchmark.py`,
    `risk_free.py` — plus `sgb_holdings.py` + `sgb_tranches.py` in the package root.
  - Math layer: `metrics.py` (pure-function CAGR / vol / Sharpe / Sortino /
    drawdowns) plus the `portfolioanalyzer/timeseries/` package (`returns.py` /
    `civ.py` / `asset.py` / `portfolio.py`) for the class surface.
  - Bookkeeping: `synthetic_civ.py`, `civ_to_returns.py`, `drawdowns_csv.py`,
    `fund_lifecycle.py`, `portfolio_calculator.py`, `bond_calculators.py`,
    `visualizer.py`, `data_loader.py` (legacy aggregator / re-exports).
  - See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full module
    map and pipeline diagram.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tomelam/PortfolioAnalyzer.git
   cd PortfolioAnalyzer
   ```
2. **Create and activate a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
   If you're using `asdf` to manage Python versions, run:
   ```bash
   asdf install
   asdf reshim python
   ```
3. **Install the package (editable) with its dependencies:**
   ```bash
   pip install -e .            # runtime deps + the portfolio-analyzer* console scripts
   pip install -e ".[dev]"     # + test/lint tooling (pytest, ruff, mypy, …)
   ```
   This installs the pinned runtime stack (pandas, numpy, requests, toml,
   urllib3, beautifulsoup4, statsmodels, matplotlib, …) **and** puts the
   `portfolio-analyzer` / `portfolio-analyzer-update` console scripts on
   `PATH`. The auto-updater's niftyindices fetch needs a stealth browser —
   add it (and the Chromium binary) only if you want live benchmark refresh:
   ```bash
   pip install -e ".[browser]" && playwright install chromium
   ```

   You don't need to keep the venv activated to use the tool — the `./pa`
   wrapper (see [Usage](#usage)) runs the analyzer with the bundled venv
   directly. For other commands, prefix with the venv interpreter, e.g.
   `./venv/bin/python -m pytest`.

---

## Usage

If the portfolio described by the TOML file includes PPF as a component, ensure that the file detailing historical PPF interest rates, `ppf_interest_rates.csv`, is up to date before running the program.

If the portfolio described by the TOML file includes gold as a component, download the CSV file of the gold prices from https://www.investing.com/commodities/gold-historical-data before running the program. Currently, both offshore vaulted gold and gold held in India are priced using the same CSV data.
Run the analyzer with the bundled-venv wrapper `./pa` — no venv activation, no
`PATH` changes (see [`./pa`](#pa--run-without-activating-the-venv) below). Inside
an activated venv the equivalent console command is `portfolio-analyzer …`.
```bash
./pa --help
./pa <path_to_portfolio_toml_file> [options]
./pa examples/port/port-1.toml --max-drawdown-threshold 10 --allow-stale
```

### `./pa` — run without activating the venv

The repo ships a tiny `pa` wrapper that runs the analyzer with the bundled
venv — no `source venv/bin/activate`, no `PATH` changes, nothing touched
outside the project directory. It `cd`s into the project root itself, so it
works from anywhere and in-repo paths resolve:
```bash
./pa examples/port/port-1.toml --max-drawdown-threshold 10 --allow-stale
./pa --help
```
It just `exec`s `venv/bin/python -m portfolioanalyzer.main "$@"`, so editing the code takes effect
immediately. Paths you pass are resolved from the project root; use an absolute
path for a portfolio/config file that lives elsewhere.
The `--max-drawdown-threshold` option (shortcut `-dt`) sets the percentage drawdown that is considered significant to count in the "Drawdowns" statistic. By default, the threshold is set to `5` (5%).

The benchmark name and benchmark/risk-free CSV paths live in the config TOML
(see `examples/config/example_config.toml`); no CLI shortcuts exist for those. Run
`./pa --help` for the authoritative flag list.

⚠️ NOTE: Files downloaded from Investing.com sometimes use different date formats (e.g., %d-%m-%Y vs %m/%d/%Y). Always check the format of the first few rows and set `benchmark_date_format` in your config TOML accordingly — there is no CLI flag for it (see [Benchmark Indices](#benchmark-indices-config-file-only-not-in-command-line-options) below).

Mac users might notice messages like `2025-02-04 20:00:14.220 python[20791:454371] +[IMKClient subclass]: chose IMKClient_Modern` cluttering the terminal output. These are OS Activity Mode messages coming from Apple's Input Method Kit (IMK). They can be suppressed by appending `2> /dev/null` to the command. This is not a perfect solution. Normally, the OS_ACTIVITY_MODE environment variable could be set to "disable" to suppress such messages, but it appears that Apple's Input Method Kit (IMK) framework does not consistently honor that variable.

### Command-Line Options

The command-line interface is organized into several categories. Only the portfolio TOML file is required; the rest are optional and grouped below for clarity. Running the command without arguments prints a short help, and running the command with the option `--help` prints extended help.

Each of the options (except for `--config`) can also be set in the config TOML, allowing you to reuse the same settings across multiple runs without cluttering your CLI commands. When both the CLI and the config file specify the same setting, the CLI takes precedence.

---

#### 🧾 Configuration Options

- `--config` (`-c`):  
  Path to a TOML file containing general runtime settings like output preferences.  
  If omitted, the program looks for a file named config.toml in the current directory by default.  
  *(This option has no config key; it specifies the config file itself.)*

---

#### 📤 Output Control

- `--output-csv` (`-co`) → `output_csv = true`:  
  If set, writes the headline metrics in CSV format.
  - The CSV is written to `<output_dir>/<portfolio>.csv` (with sibling
    `<portfolio>.drawdowns.csv` and `<portfolio>.assets.csv`). The filename stem
    is derived from the portfolio TOML file name.
  - `output_dir` defaults to `outputs/`, so by default the files land there.
    Only if `output_dir` is explicitly set to an empty string is the CSV printed
    to the terminal (`stdout`) instead.

- `--output-snapshot` (`-os`) → `output_snapshot = true`:  
  If set, saves a snapshot image (PNG) of the performance plot to
  `<output_dir>/` (default `outputs/`).

- `--output-dir <dir>` (`-od`) → `output_dir = "outputs"`:  
  Directory where output files (CSV and/or snapshot image) are written.
  Defaults to `outputs/`.

---

#### 🛠️ Execution Behavior

- `--debug` (`-d`) → `debug = true`:  
  Enables debug mode, which may trigger additional logging or relaxed error handling.
  * Every CSV that is loaded (benchmark, risk‑free, NAV, PPF rates, …) is echoed:  
    ```
    📂  Loading «path/to/file.csv»
        ↳ last record 2025‑04‑18 (2 days old, max allowed 2)
    ```
  * The per-source freshness provenance line (`last data …, fetched …`)
    is printed for each reference feed (benchmark, risk‑free).

- `--disable-plot-display` (`-dpd`) → `show_plot = false`:
  Disables on-screen display of plots. Use this when running from scripts or environments without a graphical display.

- `--allow-stale` → `allow_stale = true`:
  Reference data (benchmark + risk-free) is a correctness invariant: when a
  source cannot be certified current the run is **blocked by default**, since
  stale reference data corrupts alpha/beta/Sharpe/Sortino. `--allow-stale` is
  the single override — it proceeds and prints a warning naming the degraded
  metrics. No effect under `--as-of` / `--replay-from` (which neither fetch nor
  block). See [Data freshness](docs/DATA_REFRESH.md).

- `--quiet` (`-q`) → `quiet = true`:
  Run non-interactively (assume “yes” to any prompt). Useful for automation,
  headless runs, or testing where manual input isn’t possible.

- `--max-drawdown-threshold <float>` (`-dt`) → `max_drawdown_threshold = 5.0`:  
  Sets the percentage threshold for reporting drawdowns.

- `--lookback` (`-lb`) *YTD | 1M | 3M | 6M | 1Y | 3Y | 5Y | 10Y*
  Trims every series to the chosen trailing period before metrics are
  calculated, letting you compare the results directly with sites such as
  ValueResearchOnline (which publish 1 M, 3 M, … numbers). Example:

      ./pa -lb 6M -d portfolio.toml

- `--metrics-method <daily|monthly>` → `metrics_method = "daily"`:  
  Sampling frequency for return/risk calculations (Volatility, Sharpe, Sortino).
  Default `daily`.

- `--as-of YYYY-MM-DD`:  
  Evaluate the portfolio as of this date: every NAV/CIV series is trimmed to
  `<= as_of`, the freshness gate uses it as the reference date, and `--lookback`
  counts back from it. Makes a run deterministic regardless of when data was
  fetched, and opts out of auto-refresh/blocking. (No config key.)

- `--replay-from <dir>` / `--save-replay <dir>`:  
  `--replay-from` reads NAV/SCSS data from local fixtures in `<dir>`
  (`<dir>/navs/<fund>.csv`, `<dir>/scss_nsi.html`) instead of the network;
  combined with `--as-of` a run is fully offline and deterministic.
  `--save-replay` captures those fixtures during a live run. The two are mutually
  exclusive. (No config keys.)

---

#### Benchmark Indices (config-file only, not in command-line options)

  `benchmark_name = "NIFTY Total Returns Index"`:  
  Sets the name of the benchmark index in the outputs.

  `benchmark_returns_file = "data/reference/NIFTY Total Returns Historical Data.csv"`:  
  Sets the filename of the benchmark data CSV. (Note the key is
  `benchmark_returns_file`, not `benchmark_file`.) The bundled default is the
  NIFTY 50 TRI history the auto-updater maintains.

  `benchmark_date_format = "%m/%d/%Y"`:  
  Sets the date format of the dates in the benchmark CSV. The bundled
  `NIFTY Total Returns Historical Data.csv` uses `%m/%d/%Y` (e.g. `05/02/2025`),
  which is also the built-in default. Niftyindices.com pages typically render
  `%d %b %Y`; Investing.com exports for Indian indices typically use `%m/%d/%Y`
  (they changed the format in early 2025). Match this setting to whatever file you
  actually point `benchmark_returns_file` at.

## Metrics

### Metrics computed

For every fund, PortfolioAnalyzer computes a set of *standalone* risk/return
metrics from the fund's own NAV history: **Annualized Return (CAGR)**, the compound
yearly growth rate; **Volatility**, the annualized Standard Deviation of returns;
the **Sharpe ratio**, return earned per unit of total risk; the **Sortino ratio**,
return per unit of *downside* risk only; and the **maximum-drawdown profile**, the
largest peak-to-trough falls and their recoveries (surfaced as a recovered-drawdown
table). Two further metrics are *benchmark-relative* and need the fund's stated
benchmark as a free total-return (TR) series: **Beta**, the fund's systematic-risk
sensitivity to its benchmark (how much it moves when the market moves), and
**Alpha** (Jensen's), the annualized risk-adjusted excess return over what
Beta-scaled benchmark exposure would predict.

Of the funds modelled here, only **ICICI Bluechip** (benchmark NIFTY 100) has a
freely available benchmark TR series, so it is the only one for which Alpha and Beta
are reported. The rest have no free benchmark TR series — the hybrid/debt funds
(benchmarked to NIFTY 50 Hybrid Composite Debt 65:35 / 15:85 and NIFTY Corporate
Bond A-II) and the USD-benchmarked Franklin US Opportunities feeder (see
[docs/ARCHITECTURE.md → External-metric parity](docs/ARCHITECTURE.md) for why).
Without the benchmark you can still judge whether a fund paid you for the risk it
took *in absolute terms*, but you cannot separate manager skill (alpha) from mere
market exposure (beta), you have no measure of systematic vs idiosyncratic risk, and
you get no benchmark-relative attribution. This is purely a data-sourcing limit, not
a methodology one: supply a valid benchmark TR series and Alpha/Beta populate
automatically.

* A _peak_ is the highest value of an investment before the value began to decline. If the price goes from 80 to 100 to 70 to 95, 100 is the peak. There can be many peaks, but when calculating drawdowns, we typically focus on all-time highs — the highest value seen so far.
* A _maximum drawdown with full recovery_ is the largest drop that the investment experiences from a peak, followed by a full recovery back to or beyond the original peak. It is measured as a percentage drop from the highest previous value.
* A _trough date_ is the date of the lowest point after the peak.
* A _recovery date_ is the date when the portfolio value returns to or exceeds the previous peak.
* _Drawdown days_ are the number of days from the peak date to the trough date.
* _Recovery days_ are the number of days from the peak date to the recovery date.
```
           ▼ peak                   ▼ trough                          ▼ recovery
Portfolio: ╭─────── decline ────────╮───────── recovery climb ────────╮
           │                        │                                 │
           └────── drawdown_days ───┘                                 │
           └────────────────────────────── recovery_days ─────────────┘
```
### Summary of Key Indian Investment Screening Platforms and Their Metric-Calculation Methodologies

There are only two free Indian web platforms with a high degree of transparency regarding their
mutual-fund metric-calculation methodologies: Morningstar India and CRISIL. See
_[5 ratios to measure risk and return](https://www.morningstar.in/posts/28205/5-ratios-to-measure-returns-and-risk.aspx?utm_source=chatgpt.com)_ on [Morningstar India](morningstar.in) and _[CRISIL Mutual Fund Ranking](https://www.crisil.com/content/dam/crisil/mutual-fund-ranking/crisil-mutual-fund-ranking-march-2023.pdf?utm_source=chatgpt.com)_.

---

## Project Structure

```
.
├── pa                           # Bundled-venv launcher (the one entry point)
├── portfolioanalyzer/           # The package — all application code lives here
│   ├── main.py                  # CLI entry point + pipeline driver
│   ├── loaders/                 # Per-asset loader package (mutual_fund, ppf, scss,
│   │                            #   benchmark, risk_free, gold, data_update, vro)
│   ├── sgb_holdings.py          # Per-tranche SGB valuation engine (gold spot + coupons)
│   ├── sgb_tranches.py          # SGB tranche reference + lookup API
│   ├── data_loader.py           # Legacy aggregator; re-exports loaders for back-compat
│   ├── synthetic_civ.py         # Interest-rate series → daily-equivalent CIV
│   ├── civ_to_returns.py        # CIV → returns
│   ├── timeseries/              # Timeseries class package:
│   │   ├── returns.py           #   TimeseriesReturn (alpha/beta + thin metrics delegates + reporting helpers)
│   │   ├── civ.py               #   TimeseriesCIV (validated-NAV class)
│   │   ├── asset.py             #   AssetTimeseries dataclass (civ/ret/cumret views)
│   │   └── portfolio.py         #   PortfolioTimeseries (weighted aggregation, effective window)
│   ├── metrics.py               # Pure-function math: CAGR / Vol / Sharpe / Sortino / drawdowns
│   ├── portfolio_calculator.py  # Allocations + cumulative gains
│   ├── bond_calculators.py      # Variable-rate bond cumulative gain (used by the SCSS path)
│   ├── fund_lifecycle.py        # Inauguration + DEFUNCT status + assets-CSV writer
│   ├── drawdowns_csv.py         # Per-drawdown sibling CSV writer
│   ├── visualizer.py            # Matplotlib plotting + drawdown printout
│   ├── output_metadata.py       # Pure formatters for the metrics block / provenance / PNG tEXt
│   ├── data_update_cli.py       # Entry point for the data auto-update (portfolio-analyzer-update)
│   └── utils.py                 # info / dbg / to_cutoff_date
├── pyproject.toml               # Package + dev-tool config (ruff, pytest, etc.)
└── README.md                    # This file
```
---

## Running Tests

Test dependencies are declared as the `dev` extra of the package:
```bash
pip install -e ".[dev]"
```
The default suite excludes `network`-marked tests (live FRED + niftyindices
fetch + the full subprocess smoke; the golden-master tests are *not* network —
they replay committed fixtures). See [`docs/TESTING.md`](docs/TESTING.md) for
the marker matrix and golden-regeneration recipe.

Run from the **project root**, using the venv interpreter (no activation
needed):
```bash
./venv/bin/python -m pytest                # unit + golden + non-network integration
./venv/bin/python -m pytest -m network     # live FRED + niftyindices + e2e smoke
```

The niftyindices live test needs the optional `browser` extra
(`pip install -e ".[browser]" && playwright install chromium`); without it it
skips rather than fails.

---

## Contributing

Contributions, suggestions, and bug reports are welcome. Please open an issue or submit a pull request if you have ideas or improvements.

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## Acknowledgements

Thanks to the creators and maintainers of the website [mfapi.in](https://mfapi.in), whoever they are. They have created and made available a wonderful free API that makes analysis of Indian mutual funds relatively easy.

Thanks to the developers of the open-source libraries used in this project. Their work makes projects like this possible.
