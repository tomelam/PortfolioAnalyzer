# Portfolio Analyzer

> **In a hurry?** See [`QUICKSTART.md`](QUICKSTART.md) for the most common
> invocations, the full CLI flag table, and `make all` / `scripts/render-all.sh`
> for batch sweeps.

This repository contains the Portfolio Analyzer application. It fetches historical NAV data for Indian mutual funds from [mfapi.in](https://mfapi.in), uses benchmark data from [investing.com](https://investing.com) or [niftyindices.com](https://in.investing.com/indices/nifty-total-returns-historical-data), uses risk-free rate data from [FRED](https://fred.stlouisfed.org), [RBI](https://rbi.org.in), or [DBIE (by advanced-searching "bill" while selecting "Weekly" Report Frequency and "Publication" Function)](https://data.rbi.org.in), and uses monthly-average gold spot prices from the link "Gold price averages in a range of currencies since 1978" on [gold.org](https://www.gold.org/goldhub/data/gold-prices), computes key portfolio performance metrics (such as annualized return, volatility, Sharpe/Sortino ratios, Alpha, Beta, and maximum drawdowns), and visualizes historical returns along with benchmark data.

---

## Features

- **Data Loading & Alignment:**  
  - Fetches NAV data for each fund via API calls.
  - Loads risk-free rate data from CSV. The included file INDIRLTLT01STM.csv is one such file manually downloaded from [fred.stlouisfed.org](https://fred.stlouisfed.org).
  - Uses PPF interest rates manually encoded as CSV in the file `data/ppf_interest_rates.csv`.
  - Scrapes the SCSS interest rates from [The National Savings Institute's table "Senior Citizens' Savings Scheme--Interest Rate Since Inception"](https://www.nsiindia.gov.in/(S(2xgxs555qwdlfb2p4ub03n3n))/InternalPage.aspx?Id_Pk=181).
  - Loads the SGB issue price/unit and redemption price/unit data manually copied from the Wikipedia page [Sovereign Gold Bond](https://en.wikipedia.org/wiki/Sovereign_Gold_Bond).
  - Uses a fixed rate (5.0%) for the REC Limited 5% bond (ISIN: INE020B07MD4).
  - Uses gold futures (GCJ5) manually downloaded as CSV from [https://www.investing.com/commodities/gold-historical-data](https://www.investing.com/commodities/gold-historical-data) and stored in the file `"data/Gold Futures Historical Data.csv"`. It is difficult to source the gold spot price for free, but gold futures front-month contracts closely approximate the gold spot price, especially as the contract nears expiration. This is why PortfolioAnalyzer uses the gold futures front-month contract price as a proxy for the gold spot price.
  - Uses benchmark historical data from [niftyindices.com](https://www.niftyindices.com) or [investing.com](https://investing.com) (deprecated).
  - Aligns data to a common date range across all data sources.

- **Portfolio Metrics Calculation:**  
  - Computes annualized return, volatility, Sharpe ratio, Sortino ratio.
  - Calculates Alpha and Beta relative to the benchmark.
  - Identifies significant drawdowns.

- **Visualization:**  
  - Plots historical cumulative returns for the portfolio.
  - Overlays benchmark data and highlights significant drawdown periods.

- **Modular Design:**
  - Loaders (one per asset class): `mutual_fund_loader.py`, `ppf_loader.py`,
    `scss_loader.py`, `rec_bond_loader.py`, `sgb_holdings.py` + `sgb_tranches.py`,
    `gold_loader.py`, `benchmark_loader.py`, `risk_free_loader.py`.
  - Math layer: `metrics.py` (pure-function CAGR / vol / Sharpe / Sortino /
    drawdowns) plus `timeseries.py` / `timeseries_civ.py` /
    `asset_timeseries.py` / `portfolio_timeseries.py` for the class surface.
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
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
&nbsp;
   _Dependencies include, among others:_
   * pandas
   * numpy
   * requests
   * toml
   * urllib3
   * beautifulsoup4
   * statsmodels
   * matplotlib

---

## Usage

If the portfolio described by the TOML file includes PPF as a component, ensure that the file detailing historical PPF interest rates, `ppf_interest_rates.csv`, is up to date before running the program.

If the portfolio described by the TOML file includes gold as a component, download the CSV file of the gold prices from https://www.investing.com/commodities/gold-historical-data before running the program. Currently, both offshore vaulted gold and gold held in India are priced using the same CSV data.
After `pip install -e .`, the console entry point `portfolio-analyzer` is on `PATH` inside the venv. `python main.py …` continues to work for back-compat.
```bash
portfolio-analyzer --help
portfolio-analyzer <path_to_portfolio_toml_file> [options]
portfolio-analyzer port/port-1.toml --max-drawdown-threshold 10 --skip-age-check
```
The `--max-drawdown-threshhold` option (shortcut `-dt`) sets the percentage drawdown that is considered significant to count in the "Drawdowns" statistic. By default, the threshhold is set to `5` (5%).

The benchmark name and benchmark/risk-free CSV paths live in the config TOML
(see `config/example_config.toml`); no CLI shortcuts exist for those. Run
`portfolio-analyzer --help` for the authoritative flag list.

⚠️ NOTE: Files downloaded from Investing.com sometimes use different date formats (e.g., %d-%m-%Y vs %m/%d/%Y). Always check the format of the first few rows and pass --benchmark-date-format accordingly.

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

- `--output-csv` → `output_csv = true`:  
  If set, outputs metrics in CSV format.
  - If `--save-output-to` is set (or `output_dir` is defined in the config), the CSV is written to a file in that directory. The filename is derived from the portfolio TOML file name.
  - Otherwise, the CSV is printed to the terminal (`stdout`).

  `--output-snapshot` → `output_snapshot = true`:  
  If set, saves a snapshot image of the performance plot.
  - If --save-output-to <dir> is given, the image is saved to that directory.
  - Otherwise, it is saved to the default `outputs/` directory.

- `--save-output-to <dir>` → `output_dir = "outputs"`:  
  Specifies the directory where any output file (CSV and/or snapshot image) will be saved.  
  - If not set and `--output-snapshot` is used, output goes to the `outputs/` directory by default.
  - If not set and `--output-csv` is used, the CSV is printed to the terminal (`stdout`).

---

#### 🛠️ Execution Behavior

- `--debug` (`-d`) → `debug = true`:  
  Enables debug mode, which may trigger additional logging or relaxed error handling.
  * Every CSV that is loaded (benchmark, risk‑free, NAV, PPF rates, …) is echoed:  
    ```
    📂  Loading «path/to/file.csv»
        ↳ last record 2025‑04‑18 (2 days old, max allowed 2)
    ```
  * Default freshness limits are **2 days for the benchmark** and the value of
    `max_riskfree_delay` (default 61) for the risk‑free series.  
  * If a series is staler than its limit, the `utils.warn_if_stale` prompt is
    triggered.

- `--disable-plot-display` (`-dpd`) → `show_plot = false`:
  Disables on-screen display of plots. Use this when running from scripts or environments without a graphical display.

- `--skip-age-check` → `skip_age_check = true`:
  Bypasses the hard blocker on a stale benchmark CSV and risk-free-rate CSV.
  When active, the run prints a one-line warning so the bypass is never silent.
  Default behavior is strict: a stale CSV exits non-zero with "data is outdated".
  
  `--quiet` (`-q`) → `quiet = true`:
  Suppresses the “Continue anyway?” prompt when stale data is detected, and automatically proceeds as if you answered yes.  
  Useful for automation, headless runs, or testing where manual input isn’t possible.

- `--max-drawdown-threshold <float>` (`-dt`) → `max_drawdown_threshold = 5.0`:  
  Sets the percentage threshold for reporting drawdowns.

- `--lookback` (`-lb`) *YTD | 1M | 3M | 6M | 1Y | 3Y | 5Y | 10Y*
  Trims every series to the chosen trailing period before metrics are
  calculated, letting you compare the results directly with sites such as
  ValueResearchOnline (which publish 1 M, 3 M, … numbers). Example:

      python main.py -lb 6M -d portfolio.toml

---

#### Benchmark Indices (config-file only, not in command-line options)

  `benchmark_name = "NIFTY Total Returns Index"`:  
  Sets the name of the benchmark index in the outputs.
  
  `benchmark_file = "data/NIFTY TOTAL MARKET_Historical_PR_01012007to28032025.csv"`:  
  Sets the filename of the benchmark data.
  
  `benchmark_date_format = "%d %b %Y"`:  
  Sets the date format for the dates in the benchmark data CSV file. "%d %b %Y" is the format that Niftyindices.com typically uses. "%m/%d/%Y" is the format that Investing.com typically uses for Indian indices, since they changed the format in the first quarter of 2025.

## Metrics

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
├── main.py                  # CLI entry point + pipeline driver
├── *_loader.py              # Per-asset loaders (mutual_fund, ppf, scss, rec_bond,
│                            #   benchmark, risk_free, gold)
├── sgb_holdings.py          # Per-tranche SGB valuation engine (gold spot + coupons)
├── sgb_tranches.py          # SGB tranche reference + lookup API
├── data_loader.py           # Legacy aggregator; re-exports loaders for back-compat
├── synthetic_civ.py         # Interest-rate series → daily-equivalent CIV
├── civ_to_returns.py        # CIV → returns
├── timeseries.py            # TimeseriesReturn (alpha/beta + thin metrics delegates)
├── timeseries_civ.py        # TimeseriesCIV (validated-NAV class)
├── asset_timeseries.py      # AssetTimeseries dataclass (civ/ret/cumret views)
├── portfolio_timeseries.py  # PortfolioTimeseries (weighted aggregation, effective window)
├── metrics.py               # Pure-function math: CAGR / Vol / Sharpe / Sortino / drawdowns
├── portfolio_calculator.py  # Allocations + cumulative gains
├── bond_calculators.py      # Variable-rate bond cumulative gain (used by rec_bond_loader)
├── fund_lifecycle.py        # Inauguration + DEFUNCT status + assets-CSV writer
├── drawdowns_csv.py         # Per-drawdown sibling CSV writer
├── visualizer.py            # Matplotlib plotting + drawdown printout
├── utils.py                 # info / dbg / warn_if_stale / to_cutoff_date
├── pyproject.toml           # Package + dev-tool config (ruff, pytest, etc.)
└── README.md                # This file
```
---

## Running Tests

Test dependencies are declared as the `dev` extra of the package:
```bash
pip install -e ".[dev]"
```
The default suite excludes `network`-marked tests (golden-master + the
full subprocess smoke). Run them explicitly with `pytest -m network` or
`pytest -m 'not network or network'`. See [`docs/TESTING.md`](docs/TESTING.md)
for the marker matrix and golden-regeneration recipe.

```bash
pytest                       # unit + non-network integration
pytest -m network            # adds golden-master + e2e smoke
```

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
