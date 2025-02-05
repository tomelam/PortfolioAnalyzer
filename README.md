# Portfolio Analyzer

This repository contains the Portfolio Analyzer application. It fetches historical NAV data for Indian mutual funds from mfapi.in, computes key portfolio performance metrics (such as annualized return, volatility, Sharpe/Sortino ratios, Alpha, Beta, and maximum drawdowns), and visualizes historical returns along with benchmark data.

---

## Features

- **Data Loading & Alignment:**  
  - Fetches NAV data for each fund via API calls.
  - Loads risk-free rate data from CSV. The included file FRED--INDIRLTLT01STM.csv is one such file downloaded from [fred.stlouisfed.org](https://fred.stlouisfed.org).
  - Uses gold futures (GCJ5) downloaded manually as CSV from [https://www.investing.com/commodities/gold-historical-data](https://www.investing.com/commodities/gold-historical-data).
  - Retrieves benchmark historical data from Yahoo Finance.
  - Aligns data to a common date range across all data sources.

- **Portfolio Metrics Calculation:**  
  - Computes annualized return, volatility, Sharpe ratio, Sortino ratio.
  - Calculates Alpha and Beta relative to the benchmark.
  - Identifies significant drawdowns.

- **Visualization:**  
  - Plots historical cumulative returns for the portfolio.
  - Overlays benchmark data and highlights significant drawdown periods.

- **Modular Design:**  
  - Organized into separate modules: `data_loader.py`, `metrics_calculator.py`, and `visualizer.py`.

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
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
&nbsp;
   _Dependencies include:_
   * pandas
   * numpy
   * matplotlib
   * requests
   * toml
   * yfinance

---

## Usage

```bash
python main.py <path_to_portfolio_toml_file> [options]
python main.py portfolio.toml --benchmark-name "NIFTY 50" --benchmark-ticker "^NSEI" --risk-free-rates-file "FRED--INDIRLTLT01STM.csv" --max-drawdown-threshold 5
```
If the TOML file you use to define the analyzed and plotted portfolio includes gold as a component, download the CSV file of the gold prices from https://www.investing.com/commodities/gold-historical-data .

Mac users might notice messages like `2025-02-04 20:00:14.220 python[20791:454371] +[IMKClient subclass]: chose IMKClient_Modern` cluttering the terminal output. These are OS Activity Mode messages coming from Apple's Input Method Kit (IMK). They can be suppressed by appending `2> /dev/null` to the command. This is not a perfect solution. Normally, the OS_ACTIVITY_MODE environment variable could be set to "disable" to suppress such messages, but it appears that Apple's Input Method Kit (IMK) framework does not consistently honor that variable.

---

## Project Structure

```
.
├── main.py                  # Entry point for the application
├── data_loader.py           # Handles data fetching, standardization, and alignment
├── metrics_calculator.py    # Computes portfolio metrics and cumulative gains
├── visualizer.py            # Generates plots for historical portfolio and benchmark performance
├── requirements.txt         # List of required Python packages
└── README.md                # This file
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
