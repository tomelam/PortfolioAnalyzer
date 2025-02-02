# Portfolio Analyzer

This repository contains the Portfolio Analyzer application. It fetches historical NAV data for Indian mutual funds, computes key portfolio performance metrics (such as annualized return, volatility, Sharpe/Sortino ratios, Alpha, Beta, and maximum drawdowns), and visualizes historical returns along with benchmark data.

---

## Features

- **Data Loading & Alignment:**  
  - Fetches NAV data for each fund via API calls.
  - Loads risk-free rate data from CSV.
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
   cd PortfolioAnalyzer```
2. **Create and activate a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt```
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
python main.py portfolio.toml --benchmark-name "NIFTY 50" --benchmark-ticker "^NSEI" --risk-free-rates-file "FRED--INDIRLTLT01STM.csv" --max-drawdown-threshold 5```

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

This project is licensed under the MIT License.

---

## Acknowledgements

Thanks to the developers of the open-source libraries used in this project. Their work makes projects like this possible.
