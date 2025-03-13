from data_loader import (
    load_portfolio_details,
    get_aligned_portfolio_civs,
    fetch_and_standardize_risk_free_rates,
    align_dynamic_risk_free_rates,
    fetch_yahoo_finance_data,
    get_benchmark_gain_daily,
    extract_fund_allocations,
    load_ppf_interest_rates,
)
from metrics_calculator import (
    calculate_portfolio_metrics,
)
from portfolio_calculator import (
    calculate_gains_cumulative,
    calculate_portfolio_allocations,
    calculate_gain_daily_portfolio_series,
)
from visualizer import plot_cumulative_returns

def main():
    import pandas as pd
    args = parse_arguments()
    toml_file_path = args.toml_file
    benchmark_ticker = args.benchmark_ticker
    benchmark_name = args.benchmark_name
    drawdown_threshold = args.max_drawdown_threshold / 100

    benchmark_file = "data/NIFTRI.csv"
    try:
        benchmark_data = pd.read_csv(benchmark_file, index_col=0)
        # Convert the CSV’s date index (format: "dd-mm-yyyy") to datetime
        benchmark_data.index = pd.to_datetime(benchmark_data.index, format="%d-%m-%Y", errors="coerce")
        # Sort the data in chronological order
        benchmark_data.sort_index(inplace=True)
        # Rename "Price" to "Close" if needed
        if "Close" not in benchmark_data.columns and "Price" in benchmark_data.columns:
            benchmark_data.rename(columns={"Price": "Close"}, inplace=True)
        # Clean the "Close" column: remove commas and convert to float, if it's not already numeric.
        if benchmark_data["Close"].dtype == object:
            benchmark_data["Close"] = benchmark_data["Close"].str.replace(',', '').astype(float)
    except Exception as e:
        raise RuntimeError(f"Failed to load benchmark data from {benchmark_file}: {e}")
    benchmark_returns = get_benchmark_gain_daily(benchmark_data)

    portfolio = load_portfolio_details(toml_file_path)
    portfolio_label = portfolio["label"]
    print(f"\nCalculating portfolio metrics for {portfolio_label}.")

    # Initialize aligned portfolio data and portfolio_start_date
    aligned_portfolio_civs = pd.DataFrame()
    portfolio_start_date = None
    
    # Funds
    if "funds" in portfolio:
        aligned_portfolio_civs = get_aligned_portfolio_civs(portfolio)
        if not aligned_portfolio_civs.empty:
            portfolio_start_date = aligned_portfolio_civs.index.min()
            
    # PPF
    ppf_series = None
    if "ppf" in portfolio:
        from ppf_calculator import calculate_ppf_cumulative_gain
        print("Loading PPF interest rates...")
        ppf_rates = load_ppf_interest_rates(portfolio["ppf"]["ppf_interest_rates_file"])
        ppf_series = calculate_ppf_cumulative_gain(ppf_rates)
        portfolio_start_date = (
            ppf_series.index.min() if portfolio_start_date is None else max(portfolio_start_date, ppf_series.index.min())
        )

    # SCSS (Senior Citizen Savings Scheme)
    scss_series = None
    if "scss" in portfolio:
        from data_loader import load_scss_interest_rates
        from bond_calculators import calculate_variable_bond_cumulative_gain

        print("Loading SCSS interest rates...")
        scss_rates = load_scss_interest_rates()
        start_date_for_scss = portfolio_start_date if portfolio_start_date is not None else scss_rates.index.min()
        scss_series = calculate_variable_bond_cumulative_gain(scss_rates, start_date_for_scss)
        portfolio_start_date = (
            scss_series.index.min() if portfolio_start_date is None else max(portfolio_start_date, scss_series.index.min())
        )
        portfolio_start_date = (
            scss_series.index.min() if portfolio_start_date is None else max(portfolio_start_date, scss_series.index.min())
        )

    # REC Bonds
    rec_bond_series = None
    if "rec_bond" in portfolio:
        from bond_calculators import calculate_variable_bond_cumulative_gain

        # TODO: This probably belongs in a different module.
        print("Loading REC bond rates...")
        rec_bond_rates = pd.DataFrame(
            {'rate': [5.25]},
            index=pd.date_range("2000-01-01", pd.Timestamp.today(), freq='D')
        )
        start_date_for_rec_bond = portfolio_start_date if portfolio_start_date is not None else rec_bond_rates.index.min()
        rec_bond_series = calculate_variable_bond_cumulative_gain(rec_bond_rates, start_date_for_rec_bond)
        print("rec_bond_series head after creation:", rec_bond_series.head())
        print("rec_bond_series tail after creation:", rec_bond_series.tail())
        print("rec_bond_series empty?", rec_bond_series.empty)
        portfolio_start_date = (
            rec_bond_series.index.min()
            if portfolio_start_date is None
            else max(portfolio_start_date, rec_bond_series.index.min())
        )
        portfolio_start_date = (
            rec_bond_series.index.min()
            if portfolio_start_date is None
            else max(portfolio_start_date, rec_bond_series.index.min())
        )

    # SGB (Sovereign Gold Bonds)
    sgb_series = None
    if "sgb" in portfolio:
        from sgb_extractor import extract_sgb_series

        print("Extracting SGB tranche data...")
        sgb_series = extract_sgb_series()
        portfolio_start_date = (
            sgb_series.index.min() if portfolio_start_date is None else max(portfolio_start_date, sgb_series.index.min())
        )

    # Physical Gold
    gold_series = None
    if "gold" in portfolio:
        from gold_loader import load_gold_prices

        print("Loading physical gold price data...")
        gold_series = load_gold_prices()
        portfolio_start_date = (
            gold_series.index.min() if portfolio_start_date is None else max(portfolio_start_date, gold_series.index.min())
        )

    # Final check if still None
    if portfolio_start_date is None:
        raise RuntimeError("Portfolio data is empty; cannot determine start date.")

    gain_daily_portfolio_series = calculate_gain_daily_portfolio_series(
        portfolio,
        aligned_portfolio_civs,
        ppf_series,
        scss_series,
        rec_bond_series,
        gold_series,
    )

    risk_free_rate_series = fetch_and_standardize_risk_free_rates(args.risk_free_rates_file)
    risk_free_rates = align_dynamic_risk_free_rates(gain_daily_portfolio_series, risk_free_rate_series)
    risk_free_rate = risk_free_rates.mean()
    # Convert the annual risk-free rate to a daily rate:
    risk_free_rate_daily = (1 + risk_free_rate)**(1/252) - 1

    metrics, max_drawdowns = calculate_portfolio_metrics(
        gain_daily_portfolio_series,
        portfolio,
        risk_free_rate_daily,
        benchmark_returns
    )

    print(f"Mean Risk-Free Rate: {risk_free_rate * 100:.4f}%")
    print(f"Annualized Return: {metrics['Annualized Return'] * 100:.4f}%")
    print(f"Volatility: {metrics['Volatility'] * 100:.4f}%")
    print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.4f}")
    print(f"Sortino Ratio: {metrics['Sortino Ratio']:.4f}")
    if "Alpha" in metrics and "Beta" in metrics:
        print(f"Beta: {metrics['Beta']:.4f}")
        print(f"Alpha: {metrics['Alpha']:.4f}")

    cumulative_historical, cumulative_benchmark = calculate_gains_cumulative(
        gain_daily_portfolio_series, benchmark_returns
    )

    # Sanity checks:
    assert not gain_daily_portfolio_series.empty, "gain_daily_portfolio_series Series is empty."
    assert not cumulative_historical.empty, "cumulative_historical DataFrame is empty."
    # benchmark_data.empty will be empty if the Yahoo Finance API is down
    #assert not benchmark_data.empty, "benchmark_data is empty."
    #assert not benchmark_returns.empty, "benchmark_returns is empty."
    #assert not cumulative_benchmark.empty, "cumulative_benchmark is empty."

    plot_cumulative_returns(
        portfolio_label,
        cumulative_historical,
        "Historical Performance",
        cumulative_benchmark,
        benchmark_name,
        calculate_portfolio_allocations(portfolio),
        metrics,
        max_drawdowns,
        portfolio_start_date
    )

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Portfolio Analyzer application.")
    parser.add_argument("toml_file", type=str, help="Path to the TOML file describing the portfolio.")
    parser.add_argument("--benchmark-name", "-bn", type=str, default="NIFTY TRI", help="Benchmark name.")
    parser.add_argument("--benchmark-ticker", "-bt", type=str, default="^NSEI", help="Benchmark ticker.")
    parser.add_argument("--risk-free-rates-file", "-rf", type=str, default="FRED--INDIRLTLT01STM.csv", help="Risk-free rates file.")
    parser.add_argument("--max-drawdown-threshold", "-dt", type=float, default=5, help="Drawdown threshold, in percent.")
    return parser.parse_args()

if __name__ == "__main__":
    import sys
    import traceback
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        print(traceback.format_exc())
        sys.exit(1)
