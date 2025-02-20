from data_loader import (
    load_portfolio_details,
    get_aligned_portfolio_civs,
    fetch_and_standardize_risk_free_rates,
    align_dynamic_risk_free_rates,
    load_benchmark_navs,
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
#from ppf_calculator import calculate_ppf_cumulative_gain
from fetch_gold_spot import get_gold_adjusted_spot

def main():
    import pandas as pd
    args = parse_arguments()
    toml_file_path = args.toml_file
    benchmark_csv_file = args.benchmark_csv_file
    benchmark_name = args.benchmark_name
    drawdown_threshold = args.max_drawdown_threshold / 100

    portfolio = load_portfolio_details(toml_file_path)
    portfolio_label = portfolio["label"]
    print(f"\nCalculating portfolio metrics for {portfolio_label}.")

    aligned_portfolio_civs = get_aligned_portfolio_civs(portfolio) if "funds" in portfolio else pd.DataFrame()
    portfolio_start_date = aligned_portfolio_civs.index.min() if not aligned_portfolio_civs.empty else None

    gold_series = None
    if "gold" in portfolio or "sgb" in portfolio:
        gold_series = get_gold_adjusted_spot(start_date="2000-01-01")

    # PPF (Public Provident Fund)
    ppf_series = None
    if "ppf" in portfolio:
        from ppf_calculator import calculate_ppf_cumulative_gain

        print("Loading PPF interest rates...")
        ppf_rates = load_ppf_interest_rates(portfolio["ppf"]["ppf_interest_rates_file"])
        ppf_series = calculate_ppf_cumulative_gain(ppf_rates)

    # Funds (Mutual Funds, Equity, Debt)
    funds_data = None
    if "funds" in portfolio:
        print("Processing mutual funds allocations...")
        funds_data = extract_fund_allocations(portfolio)

    # SCSS (Senior Citizen Savings Scheme)
    scss_series = None
    if "scss" in portfolio:
        from data_loader import load_scss_interest_rates
        from bond_calculators import calculate_variable_bond_cumulative_gain

        # Determine a valid start date:
        if not aligned_portfolio_civs.empty:
            start_date = aligned_portfolio_civs.index.min()
        else:
            start_date = pd.Timestamp("2010-01-01")  # or another appropriate default
    
        print("Loading SCSS interest rates...")
        scss_rates = load_scss_interest_rates()
        #scss_series = calculate_variable_bond_cumulative_gain(scss_rates, aligned_portfolio_civs.index.min())
        scss_series = calculate_variable_bond_cumulative_gain(scss_rates, start_date)

    # SGB (Sovereign Gold Bonds)
    sgb_data = None
    if "sgb" in portfolio:
        from sgb_extractor import extract_sgb_series

        print("Extracting SGB tranche data...")
        sgb_data = extract_sgb_series()

    # REC Bonds (Rural Electrification Corporation Bonds)
    rec_bond_series = None
    if "rec_bond" in portfolio:
        from bond_calculators import calculate_variable_bond_cumulative_gain

        print("Loading REC bond rates...")
        rec_bond_rates = portfolio["rec_bond"]["rates"]
        rec_bond_series = calculate_variable_bond_cumulative_gain(rec_bond_rates, aligned_portfolio_civs.index.min())

    gain_daily_portfolio_series = calculate_gain_daily_portfolio_series(
        portfolio,
        aligned_portfolio_civs,
        gold_series,
    )

    risk_free_rate_series = fetch_and_standardize_risk_free_rates(args.risk_free_rates_file)
    print(f"Loading benchmark {benchmark_name} data from {benchmark_csv_file}")
    benchmark_data = load_benchmark_navs(benchmark_csv_file)
    benchmark_returns = get_benchmark_gain_daily(benchmark_data)
    risk_free_rates = align_dynamic_risk_free_rates(gain_daily_portfolio_series, risk_free_rate_series)
    risk_free_rate = risk_free_rates.mean()

    metrics, max_drawdowns = calculate_portfolio_metrics(
        gain_daily_portfolio_series,
        portfolio,
        risk_free_rate,
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
    )

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Portfolio Analyzer application.")
    parser.add_argument("toml_file", type=str, help="Path to the TOML file describing the portfolio.")
    parser.add_argument("--benchmark-name", "-bn", type=str, default="NIFTY 50", help="Benchmark name.")
    parser.add_argument("--benchmark-csv-file", "-bf", type=str, default="data/Nifty 50 Historical Data.csv", help="Benchmark CSV file.")
    parser.add_argument("--risk-free-rates-file", "-rf", type=str, default="INDIRLTLT01STM.csv", help="Risk-free rates file.")
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
