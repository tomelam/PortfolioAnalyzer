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
    calculate_benchmark_cumulative,
    calculate_portfolio_metrics,
    calculate_gain_daily_portfolio_series,
    calculate_gains_cumulative,
    calculate_portfolio_allocations,
)
from visualizer import plot_cumulative_returns
from ppf_calculator import calculate_ppf_cumulative_gain
import argparse
import pandas as pd

def main():
    args = parse_arguments()
    toml_file_path = args.toml_file
    benchmark_ticker = args.benchmark_ticker
    benchmark_name = args.benchmark_name
    drawdown_threshold = args.max_drawdown_threshold / 100

    # Load portfolio details from the TOML file.
    portfolio = load_portfolio_details(toml_file_path)
    portfolio_label = portfolio["label"]

    # Load and align mutual fund CIV data.
    aligned_portfolio_civs = get_aligned_portfolio_civs(portfolio)

    # If a PPF section exists, compute its CIV series and add it as an input.
    if "ppf" in portfolio:
        ppf_file = portfolio["ppf"].get("ppf_interest_rates_file", "ppf_interest_rates.csv")
        ppf_rates = load_ppf_interest_rates(ppf_file)
        # Use the mutual funds' common start date as the portfolio start.
        portfolio_start_date = aligned_portfolio_civs.index.min()
        ppf_series = calculate_ppf_cumulative_gain(ppf_rates, portfolio_start_date)
        # Align the PPF series to the mutual funds date range.
        ppf_series = ppf_series.reindex(aligned_portfolio_civs.index, method="ffill")
        # Add the PPF computed CIV as a new column.
        aligned_portfolio_civs["PPF"] = ppf_series["ppf_value"]

    # Calculate the portfolio daily returns (which now include the PPF input).
    gain_daily_portfolio_series = calculate_gain_daily_portfolio_series(portfolio, aligned_portfolio_civs)

    # Load risk-free rate data.
    risk_free_rate_series = fetch_and_standardize_risk_free_rates(args.risk_free_rates_file)

    # Load benchmark data from Yahoo Finance.
    benchmark_data = fetch_yahoo_finance_data(benchmark_ticker, refresh_hours=1, period="max")
    assert benchmark_data.index.name == "Date", "Index name mismatch: expected 'Date'"
    benchmark_returns = get_benchmark_gain_daily(benchmark_data)

    # Align risk-free rates with portfolio dates and compute the mean risk-free rate.
    risk_free_rates = align_dynamic_risk_free_rates(gain_daily_portfolio_series, risk_free_rate_series)
    risk_free_rate = risk_free_rates.mean()

    # Calculate portfolio metrics, including the PPF component.
    metrics, max_drawdowns = calculate_portfolio_metrics(
        gain_daily_portfolio_series, risk_free_rate, benchmark_returns
    )
    
    print("\nPortfolio Metrics:")
    print(f"Mean Risk-Free Rate: {risk_free_rate * 100:.4f}%")
    print(f"Annualized Return: {metrics['Annualized Return'] * 100:.4f}%")
    print(f"Volatility: {metrics['Volatility'] * 100:.4f}%")
    print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.4f}")
    print(f"Sortino Ratio: {metrics['Sortino Ratio']:.4f}")
    if "Alpha" in metrics and "Beta" in metrics:
        print(f"Beta: {metrics['Beta']:.4f}")
        print(f"Alpha: {metrics['Alpha']:.4f}")

    if max_drawdowns:
        print(f"\nNumber of maximum drawdowns with full retracements: {len(max_drawdowns)}")
        print("\nMaximum Drawdowns:")
        for drawdown in max_drawdowns:
            print(
                f"Start: {drawdown['start_date']}, Trough: {drawdown['trough_date']}, "
                f"End: {drawdown['end_date']}, Drawdown: {drawdown['drawdown']:.2f}%"
            )

    fund_allocations = extract_fund_allocations(portfolio)
    portfolio_allocations = calculate_portfolio_allocations(fund_allocations)

    # Calculate cumulative returns for the portfolio and benchmark.
    cumulative_historical, cumulative_benchmark = calculate_gains_cumulative(
        gain_daily_portfolio_series, benchmark_returns
    )

    # Plot historical performance (the PPF is now part of the portfolio).
    plot_cumulative_returns(
        portfolio_label,
        cumulative_historical,
        "Historical Performance",
        cumulative_benchmark,
        benchmark_name,
        portfolio_allocations,
        metrics,
        max_drawdowns,
    )

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Portfolio Analyzer application."
    )
    parser.add_argument("toml_file", type=str, help="Path to the TOML file describing the portfolio.")
    parser.add_argument("--benchmark-name", "-bn", type=str, default="NIFTY 50", help="Benchmark name (default: NIFTY 50).")
    parser.add_argument("--benchmark-ticker", "-bt", type=str, default="^NSEI", help="Benchmark ticker (default: ^NSEI).")
    parser.add_argument("--risk-free-rates-file", "-rf", type=str, default="FRED--INDIRLTLT01STM.csv", help="File containing historical risk-free rates.")
    parser.add_argument("--max-drawdown-threshold", "-dt", type=float, default=5, help="Threshold for significant drawdowns, in percent (default: 5).")
    return parser.parse_args()

if __name__ == "__main__":
    main()
