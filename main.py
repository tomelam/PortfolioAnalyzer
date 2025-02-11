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
#from ppf_calculator import calculate_ppf_cumulative_gain
from fetch_gold_spot import get_gold_adjusted_spot
import argparse
import pandas as pd

def main():
    args = parse_arguments()
    toml_file_path = args.toml_file
    benchmark_ticker = args.benchmark_ticker
    benchmark_name = args.benchmark_name
    drawdown_threshold = args.max_drawdown_threshold / 100

    portfolio = load_portfolio_details(toml_file_path)
    portfolio_label = portfolio["label"]
    print(f"\nCalculating portfolio metrics for {portfolio_label}.")

    if not any(k in portfolio for k in ["funds", "gold", "ppf"]):
        raise ValueError("Portfolio must contain at least one asset (mutual funds, gold, or PPF).")

    aligned_portfolio_civs = get_aligned_portfolio_civs(portfolio) if "funds" in portfolio else pd.DataFrame()
    portfolio_start_date = aligned_portfolio_civs.index.min() if not aligned_portfolio_civs.empty else None

    if "gold" in portfolio:
        gold_series = get_gold_adjusted_spot(start_date=portfolio_start_date.strftime("%Y-%m-%d") if portfolio_start_date else "2000-01-01")
        if gold_series is not None:
            gold_series = gold_series.reindex(aligned_portfolio_civs.index, method="ffill") if not aligned_portfolio_civs.empty else gold_series
            aligned_portfolio_civs["gold"] = gold_series["Adjusted Spot Price"]
        else:
            raise ValueError("Gold spot price data could not be retrieved.")

    gain_daily_portfolio_series = calculate_gain_daily_portfolio_series(portfolio, aligned_portfolio_civs)
    risk_free_rate_series = fetch_and_standardize_risk_free_rates(args.risk_free_rates_file)
    benchmark_data = fetch_yahoo_finance_data(benchmark_ticker, refresh_hours=1, period="max")
    benchmark_returns = get_benchmark_gain_daily(benchmark_data)
    risk_free_rates = align_dynamic_risk_free_rates(gain_daily_portfolio_series, risk_free_rate_series)
    risk_free_rate = risk_free_rates.mean()

    metrics, max_drawdowns = calculate_portfolio_metrics(
        gain_daily_portfolio_series, risk_free_rate, benchmark_returns
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

    plot_cumulative_returns(
        portfolio_label,
        cumulative_historical,
        "Historical Performance",
        cumulative_benchmark,
        benchmark_name,
        calculate_portfolio_allocations(portfolio, extract_fund_allocations(portfolio)),
        metrics,
        max_drawdowns,
    )

def parse_arguments():
    parser = argparse.ArgumentParser(description="Portfolio Analyzer application.")
    parser.add_argument("toml_file", type=str, help="Path to the TOML file describing the portfolio.")
    parser.add_argument("--benchmark-name", "-bn", type=str, default="NIFTY 50", help="Benchmark name.")
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
