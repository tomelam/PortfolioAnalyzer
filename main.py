from dev_support import (
    configure_logging,
    set_aspect_logging,
    log_function_entry,
    log_function_details,
)
from data_loader import (
    load_portfolio_details,
    get_aligned_portfolio_civs,  # fetch_nav_data,    ???
    fetch_and_standardize_risk_free_rates,  # get_standardized_risk_free_rates,
    align_dynamic_risk_free_rates,  # get_dynamic_risk_free_rate,
    get_benchmark_navs,  # load_benchmark_data,
    extract_fund_allocations,
)
from metrics_calculator import (
    calculate_benchmark_cumulative,
    calculate_portfolio_metrics,
    calculate_gain_daily_portfolio_series,
    calculate_gains_cumulative,
    calculate_portfolio_allocations,
)
from stress_test import simulate_multiple_shocks
from visualizer import plot_cumulative_returns
import argparse


@log_function_details("fn_logger")
def main():
    junkx = 0

    configure_logging()

    args = parse_arguments()
    toml_file_path = args.toml_file
    benchmark_ticker = args.benchmark_ticker
    benchmark_name = args.benchmark_name
    no_growth_period = args.no_growth_period
    drawdown_threshold = args.max_drawdown_threshold / 100

    portfolio = load_portfolio_details(args.toml_file)
    portfolio_label = portfolio["label"]

    # TODO: Change "aligned_portfolio_civs" to something else.
    aligned_portfolio_civs = get_aligned_portfolio_civs(portfolio)
    gain_daily_portfolio_series = calculate_gain_daily_portfolio_series(
        portfolio, aligned_portfolio_civs
    )

    # TODO: Maybe change this to risk_free_rates.
    risk_free_rate_series = fetch_and_standardize_risk_free_rates(
        args.risk_free_rates_file
    )
    benchmark_returns = get_benchmark_navs(args.benchmark_ticker)

    # interpolate risk-free rate to match portfolio dates
    risk_free_rates = align_dynamic_risk_free_rates(
        gain_daily_portfolio_series, risk_free_rate_series
    )

    risk_free_rate = risk_free_rates.mean()

    # Calculate metrics
    earliest_date_of_portfolio = gain_daily_portfolio_series.index.min()
    benchmark_cumulative = calculate_benchmark_cumulative(
        benchmark_returns, earliest_date_of_portfolio
    )
    ### FIXME: benchmark_cumulative is wrong here, as evidenced by Beta being
    ### wrong.
    ### What's the difference between this benchmark_cumulative and
    ### cumulative_benchmark?
    metrics, max_drawdowns = calculate_portfolio_metrics(
        gain_daily_portfolio_series, risk_free_rate, benchmark_returns
    )
    annualized_return = metrics["Annualized Return"]

    # Log portfolio metrics
    print("\nPortfolio Metrics Before Stress Test:")
    print(f"Mean Risk-Free Rate: {risk_free_rate * 100:.4f}%")
    print(f"Annualized Return: {metrics['Annualized Return'] * 100:.4f}%")
    print(f"Volatility: {metrics['Volatility'] * 100:.4f}%")
    print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.4f}")
    print(f"Sortino Ratio: {metrics['Sortino Ratio']:.4f}")
    if "Alpha" in metrics and "Beta" in metrics:
        print(f"Beta: {metrics['Beta']:.4f}")
        print(f"Alpha: {metrics['Alpha']:.4f}")

    # Define shock scenarios
    shock_scenarios = [
        {
            "name": "Mild Shock: stocks -5%, bonds -2%",
            "equity_shock": -0.05,
            "debt_shock": -0.02,
            "shock_day": 30,
            "projection_days": 756,
            "plot_color": "orange",
        },
        {
            "name": "Severe Shock: stocks -20%, bonds -10%",
            "equity_shock": -0.20,
            "debt_shock": -0.10,
            "shock_day": 50,
            "projection_days": 756,
            "plot_color": "red",
        },
    ]

    latest_portfolio_value = gain_daily_portfolio_series.iloc[-1]
    # cumulative_historical = (1 + gain_daily_portfolio_series).cumprod() - 1
    cumulative_historical, cumulative_benchmark = calculate_gains_cumulative(
        gain_daily_portfolio_series, benchmark_returns
    )
    fund_allocations = extract_fund_allocations(portfolio)
    portfolio_allocations = calculate_portfolio_allocations(portfolio, fund_allocations)

    # Simulate shocks
    shock_results = simulate_multiple_shocks(
        portfolio_label,
        latest_portfolio_value,
        annualized_return,
        portfolio_allocations,
        shock_scenarios,
        args.no_growth_period,
    )

    # Generate visualizations
    # Plot results with benchmark
    for scenario_name, (plot_color, cumulative_projected) in shock_results.items():
        plot_cumulative_returns(
            portfolio_label,
            cumulative_historical,
            cumulative_projected,
            scenario_name,
            cumulative_benchmark,
            benchmark_name,
            portfolio_allocations,
            metrics,
            max_drawdowns,
        )

    # generate_plots(portfolio_data, metrics, shock_results, benchmark_returns)


@log_function_details("fn_loggern")
def parse_arguments():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Portfolio stress test with PPF-specific handling."
    )
    parser.add_argument(
        "toml_file", type=str, help="Path to the TOML file describing the portfolio."
    )
    parser.add_argument(
        "--benchmark-name",
        "-bn",
        type=str,
        default="NIFTY 50",
        help="Benchmark name (default: NIFTY 50).",
    )
    parser.add_argument(
        "--benchmark-ticker",
        "-bt",
        type=str,
        default="^NSEI",
        help="Benchmark ticker (default: ^NSEI).",
    )
    parser.add_argument(
        "--risk-free-rates-file",
        "-rf",
        type=str,
        default="FRED--INDIRLTLT01STM.csv",
        help="File containing historical risk-free rates, e.g GoI 10-Year bond yields",
    )
    parser.add_argument(
        "--max-drawdown-threshold",
        "-dt",
        type=float,
        default=5,
        help="Threshold for significant drawdowns, in percent (default: 5).",
    )
    parser.add_argument(
        "--no-growth-period",
        "-ng",
        type=int,
        default=90,
        help="Number of days after shocks during which projections do not grow (default: 90).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # configure_logging()
    main()
