from data_loader import (
    load_portfolio_details,
    fetch_portfolio_civs,
    align_portfolio_civs,
    fetch_and_standardize_risk_free_rates,
    align_dynamic_risk_free_rates,
    get_benchmark_gain_daily,
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
    benchmark_name = args.benchmark_name
    drawdown_threshold = args.max_drawdown_threshold / 100

    benchmark_file = "data/NIFTRI.csv"
    benchmark_data = pd.read_csv(benchmark_file, index_col=0)
    benchmark_data.index = pd.to_datetime(benchmark_data.index, format="%d-%m-%Y", errors="coerce")
    benchmark_data.sort_index(inplace=True)
    if "Close" not in benchmark_data.columns and "Price" in benchmark_data.columns:
        benchmark_data.rename(columns={"Price": "Close"}, inplace=True)
    if benchmark_data["Close"].dtype == object:
        benchmark_data["Close"] = benchmark_data["Close"].str.replace(',', '').astype(float)
    benchmark_returns = get_benchmark_gain_daily(benchmark_data)

    portfolio = load_portfolio_details(toml_file_path)
    portfolio_label = portfolio["label"]
    print(f"\nCalculating portfolio metrics for {portfolio_label}.")

    aligned_portfolio_civs = pd.DataFrame()
    portfolio_start_date = None

    if "funds" in portfolio:
        unaligned_portfolio_civs = fetch_portfolio_civs(portfolio)
        aligned_portfolio_civs = align_portfolio_civs(unaligned_portfolio_civs)
        if isinstance(aligned_portfolio_civs.columns, pd.MultiIndex):
            aligned_portfolio_civs.columns = aligned_portfolio_civs.columns.droplevel(1)
        if not aligned_portfolio_civs.empty:
            portfolio_start_date = aligned_portfolio_civs.index.min()

    ppf_series = scss_series = rec_bond_series = sgb_series = gold_series = None

    if "ppf" in portfolio:
        from ppf_calculator import calculate_ppf_cumulative_gain
        ppf_rates = load_ppf_interest_rates(portfolio["ppf"]["ppf_interest_rates_file"])
        ppf_series = calculate_ppf_cumulative_gain(ppf_rates)

    if "scss" in portfolio:
        from data_loader import load_scss_interest_rates
        from bond_calculators import calculate_variable_bond_cumulative_gain
        scss_rates = load_scss_interest_rates()
        scss_series = calculate_variable_bond_cumulative_gain(scss_rates, scss_rates.index.min())

    if "rec_bond" in portfolio:
        from bond_calculators import calculate_variable_bond_cumulative_gain
        rec_bond_rates = pd.DataFrame({'rate': [5.25]}, index=pd.date_range("2000-01-01", pd.Timestamp.today(), freq='D'))
        rec_bond_series = calculate_variable_bond_cumulative_gain(rec_bond_rates, rec_bond_rates.index.min())

    if "sgb" in portfolio:
        from sgb_loader import create_sgb_daily_returns
        sgb_series = create_sgb_daily_returns("data/sgb_data.csv")

    if "gold" in portfolio:
        from gold_loader import load_gold_prices
        gold_series = load_gold_prices()

    # === ROBUST PORTFOLIO START DATE LOGIC ===
    asset_series_list = [
        aligned_portfolio_civs, ppf_series, scss_series,
        rec_bond_series, sgb_series, gold_series
    ]

    asset_start_dates = []
    for series in asset_series_list:
        if series is not None and not series.empty:
            if not isinstance(series.index, pd.DatetimeIndex):
                series.index = pd.to_datetime(series.index, errors='coerce')
            min_date = series.dropna().index.min()
            if pd.notna(min_date):
                asset_start_dates.append(min_date)

    if not asset_start_dates:
        raise ValueError("No valid asset data found to set portfolio start date!")

    portfolio_start_date = max(asset_start_dates)

    # Trim series only if not empty
    if aligned_portfolio_civs is not None and not aligned_portfolio_civs.empty:
        aligned_portfolio_civs = aligned_portfolio_civs[aligned_portfolio_civs.index >= portfolio_start_date]

    if ppf_series is not None:
        ppf_series = ppf_series[ppf_series.index >= portfolio_start_date]
    if scss_series is not None:
        scss_series = scss_series[scss_series.index >= portfolio_start_date]
    if rec_bond_series is not None:
        rec_bond_series = rec_bond_series[rec_bond_series.index >= portfolio_start_date]
    if sgb_series is not None:
        sgb_series = sgb_series[sgb_series.index >= portfolio_start_date]
    if gold_series is not None:
        gold_series = gold_series[gold_series.index >= portfolio_start_date]
    # === END OF ROBUST LOGIC ===

    gain_daily_portfolio_series = calculate_gain_daily_portfolio_series(
        portfolio,
        aligned_portfolio_civs,
        ppf_series,
        scss_series,
        rec_bond_series,
        sgb_series,
        gold_series,
    )

    risk_free_rate_series = fetch_and_standardize_risk_free_rates(args.risk_free_rates_file)
    risk_free_rates = align_dynamic_risk_free_rates(gain_daily_portfolio_series, risk_free_rate_series)
    risk_free_rate = risk_free_rates.mean()
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

    plot_cumulative_returns(
        portfolio_label,
        cumulative_historical,
        "Historical Performance",
        toml_file_path,
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
    parser.add_argument("--risk-free-rates-file", "-rf", type=str, default="FRED--INDIRLTLT01STM.csv", help="Risk-free rates file.")
    parser.add_argument("--max-drawdown-threshold", "-dt", type=float, default=5, help="Drawdown threshold, in percent.")

    parser.add_argument("--save-golden-data", "-sgd", action="store_true", help="Save golden data as a Pickle file for testing.")
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
