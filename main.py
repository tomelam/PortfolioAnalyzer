from data_loader import (
    get_aligned_portfolio_civs,
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
    print_major_drawdowns,
)
from portfolio_calculator import (
    calculate_gains_cumulative,
    calculate_portfolio_allocations,
    calculate_gain_daily_portfolio_series,
)
from visualizer import plot_cumulative_returns
from utils import info


def main():
    import pandas as pd
    args = parse_arguments()
    toml_file_path = args.toml_file
    benchmark_file = args.benchmark_returns_file
    benchmark_name = args.benchmark_name
    drawdown_threshold = args.max_drawdown_threshold / 100

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
    info(f"\nCalculating portfolio metrics for {portfolio_label}.")

    aligned_portfolio_civs = pd.DataFrame()
    portfolio_start_date = None

    if "funds" in portfolio:
        unaligned_portfolio_civs = fetch_portfolio_civs(portfolio)
        aligned_portfolio_civs = align_portfolio_civs(unaligned_portfolio_civs)
        multiindex_aligned_civs = aligned_portfolio_civs.copy()  # Save in case it's needed for the "golden" data
        if isinstance(aligned_portfolio_civs.columns, pd.MultiIndex):
            aligned_portfolio_civs.columns = aligned_portfolio_civs.columns.droplevel(1)
        if not aligned_portfolio_civs.empty:
            portfolio_start_date = aligned_portfolio_civs.index.min()
        fund_start_dates = {
            fund_name: df.index.min()
            for fund_name, df in fetch_portfolio_civs(portfolio).items()
            if not df.empty
        }

        latest_fund, latest_date = max(fund_start_dates.items(), key=lambda x: x[1])

        info(f"Latest launch date among all mutual funds: {latest_date.date()}")
        info(f"Fund with the latest launch date: {latest_fund}")

    ppf_series = scss_series = rec_bond_series = sgb_series = gold_series = None

    if "ppf" in portfolio:
        from ppf_calculator import calculate_ppf_cumulative_gain
        ppf_rates = load_ppf_interest_rates()
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

    cagr = metrics["Annualized Return"] * 100
    vol = metrics["Volatility"] * 100
    alpha = metrics["Alpha"] * 100
    if max_drawdowns:
        max_dd_info = min(max_drawdowns, key=lambda dd: dd["drawdown"])
        drawdown_days = max_dd_info["drawdown_days"]
        recovery_days = max_dd_info["recovery_days"]
        max_dd = max_dd_info["drawdown"]
        max_dd_start = max_dd_info["start_date"].strftime("%Y-%m-%d")
    else:
        drawdown_days = 0
        recovery_days = 0
        max_dd = 0.0
        max_dd_start = "N/A"

    if args.csv_output:
        print(
            f"\"{portfolio_label}\","  # Escape commas in the portfolio label.
            f"{cagr:.2f}%,"
            f"{vol:.2f}%,"
            f"{metrics['Sharpe Ratio']:.4f},"
            f"{metrics['Sortino Ratio']:.4f},"
            f"{alpha:.2f}%,"
            f"{metrics['Beta']:.4f},"
            f"{len(max_drawdowns)},"
            f"{max_dd:.2f}%,"
            f"{max_dd_start},"
            f"{drawdown_days},"
            f"{recovery_days}"
        )
    else:
        print(f"Mean Risk-Free Rate: {risk_free_rate * 100:.4f}%")
        print(f"Annualized Return (CAGR): {cagr:.2f}%")
        print(f"Volatility: {vol:.2f}%")
        print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.4f}")
        print(f"Sortino Ratio: {metrics['Sortino Ratio']:.4f}")
        print(f"Alpha: {alpha:.2f}%")
        print(f"Beta: {metrics['Beta']:.4f}")
        print(f"Drawdowns: {len(max_drawdowns)}")
        print(f"Max Drawdown: {max_dd:.2f}%")
        print(f"Max Drawdown Start: {max_dd_start}")
        print(f"Drawdown Days: {drawdown_days}")
        print(f"Recovery Days: {recovery_days}")
        print_major_drawdowns(max_drawdowns)
    
    cumulative_historical, cumulative_benchmark = calculate_gains_cumulative(
        gain_daily_portfolio_series, benchmark_returns
    )

    # Optionally, if the flag is set, dump portfolio data to a pickle file.
    if args.save_golden_data:
        import pickle
        portfolio_data = {
            "gain_daily": gain_daily_portfolio_series,
            "allocations": calculate_portfolio_allocations(portfolio),
            # You can extend this dictionary with other intermediate outputs
            # such as cumulative returns if desired.
        }
        dump_pickle("tests/data/aligned_civs.pkl", multiindex_aligned_civs)
        dump_pickle("tests/data/aligned_portfolio_civs.pkl", aligned_portfolio_civs)
        dump_pickle("tests/data/benchmark_data.pkl", benchmark_data)
        dump_pickle("tests/data/benchmark_returns.pkl", benchmark_returns)
        dump_pickle("tests/data/portfolio_civs.pkl", unaligned_portfolio_civs)
        info("Golden portfolio data generated.")

    # For more automated operation, the plotting can be skipped.
    if not args.do_not_plot:
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


def dump_pickle(filepath, obj):
    """Dump data to a Pickle file."""
    import pickle

    with open(filepath, "wb") as f:
        return pickle.dump(obj, f)


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Portfolio Analyzer application.")
    parser.add_argument("toml_file", type=str,
                        help="Path to the TOML file describing the portfolio."
    )
    parser.add_argument("--benchmark-name", "-bn", type=str, default="NIFTY TRI",
                        help="Benchmark name."
    )
    parser.add_argument("--benchmark-returns-file", "-br", type=str, default="data/NIFTRI.csv",
                        help="Risk-free rates file."
    )
    parser.add_argument("--risk-free-rates-file", "-rf", type=str, default="data/INDIRLTLT01STM.csv",
                        help="Risk-free rates file."
    )
    parser.add_argument("--max-drawdown-threshold", "-dt", type=float, default=5,
                        help="Drawdown threshold, in percent."
    )
    parser.add_argument("--do-not-plot", "-np", action="store_true",
                        help="Do not make the plot; only calculate the metrics."
    )
    parser.add_argument("--save-golden-data", "-sgd", action="store_true",
                        help="Save golden data as a Pickle file for testing."
    )
    parser.add_argument("--csv-output", "-co", action="store_true",
                        help="Print metrics in machine-readable CSV format."
    )
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
