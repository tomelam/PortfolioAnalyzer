from logging import debug
import pandas as pd
import numpy as np
from portfolio_timeseries import from_multiple_nav_series
from timeseries import TimeseriesReturn  # TODO: FIXME: TimeseriesReturn is being obsoleted
from timeseries_civ import TimeseriesCIV
from timeseries_return import TimeseriesReturn
from civ_to_returns import civ_to_returns
from data_loader import (
    load_config_toml,
    load_timeseries_csv,
    load_index_data,
    get_aligned_portfolio_civs,
    load_portfolio_details,
    fetch_portfolio_civs,
    align_portfolio_civs,
    fetch_and_standardize_risk_free_rates,
    align_dynamic_risk_free_rates,
    get_benchmark_gain_daily,
    load_ppf_interest_rates,
)
from portfolio_calculator import (
    calculate_gains_cumulative,
    calculate_portfolio_allocations,
    calculate_gain_daily_portfolio_series,
)
from visualizer import plot_cumulative_returns, print_major_drawdowns
import utils
from utils import (
    info,
    dbg,
    warn_if_stale,
    to_cutoff_date,
)


def main(args):
    import os
    from pathlib import Path

    portfolio_dict = load_portfolio_details(settings["portfolio_file"])
    portfolio_label = portfolio_dict["label"]
    print(f"\nPortfolio metrics for {portfolio_label}.\n")
    if settings["debug"]:
        info(f"Portfolio label: {portfolio_label}.")
        info("Merged settings:")
        for k, v in settings.items():
            info(f"  {k}: {v}")

    benchmark_returns = None
    if settings.get("use_benchmark"):
        dbg(f"📂 Loading benchmark timeseries from \"{settings['benchmark_file']}\"")
        benchmark_data = load_timeseries_csv(
            settings["benchmark_file"],
            settings["benchmark_date_format"],
            max_delay_days=None if settings["skip_age_check"] else 2,
        )
        benchmark_returns = get_benchmark_gain_daily(benchmark_data)

    aligned_portfolio_civs = pd.DataFrame()
    portfolio_start_date = None
    if "funds" in portfolio_dict:
        unaligned_portfolio_civs = fetch_portfolio_civs(portfolio_dict)
        aligned_portfolio_civs = align_portfolio_civs(unaligned_portfolio_civs)
        multiindex_aligned_civs = aligned_portfolio_civs.copy()  # Save in case it's needed for the "golden" data
        if isinstance(aligned_portfolio_civs.columns, pd.MultiIndex):
            aligned_portfolio_civs.columns = aligned_portfolio_civs.columns.droplevel(1)
        if not aligned_portfolio_civs.empty:
            portfolio_start_date = aligned_portfolio_civs.index.min()
        fund_start_dates = {
            fund_name: df.index.min()
            for fund_name, df in fetch_portfolio_civs(portfolio_dict).items()
            if not df.empty
        }

        latest_fund, latest_date = max(fund_start_dates.items(), key=lambda x: x[1])

        dbg(f"Latest launch date among all mutual funds: {latest_date.date()}")
        dbg(f"Fund with the latest launch date: \"{latest_fund}\"")

    ppf_series = scss_series = rec_bond_series = sgb_series = gold_series = None

    if "ppf" in portfolio_dict:
        aligned_portfolio_civs["PPF"] = load_ppf_civ()

    if "scss" in portfolio_dict:
        from data_loader import load_scss_interest_rates
        from bond_calculators import calculate_variable_bond_cumulative_gain
        scss_rates = load_scss_interest_rates()
        scss_series = calculate_variable_bond_cumulative_gain(scss_rates, scss_rates.index.min())

    if "rec_bond" in portfolio_dict:
        from bond_calculators import calculate_variable_bond_cumulative_gain
        rec_bond_rates = pd.DataFrame({'rate': [5.25]}, index=pd.date_range("2000-01-01", pd.Timestamp.today(), freq='D'))
        rec_bond_series = calculate_variable_bond_cumulative_gain(rec_bond_rates, rec_bond_rates.index.min())

    if "sgb" in portfolio_dict:
        from sgb_loader import create_sgb_daily_returns
        sgb_series = create_sgb_daily_returns("data/sgb_data.csv")

    if "gold" in portfolio_dict:
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

    assert aligned_portfolio_civs is not None and not aligned_portfolio_civs.empty, "aligned_portfolio_civs is missing or empty"
    dbg("Examining the types in the portfolio's CIV series:")
    for name, var in [
        ("aligned_portfolio_civs", aligned_portfolio_civs),
        ("ppf_series", ppf_series),
        ("scss_series", scss_series),
        ("rec_bond_series", rec_bond_series),
        ("sgb_series", sgb_series),
        ("gold_series", gold_series),
    ]:
        dbg(f"{name}: {type(var)}")

    # Stage 3: Convert aligned DataFrame to dict of Series
    fund_series_dict = {
        fund_name: aligned_portfolio_civs[fund_name]
        for fund_name in aligned_portfolio_civs.columns
    }

    # Convert to Series format before passing to from_multiple_nav_series
    nav_inputs = {
        **fund_series_dict,
        "PPF": ppf_series["value"] if ppf_series is not None else None,
        "SCSS": scss_series["value"] if scss_series is not None else None,
        "REC": rec_bond_series["value"] if rec_bond_series is not None else None,
        "SGB": sgb_series["price"] if sgb_series is not None else None,
        "Gold": gold_series["price"] if gold_series is not None else None,
    }

    # Clean out None values before constructing portfolio
    nav_inputs = {k: v for k, v in nav_inputs.items() if v is not None}
    portfolio_ts = from_multiple_nav_series(nav_inputs,
                    weights={f['name']: f['allocation'] for f in portfolio_dict['funds']})

    ### Comment and call to `civ_and_returns()` removed from this point
    ### 10 lines moved here
    # Build portfolio NAV (CIV) and returns
    gain_daily_portfolio_series = portfolio_ts.combined_daily_returns()
    portfolio_civ_series = portfolio_ts.combined_civ_series()

    if settings.get("lookback"):
        cutoff = to_cutoff_date(settings["lookback"])
        if settings["debug"]:
            print(f"📅 Look‑back window {settings['lookback']} → cutting data at {cutoff.date()}")
        gain_daily_portfolio_series = gain_daily_portfolio_series[gain_daily_portfolio_series.index >= cutoff]
        benchmark_returns          = benchmark_returns[benchmark_returns.index >= cutoff]
        risk_free_rate_series      = risk_free_rate_series[risk_free_rate_series.index >= cutoff]

    risk_free_rate_series = fetch_and_standardize_risk_free_rates(
        settings["risk_free_rates_file"],
        date_format=settings["riskfree_date_format"],
        max_allowed_delay_days=settings["max_riskfree_delay"],
    )

    risk_free_rates = align_dynamic_risk_free_rates(gain_daily_portfolio_series, risk_free_rate_series)
    risk_free_rate = risk_free_rates.mean()
    risk_free_rate_daily = (1 + risk_free_rate)**(1/252) - 1
    dbg(f"risk_free_rate_daily: {risk_free_rate_daily}")

    # Two data pipeline paths: NAVs for CAGR/Drawdowns, returns for Sharpe/Alpha/Beta
    tsf_civ_obj = portfolio_civ_series
    tsf_returns_obj = TimeseriesReturn(tsf_civ_obj.to_returns(frequency="monthly"))
    portfolio_civ_obj     = portfolio_civ_series
    portfolio_returns_obj = TimeseriesReturn(portfolio_civ_obj.to_returns(frequency="monthly"))

    # Optional manual CAGR sanity calculator
    """
    start_value = 1.008298
    end_value = 10.492101
    start_date = pd.Timestamp("2013-01-02")
    end_date = pd.Timestamp("2025-04-24")
    days = (end_date - start_date).days
    years = days / 365.25
    cagr_manual = (end_value / start_value) ** (1 / years) - 1
    dbg(f"Sanity CAGR: {cagr_manual:.4%}")
    """

    # Set frequency and scaling based on metrics method
    METRICS_METHOD = "morningstar"
    if METRICS_METHOD == "morningstar":
        frequency = "monthly"
        periods_per_year = 12
    else:
        frequency = "daily"
        periods_per_year = 252

    metrics = {
        "Annualized Return": portfolio_returns_obj.cagr(),
        "Volatility": portfolio_returns_obj.volatility(frequency=frequency),
        "Sharpe Ratio": portfolio_returns_obj.sharpe(
            risk_free_rate=risk_free_rate_daily,
            frequency=frequency,
            periods_per_year=periods_per_year
        ),
        "Sortino Ratio": portfolio_returns_obj.sortino(
            risk_free_rate=risk_free_rate_daily,
            frequency=frequency,
            periods_per_year=periods_per_year
        ),
    }

    # Benchmark returns object
    benchmark_returns_obj = TimeseriesReturn(benchmark_returns)
    if benchmark_returns is not None:
        metrics["Alpha"] = portfolio_returns_obj.alpha_capm(
            benchmark_returns_obj,
            risk_free_rate=risk_free_rate_daily
        )
        metrics["Beta"]  = portfolio_returns_obj.beta_capm(
            benchmark_returns_obj,
            risk_free_rate=risk_free_rate_daily
        )
    else:
        metrics["Alpha"] = None
        metrics["Beta"] = None
    cagr = metrics["Annualized Return"] * 100
    vol = metrics["Volatility"] * 100
    if metrics["Alpha"] is not None:
        alpha = metrics["Alpha"] * 100
    else:
        alpha = None
    beta = metrics["Beta"]
    max_drawdowns = portfolio_civ_obj.max_drawdowns(threshold=0.05)
    if max_drawdowns:
        worst = max(max_drawdowns, key=lambda d: d["drawdown"])  # smallest (most negative) drawdown
        drawdown_days = worst["drawdown_days"]
        recovery_days = worst["recovery_days"]
        max_dd = worst["drawdown"] * 100
        max_dd_start = worst["start"].strftime("%Y-%m-%d")
    else:
        drawdown_days = 0
        recovery_days = 0
        max_dd = 0.0
        max_dd_start = "N/A"

    if settings["output_csv"]:
        if settings["output_csv"]:
            csv_line = (
                f"\"{portfolio_label}\","  # Escape commas in the portfolio label.
                f"{cagr:.2f}%,"
                f"{vol:.2f}%,"
                f"{metrics['Sharpe Ratio']:.4f},"
                f"{metrics['Sortino Ratio']:.4f},"
                f"{f'{alpha:.2f}%' if alpha is not None else 'N/A'},"
                f"{f'{beta:.4f}'   if beta  is not None else 'N/A'},"
                f"{len(max_drawdowns)},"
                f"{max_dd:.2f}%,"
                f"{max_dd_start},"
                f"{drawdown_days},"
                f"{recovery_days}"
            )

    # If output_csv is enabled, write to file or stdout
    if settings["output_csv"]:
        if settings.get("output_dir"):
            os.makedirs(settings["output_dir"], exist_ok=True)
            csv_path = os.path.join(
                settings["output_dir"],
                Path(settings["portfolio_file"]).stem + ".csv"
            )
            with open(csv_path, "w") as f:
                f.write(csv_line + "\n")
            print(f"📄 CSV written to {csv_path}")
        else:
            print(csv_line)

    # Always print human-readable summary unless suppressed (optional setting later)
    print(f"Mean Risk-Free Rate: {risk_free_rate * 100:.4f}%")
    print(f"Annualized Return (CAGR): {cagr:.2f}%")
    print(f"Volatility: {vol:.2f}%")
    print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.4f}")
    print(f"Sortino Ratio: {metrics['Sortino Ratio']:.4f}")
    print(f"Alpha: {alpha:.2f}%" if alpha is not None else "Alpha: N/A")
    print(f"Beta:  {beta:.4f}"   if beta  is not None else "Beta:  N/A")
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
    if settings["save_golden"]:
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
    if settings["output_snapshot"]:
        import os
        os.makedirs(settings["output_dir"], exist_ok=True)
        # Strip "port-" prefix and ".toml" suffix
        base_name = os.path.basename(settings["portfolio_file"])
        image_name = base_name.replace("port-", "").replace(".toml", "") + ".png"
        image_path = os.path.join(settings["output_dir"], image_name)
        plot_cumulative_returns(
            portfolio_label,
            cumulative_historical,
            "Historical Performance",
            settings["portfolio_file"],
            cumulative_benchmark,
            settings["benchmark_name"],
            calculate_portfolio_allocations(portfolio_ts),
            metrics,
            max_drawdowns,
            portfolio_start_date,
            save_path=image_path,
        )

    if settings.get("show_plot", True):
        plot_cumulative_returns(
            portfolio_label,
            cumulative_historical,
            "Historical Performance",
            settings["portfolio_file"],
            cumulative_benchmark,
            settings["benchmark_name"],
            calculate_portfolio_allocations(portfolio_ts),
            metrics,
            max_drawdowns,
            portfolio_start_date,
        )

    if settings["output_dir_explicit"] and not (settings["output_csv"] or settings["output_snapshot"]):
        info(f"⚠️  Warning: output_dir is set to '{settings['output_dir']}' but no output will be written to it.")


def dump_pickle(filepath, obj):
    """Dump data to a Pickle file."""
    import pickle

    with open(filepath, "wb") as f:
        return pickle.dump(obj, f)


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Portfolio Analyzer application.")
    parser.set_defaults(show_plot=None)
    parser.add_argument("toml_file", type=str,
                        help="Path to the TOML file describing the portfolio."
    )
    parser.add_argument("--config", "-c", type=str, default="config.toml",
                        help="Optional config file"
    )
    parser.add_argument("--disable-plot-display", "-dpd", action="store_false", dest="show_plot",
                        help="Disables on-screen display of the performance plot (useful in automation or headless mode)"
    )
    parser.add_argument("--output-snapshot", "-os", action="store_true",
                        help="Saves a snapshot image of the performance plot."
    )
    parser.add_argument("--output-csv", "-co", action="store_true",
                        help="Output metrics in machine-readable CSV format."
    )
    parser.add_argument("--output-dir", "-od",
                        help="If specified, save plot image there instead of in the default `outputs/` directory."
    )
    parser.add_argument("--max-drawdown-threshold", "-dt", type=float, default=5,
                        help="Drawdown threshold, in percent."
    )
    parser.add_argument("--max-riskfree-delay", "-mrd", type=int,
                        help="Maximum allowed delay (in days) for the most recent risk-free rate entry."
    )
    parser.add_argument("--lookback", "-lb",
                        choices=["YTD", "1M", "3M", "6M", "1Y", "3Y", "5Y", "10Y"],
                        help=("Trim all series to the chosen trailing period "
                              "(e.g. 3M = last 3 months) before calculating metrics."))
    parser.add_argument("--save-golden-data", "-sgd", action="store_true",
                        help="Save golden data as a Pickle file for testing."
    )
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppresses the 'Continue anyway?' prompt when stale data is detected, and automatically proceeds as if you answered yes.")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Show full tracebacks for debugging."
    )

    args = parser.parse_args()
    # overwrite the module‑level debug flag so utils.dbg() can see it
    utils.DEBUG = args.debug
    return args


if __name__ == "__main__":
    import sys
    import traceback

    args = parse_arguments()

    # If config file is missing, fallback to empty config dict (intended behavior)
    config = load_config_toml(args.config)

    try:
        settings = {
            "portfolio_file": args.toml_file,
            "show_plot": args.show_plot if args.show_plot is not None else config.get("show_plot", True),
            "output_snapshot": config.get("output_snapshot", False),
            "output_csv": args.output_csv or config.get("output_csv", False),
            "output_dir": args.output_dir or config.get("output_dir", "outputs"),
            "output_dir_explicit": bool(args.output_dir),
            "drawdown_threshold": args.max_drawdown_threshold or config.get("max_drawdown_threshold", 5.0),
            "skip_age_check": config.get("skip_age_check", False),
            "quiet": args.quiet or config.get("quiet", False),            
            "save_golden": args.save_golden_data or config.get("save_golden_data", False),
            "debug": args.debug or config.get("debug", False),
            "lookback": args.lookback or config.get("lookback"),  # None → full history
            "risk_free_rates_file": config.get("risk_free_rates_file", "data/India 10-Year Bond Yield Historical Data.csv"),
            "use_benchmark": config.get("use_benchmark", True),
            "benchmark_name": config.get("benchmark_name", "NIFTY Total Returns Index"),
            "benchmark_file": config.get("benchmark_returns_file", "data/NIFTY Total Returns Historical Data.csv"),
            "benchmark_date_format": config.get("benchmark_date_format", "%d-%m-%Y"),
            "riskfree_date_format": config.get("riskfree_date_format", "%m/%d/%Y"),
            "max_riskfree_delay": args.max_riskfree_delay or config.get("max_riskfree_delay", 61),
        }
        main(settings)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if settings["debug"]:
            print(traceback.format_exc(), file=sys.stderr)
        else:
            print("Run again with --debug for more details.", file=sys.stderr)
        sys.exit(1)
