import pandas as pd

import utils
from data_loader import (
    align_dynamic_risk_free_rates,
    align_portfolio_civs,
    extract_weights,
    fetch_and_standardize_risk_free_rates,
    fetch_portfolio_civs,
    get_benchmark_gain_daily,
    load_config_toml,
    load_portfolio_details,
    load_ppf_civ,
    load_timeseries_csv,
)
from portfolio_calculator import (
    calculate_gains_cumulative,
    calculate_portfolio_allocations,
)
from timeseries.portfolio import from_multiple_nav_series
from timeseries.returns import TimeseriesReturn  # TODO: FIXME: TimeseriesReturn is being obsoleted
from utils import (
    dbg,
    info,
    to_cutoff_date,
)
from visualizer import plot_cumulative_returns, print_major_drawdowns


def main(settings):
    import os
    from pathlib import Path

    portfolio_dict = load_portfolio_details(settings["portfolio_file"])
    portfolio_label = portfolio_dict["label"]
    print(
        f"\nPortfolio metrics for {portfolio_label} (direct, growth) using "
        f" {settings["metrics_method"]} metrics method\n"
    )
    if settings.get("skip_age_check"):
        print(
            "⚠️  --skip-age-check active: benchmark/risk-free staleness "
            "gate bypassed. Refresh the CSVs in data/ when convenient."
        )
    if settings["debug"]:
        info(f"Portfolio label: {portfolio_label}.")
        info("Merged settings:")
        for k, v in settings.items():
            info(f"  {k}: {v}")

    benchmark_returns_series = None
    if settings.get("use_benchmark"):
        dbg(f"\n📂 Loading benchmark timeseries from \"{settings['benchmark_file']}\"")
        benchmark_data = load_timeseries_csv(
            settings["benchmark_file"],
            settings["benchmark_date_format"],
            max_delay_days=None if settings["skip_age_check"] else 3,
        )
        benchmark_returns_series = get_benchmark_gain_daily(benchmark_data)

    aligned_portfolio_civs = pd.DataFrame()
    portfolio_start_date = None
    unaligned_portfolio_civs: dict = {}
    if "funds" in portfolio_dict:
        unaligned_portfolio_civs = fetch_portfolio_civs(portfolio_dict)
        aligned_portfolio_civs = align_portfolio_civs(unaligned_portfolio_civs)
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

        dbg(f"\nLatest launch date among all mutual funds: {latest_date.date()}")
        dbg(f'Fund with the latest launch date: "{latest_fund}"')

    ppf_series = scss_series = rec_bond_series = gold_series = None
    sgb_series_by_tranche: dict[str, pd.Series] = {}

    if "ppf" in portfolio_dict:
        aligned_portfolio_civs["PPF"] = load_ppf_civ()

    if "scss" in portfolio_dict:
        from bond_calculators import calculate_variable_bond_cumulative_gain
        from data_loader import load_scss_interest_rates

        scss_rates = load_scss_interest_rates()
        scss_series = calculate_variable_bond_cumulative_gain(scss_rates, scss_rates.index.min())

    if "rec_bond" in portfolio_dict:
        from loaders.rec_bond import load_rec_bond_series

        rec_bond_series = load_rec_bond_series(portfolio_dict["rec_bond"])

    if "sgb" in portfolio_dict:
        # Phase 2: each [[sgb]] entry is a distinct holding. Per-tranche
        # CIV is built from per-gram gold spot plus accrued coupons by
        # the sgb_holdings engine (Phase 1 work).
        from loaders.gold import load_gold_prices_per_gram
        from sgb_holdings import sgb_holding_civ

        gold_per_gram = load_gold_prices_per_gram()
        for entry in portfolio_dict["sgb"]:
            asset_name = f"SGB {entry['tranche_id']}"
            sgb_series_by_tranche[asset_name] = sgb_holding_civ(
                tranche_id=entry["tranche_id"],
                units_grams=entry["units_grams"],
                gold_prices=gold_per_gram,
            )

    if "gold" in portfolio_dict:
        from loaders.gold import load_gold_prices

        gold_series = load_gold_prices()

    # === ROBUST PORTFOLIO START DATE LOGIC ===
    asset_series_list = [
        aligned_portfolio_civs,
        ppf_series,
        scss_series,
        rec_bond_series,
        gold_series,
        *sgb_series_by_tranche.values(),
    ]

    asset_start_dates = []
    for series in asset_series_list:
        if series is not None and not series.empty:
            if not isinstance(series.index, pd.DatetimeIndex):
                series.index = pd.to_datetime(series.index, errors="coerce")
            min_date = series.dropna().index.min()
            if pd.notna(min_date):
                asset_start_dates.append(min_date)

    if not asset_start_dates:
        raise ValueError("No valid asset data found to set portfolio start date!")

    portfolio_start_date = max(asset_start_dates)

    # Trim series only if not empty
    if aligned_portfolio_civs is not None and not aligned_portfolio_civs.empty:
        aligned_portfolio_civs = aligned_portfolio_civs[
            aligned_portfolio_civs.index >= portfolio_start_date
        ]

    if ppf_series is not None:
        ppf_series = ppf_series[ppf_series.index >= portfolio_start_date]
    if scss_series is not None:
        scss_series = scss_series[scss_series.index >= portfolio_start_date]
    if rec_bond_series is not None:
        rec_bond_series = rec_bond_series[rec_bond_series.index >= portfolio_start_date]
    sgb_series_by_tranche = {
        name: s[s.index >= portfolio_start_date]
        for name, s in sgb_series_by_tranche.items()
    }
    if gold_series is not None:
        gold_series = gold_series[gold_series.index >= portfolio_start_date]
    # === END OF ROBUST LOGIC ===

    if "funds" in portfolio_dict:
        assert (
            aligned_portfolio_civs is not None and not aligned_portfolio_civs.empty
        ), "aligned_portfolio_civs is missing or empty"
    dbg("\nExamining the types in the portfolio's CIV series:")
    for name, var in [
        ("aligned_portfolio_civs", aligned_portfolio_civs),
        ("ppf_series", ppf_series),
        ("scss_series", scss_series),
        ("rec_bond_series", rec_bond_series),
        ("gold_series", gold_series),
    ]:
        dbg(f"{name}: {type(var)}")
    for name, var in sgb_series_by_tranche.items():
        dbg(f"{name}: {type(var)}")

    # Mutual funds as a dict of Series
    fund_series_dict = {
        fund_name: aligned_portfolio_civs[fund_name] for fund_name in aligned_portfolio_civs.columns
    }

    # Other assets as a dict of Series
    extra_assets = {
        k: s
        for k, s, col in [
            ("PPF", ppf_series, "value"),
            ("SCSS", scss_series, "var_rate_bond_value"),
            ("REC", rec_bond_series, "var_rate_bond_value"),
            ("Gold", gold_series, "price"),
        ]
        if s is not None
    }

    # Final nav_inputs: per-tranche SGB series merge in alongside the
    # other asset classes; each tranche is its own asset with its own
    # weight in the portfolio.
    nav_inputs = {**fund_series_dict, **extra_assets, **sgb_series_by_tranche}
    for v in nav_inputs.values():
        assert isinstance(v, pd.Series), f"Expected pd.Series, got {type(v)}"

    weights = extract_weights(portfolio_dict)
    portfolio_ts = from_multiple_nav_series(nav_inputs, weights)

    # Effective window banner. combined_civ_series already clips the
    # portfolio CIV at the earliest-dying asset's last NAV; surface
    # which asset set the cutoff so a portfolio carrying one defunct
    # fund doesn't silently report metrics that stop months in the past.
    window = portfolio_ts.effective_window()
    print(
        f"Effective window: {window['start'].strftime('%Y-%m-%d')} → "
        f"{window['end'].strftime('%Y-%m-%d')}. "
        f"End set by '{window['end_limited_by']}'; "
        f"start set by '{window['start_limited_by']}'."
    )
    if benchmark_returns_series is not None:
        benchmark_returns_series = benchmark_returns_series[
            benchmark_returns_series.index <= window["end"]
        ]

    # Build portfolio NAV (CIV) and returns. portfolio_civ_series is the
    # canonical daily CIV from combined_civ_series — Phase D's frequency
    # fix guarantees it's reindexed onto the common business-day calendar.
    portfolio_civ_series = portfolio_ts.combined_civ_series()

    # Daily-return objects for CAPM Alpha/Beta come from pct_change() of
    # the daily CIV. Historically there was a parallel
    # ``combined_daily_returns`` aggregator that summed asset returns
    # weighted; it inner-joined to the monthly intersection when any
    # monthly asset was present and then ``alpha_capm`` ^252-annualized
    # what were actually monthly means. Symptom: Alpha = 139% on
    # port-everything. The aggregator is deleted; only the CIV-derived
    # path remains. See tests/unit/test_plot_metric_consistency.py for
    # the math fact (weighted-sum-of-returns ≠ return-of-weighted-sum).
    gain_daily_portfolio_series = portfolio_civ_series.series.pct_change().dropna()
    portfolio_daily_ret = TimeseriesReturn(gain_daily_portfolio_series.rename("value"))

    risk_free_rate_series = fetch_and_standardize_risk_free_rates(
        settings["risk_free_rates_file"],
        date_format=settings["riskfree_date_format"],
        max_allowed_delay_days=settings["max_riskfree_delay"],
    )

    if settings.get("lookback"):
        cutoff = to_cutoff_date(settings["lookback"])
        if settings["debug"]:
            print(f"📅 Look‑back window {settings['lookback']} → cutting data at {cutoff.date()}")
        gain_daily_portfolio_series = gain_daily_portfolio_series[
            gain_daily_portfolio_series.index >= cutoff
        ]
        benchmark_returns_series = benchmark_returns_series[benchmark_returns_series.index >= cutoff]
        risk_free_rate_series = risk_free_rate_series[risk_free_rate_series.index >= cutoff]
        portfolio_civ_series.series = portfolio_civ_series.series[
            portfolio_civ_series.series.index >= cutoff
        ]
        portfolio_daily_ret._series = portfolio_daily_ret._series[
            portfolio_daily_ret._series.index >= cutoff
        ]

    benchmark_daily_ret = TimeseriesReturn(benchmark_returns_series.rename("value"))

    aligned_risk_free_rate_series = align_dynamic_risk_free_rates(
        gain_daily_portfolio_series, risk_free_rate_series
    )
    risk_free_rate_annual = aligned_risk_free_rate_series.mean()
    risk_free_rate_daily = (1 + risk_free_rate_annual) ** (1 / 252) - 1
    dbg(f"\nrisk_free_rate_daily: {risk_free_rate_daily}")

    # Two data pipeline paths: NAVs for CAGR/Drawdowns, returns for Sharpe/Alpha/Beta
    portfolio_returns = TimeseriesReturn(portfolio_civ_series.series)

    if settings["metrics_method"] == "daily":
        frequency = "daily"
        periods_per_year = 252
    else:  # "monthly"
        frequency = "monthly"
        periods_per_year = 12
    risk_free_rate_adjusted = (1 + risk_free_rate_annual) ** (1 / periods_per_year) - 1

    dbg(f"\nfrequency: {frequency}, periods_per_year: {periods_per_year}\n")
    metrics = {
        "Annualized Return": portfolio_returns.cagr(),
        "Volatility": portfolio_returns.volatility(frequency=frequency),
        "Sharpe Ratio": portfolio_returns.sharpe(
            risk_free_rate=risk_free_rate_adjusted,
            frequency=frequency,
            periods_per_year=periods_per_year,
        ),
        "Sortino Ratio": portfolio_returns.sortino(
            risk_free_rate=risk_free_rate_adjusted,
            frequency=frequency,
            periods_per_year=periods_per_year,
        ),
    }

    if benchmark_returns_series is not None:
        metrics["Alpha"] = portfolio_daily_ret.alpha_capm(
            benchmark_daily_ret, risk_free_rate=risk_free_rate_daily
        )
        metrics["Beta"] = portfolio_daily_ret.beta_capm(
            benchmark_daily_ret, risk_free_rate=risk_free_rate_daily
        )
    else:
        metrics["Alpha"] = None
        metrics["Beta"] = None
    cagr = metrics["Annualized Return"] * 100
    vol = metrics["Volatility"] * 100
    alpha = metrics["Alpha"] * 100 if metrics["Alpha"] is not None else None
    beta = metrics["Beta"]
    max_drawdowns = portfolio_civ_series.max_drawdowns(threshold=0.05)
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
        csv_line = (
            f'"{portfolio_label}",'  # Escape commas in the portfolio label.
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

    # Per-asset metadata: inauguration date + defunct-status check.
    # Reuses the already-fetched per-fund NAV DataFrames; no extra network.
    from fund_lifecycle import build_assets_meta
    assets_meta = build_assets_meta(portfolio_dict, unaligned_portfolio_civs)
    defunct = [r for r in assets_meta if r["status"] == "DEFUNCT"]
    for r in defunct:
        print(
            f"⚠️  {r['name']} appears DEFUNCT — last NAV "
            f"{r['last_nav']} ({r['days_since_last_nav']} days ago)."
        )

    # If output_csv is enabled, write to file or stdout
    if settings["output_csv"]:
        if settings.get("output_dir"):
            os.makedirs(settings["output_dir"], exist_ok=True)
            stem = Path(settings["portfolio_file"]).stem
            csv_path = os.path.join(settings["output_dir"], stem + ".csv")
            with open(csv_path, "w") as f:
                f.write(csv_line + "\n")
            print(f"📄 CSV written to {csv_path}")
            # Sibling per-drawdown CSV: one row per recovered drawdown,
            # plus the final unrecovered drawdown if any.
            from drawdowns_csv import write_drawdowns_csv
            dd_path = os.path.join(settings["output_dir"], stem + ".drawdowns.csv")
            write_drawdowns_csv(max_drawdowns, dd_path)
            print(f"📄 Drawdown table written to {dd_path}")
            # Sibling per-asset CSV: one row per asset with inauguration /
            # last-NAV / status (LIVE / DEFUNCT / N/A).
            from fund_lifecycle import write_assets_csv
            assets_path = os.path.join(settings["output_dir"], stem + ".assets.csv")
            write_assets_csv(assets_meta, assets_path)
            print(f"📄 Assets table written to {assets_path}")
        else:
            print(csv_line)

    # Always print human-readable summary unless suppressed (optional setting later)
    print(f"Mean Risk-Free Rate: {risk_free_rate_annual * 100:.4f}%")
    print(f"Annualized Return (CAGR): {cagr:.2f}%")
    print(f"Volatility: {vol:.2f}%")
    print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.4f}")
    print(f"Sortino Ratio: {metrics['Sortino Ratio']:.4f}")
    print(f"Alpha: {alpha:.4f}%" if alpha is not None else "Alpha: N/A")
    print(f"Beta:  {beta:.4f}" if beta is not None else "Beta:  N/A")
    print(f"Drawdowns: {len(max_drawdowns)}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Max Drawdown Start: {max_dd_start}")
    print(f"Drawdown Days: {drawdown_days}")
    print(f"Recovery Days: {recovery_days}")
    print_major_drawdowns(max_drawdowns)

    # Plot the same series the headline metrics are derived from. Using
    # cumprod(1 + combined_daily_returns) would be a parallel aggregation
    # path that summed asset returns instead of asset values; for portfolios
    # with mixed-frequency assets (e.g. daily MFs + monthly gold) it diverges
    # from the CAGR by an order of magnitude. The CIV series already starts
    # at 1.0 by construction (combined_civ_series normalizes each asset).
    cumulative_historical = portfolio_civ_series.series
    _, cumulative_benchmark = calculate_gains_cumulative(
        gain_daily_portfolio_series, benchmark_returns_series
    )

    # For more automated operation, the plotting can be skipped.
    if settings["output_snapshot"]:
        import os

        os.makedirs(settings["output_dir"], exist_ok=True)
        # Strip "port-" prefix and ".toml" suffix
        base_name = os.path.splitext(os.path.basename(settings["portfolio_file"]))[0]
        image_name = base_name + ".png"
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
            assets_meta=assets_meta,
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
            assets_meta=assets_meta,
        )

    if settings["output_csv"] or settings["output_snapshot"]:
        os.makedirs(settings["output_dir"], exist_ok=True)


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description="Portfolio Analyzer application.")
    parser.set_defaults(show_plot=None)
    parser.add_argument("toml_file", type=str, help="Path to the TOML file describing the portfolio.")
    parser.add_argument(
        "--config", "-c", type=str, default="config.toml", help="Optional config file"
    )
    parser.add_argument(
        "--disable-plot-display",
        "-dpd",
        action="store_false",
        dest="show_plot",
        help=(
            "Disables on-screen display of the performance plot "
            "(useful in automation or headless mode)"
        ),
    )
    parser.add_argument(
        "--output-snapshot",
        "-os",
        action="store_true",
        help="Saves a snapshot image of the performance plot.",
    )
    parser.add_argument(
        "--output-csv",
        "-co",
        action="store_true",
        help="Output metrics in machine-readable CSV format.",
    )
    parser.add_argument(
        "--output-dir",
        "-od",
        help="If specified, save plot image there instead of in the default `outputs/` directory.",
    )
    parser.add_argument(
        "--max-drawdown-threshold",
        "-dt",
        type=float,
        default=5,
        help="Drawdown threshold, in percent.",
    )
    parser.add_argument(
        "--metrics-method",
        choices=["daily", "monthly"],
        default="daily",
        help="Choose frequency for return/risk calculations: daily or monthly",
    )
    parser.add_argument(
        "--max-riskfree-delay",
        "-mrd",
        type=int,
        help="Maximum allowed delay (in days) for the most recent risk-free rate entry.",
    )
    parser.add_argument(
        "--lookback",
        "-lb",
        choices=["YTD", "1M", "3M", "6M", "1Y", "3Y", "5Y", "10Y"],
        help=(
            "Trim all series to the chosen trailing period "
            "(e.g. 3M = last 3 months) before calculating metrics."
        ),
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help=(
            "Suppresses the 'Continue anyway?' prompt when stale data is detected, and "
            "automatically proceeds as if you answered yes."
        ),
    )
    parser.add_argument(
        "--skip-age-check",
        action="store_true",
        help=(
            "Bypass the stale-data blocker on the benchmark CSV and "
            "risk-free-rate CSV. Use when the auto-update path is "
            "unavailable; the run will still warn so silent staleness "
            "doesn't go unnoticed."
        ),
    )
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Show full tracebacks for debugging."
    )

    args = parser.parse_args()
    # overwrite the module‑level debug flag so utils.dbg() can see it
    utils.DEBUG = args.debug
    return args


def cli():
    """Console entry point. Wired into pyproject [project.scripts].

    Installs as `portfolio-analyzer` on PATH inside the venv after
    `pip install -e .`. Also invoked by `python main.py ...` for back-compat.
    """
    import sys
    import traceback

    args = parse_arguments()

    # If config file is missing, fallback to empty config dict (intended behavior)
    config = load_config_toml(args.config)

    settings = None
    try:
        settings = {
            "portfolio_file": args.toml_file,
            "show_plot": (
                args.show_plot if args.show_plot is not None else config.get("show_plot", True)
            ),
            "output_snapshot": args.output_snapshot or config.get("output_snapshot", False),
            "output_csv": args.output_csv or config.get("output_csv", False),
            "output_dir": args.output_dir or config.get("output_dir", "outputs"),
            "drawdown_threshold": args.max_drawdown_threshold
            or config.get("max_drawdown_threshold", 5.0),
            "metrics_method": args.metrics_method or config.get("metrics_method", "daily"),
            "skip_age_check": args.skip_age_check or config.get("skip_age_check", False),
            "quiet": args.quiet or config.get("quiet", False),
            "debug": args.debug or config.get("debug", False),
            "lookback": args.lookback or config.get("lookback"),  # None → full history
            "risk_free_rates_file": config.get(
                "risk_free_rates_file", "data/India 10-Year Bond Yield Historical Data.csv"
            ),
            "use_benchmark": config.get("use_benchmark", True),
            "benchmark_name": config.get("benchmark_name", "NIFTY Total Returns Index"),
            "benchmark_file": config.get(
                "benchmark_returns_file", "data/NIFTY Total Returns Historical Data.csv"
            ),
            "benchmark_date_format": config.get("benchmark_date_format", "%m/%d/%Y"),
            "riskfree_date_format": config.get("riskfree_date_format", "%m/%d/%Y"),
            "max_riskfree_delay": args.max_riskfree_delay or config.get("max_riskfree_delay", 61),
        }
        # --skip-age-check loosens the risk-free CSV gate too, otherwise the
        # user is one step closer but still blocked by a separate check.
        # An explicit --max-riskfree-delay wins to keep that knob useful.
        if settings["skip_age_check"] and not args.max_riskfree_delay:
            settings["max_riskfree_delay"] = 99999
        main(settings)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if settings and settings.get("debug"):
            print(traceback.format_exc(), file=sys.stderr)
        else:
            print("Run again with --debug for more details.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
