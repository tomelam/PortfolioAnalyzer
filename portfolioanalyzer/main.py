import pandas as pd

from portfolioanalyzer import utils
from portfolioanalyzer.data_loader import (
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
from portfolioanalyzer.portfolio_calculator import (
    calculate_gains_cumulative,
    calculate_portfolio_allocations,
)
from portfolioanalyzer.timeseries.portfolio import from_multiple_nav_series
from portfolioanalyzer.timeseries.returns import TimeseriesReturn
from portfolioanalyzer.utils import (
    dbg,
    info,
    to_cutoff_date,
)
from portfolioanalyzer.visualizer import plot_cumulative_returns, print_major_drawdowns


def _reference_paths(settings, portfolio_dict):
    """The reference CSVs in use this run: benchmark (if enabled) + risk-free,
    plus the gold price series when the portfolio actually holds gold or SGBs
    (both are valued off it), plus the USD/INR FX series when it holds SGBs (an
    INR instrument whose rupee coupons are converted to USD). Only gold-bearing
    runs gate on the gold feed; SGB runs additionally gate on the FX feed."""
    paths = []
    if settings.get("use_benchmark"):
        paths.append(settings["benchmark_file"])
    paths.append(settings["risk_free_rates_file"])
    if "gold" in portfolio_dict or "sgb" in portfolio_dict:
        paths.append(settings["gold_prices_file"])
    if "sgb" in portfolio_dict:
        paths.append(settings["fx_usd_inr_file"])
    return paths


def _enforce_reference_freshness(settings, portfolio_dict):
    """Refresh stale reference feeds and enforce the block-by-default gate.

    Refreshes the benchmark / risk-free / gold sources that are behind their
    cadence (status messages go to stderr) and — for any source that could not
    be certified current — either blocks the run (naming the degraded metrics)
    or, under ``--allow-stale``, warns and proceeds. Provenance is reported
    separately by :func:`_report_reference_provenance` so it also rides every
    deterministic run and the PNG output.
    """
    from portfolioanalyzer.loaders.data_update import ensure_reference_data_fresh

    results = ensure_reference_data_fresh(_reference_paths(settings, portfolio_dict))
    for r in results:
        if r["message"]:
            info(r["message"])

    stale = [r for r in results if r["status"] == "stale"]
    if not stale:
        return
    names = "; ".join(r["name"] for r in stale)
    degraded = ", ".join(
        sorted({m.strip() for r in stale for m in r["affects"].split(",") if m.strip()})
    )
    if settings["allow_stale"]:
        info(
            f"⚠️  --allow-stale: proceeding with stale reference data ({names}). "
            f"Degraded metrics: {degraded}."
        )
    else:
        raise RuntimeError(
            f"Reference data is stale and could not be refreshed ({names}); this "
            f"degrades {degraded}. Re-run when the source is reachable, or pass "
            f"--allow-stale to proceed with degraded metrics."
        )


def _report_reference_provenance(settings, portfolio_dict):
    """Read the reference-feed provenance (all run modes), echo it to stderr for
    the run log, and return it for embedding in the PNG snapshot."""
    from portfolioanalyzer.loaders.data_update import reference_provenance

    provenance = reference_provenance(_reference_paths(settings, portfolio_dict))
    if provenance:
        from portfolioanalyzer import output_metadata as om

        info("📊 Reference-data provenance:")
        for line in om.format_provenance(provenance).splitlines():
            info(f"   {line}")
    return provenance


def main(settings):
    import os
    from pathlib import Path

    portfolio_dict = load_portfolio_details(settings["portfolio_file"])
    portfolio_label = portfolio_dict["label"]
    print(
        f"\nPortfolio metrics for {portfolio_label} (direct, growth) using "
        f" {settings["metrics_method"]} metrics method\n"
    )
    if settings["debug"]:
        info(f"Portfolio label: {portfolio_label}.")
        info("Merged settings:")
        for k, v in settings.items():
            info(f"  {k}: {v}")

    # --- data-freshness invariant ------------------------------------------
    # Stale reference data (benchmark, risk-free, and — for gold/SGB-bearing
    # portfolios — the LBMA gold price) silently corrupts metrics, so freshness
    # is enforced rather than advised: a source behind its
    # publication cadence is refreshed before computing, and one that cannot
    # be certified current BLOCKS the run — unless --allow-stale. Deterministic
    # modes (--as-of / --replay-from) opt out cleanly: data pinned through the
    # as-of date is current as of that date, so they neither fetch nor block.
    deterministic = settings.get("as_of") is not None or settings.get("replay_from")
    if not deterministic:
        _enforce_reference_freshness(settings, portfolio_dict)
    # Provenance is echoed to stderr in every mode and embedded in the PNG.
    reference_provenance = _report_reference_provenance(settings, portfolio_dict)

    benchmark_returns_series = None
    if settings.get("use_benchmark"):
        dbg(f"\n📂 Loading benchmark timeseries from \"{settings['benchmark_file']}\"")
        benchmark_data = load_timeseries_csv(
            settings["benchmark_file"],
            settings["benchmark_date_format"],
            # Freshness is enforced up front by _enforce_reference_freshness
            # (or deliberately skipped in deterministic modes), so the loader's
            # own staleness gate is disabled here to avoid double-gating.
            max_delay_days=None,
            as_of=settings.get("as_of"),
        )
        benchmark_returns_series = get_benchmark_gain_daily(benchmark_data)

    aligned_portfolio_civs = pd.DataFrame()
    portfolio_start_date = None
    unaligned_portfolio_civs: dict = {}
    if "funds" in portfolio_dict:
        unaligned_portfolio_civs = fetch_portfolio_civs(
            portfolio_dict,
            replay_from=settings.get("replay_from"),
            save_replay=settings.get("save_replay"),
        )
        aligned_portfolio_civs = align_portfolio_civs(unaligned_portfolio_civs)
        if isinstance(aligned_portfolio_civs.columns, pd.MultiIndex):
            aligned_portfolio_civs.columns = aligned_portfolio_civs.columns.droplevel(1)
        if not aligned_portfolio_civs.empty:
            portfolio_start_date = aligned_portfolio_civs.index.min()
        fund_start_dates = {
            fund_name: df.index.min()
            for fund_name, df in unaligned_portfolio_civs.items()
            if not df.empty
        }

        latest_fund, latest_date = max(fund_start_dates.items(), key=lambda x: x[1])

        dbg(f"\nLatest launch date among all mutual funds: {latest_date.date()}")
        dbg(f'Fund with the latest launch date: "{latest_fund}"')

    ppf_series = scss_series = gold_series = None
    sgb_series_by_tranche: dict[str, pd.Series] = {}

    if "ppf" in portfolio_dict:
        aligned_portfolio_civs["PPF"] = load_ppf_civ()

    if "scss" in portfolio_dict:
        from portfolioanalyzer.bond_calculators import calculate_variable_bond_cumulative_gain
        from portfolioanalyzer.data_loader import load_scss_interest_rates

        scss_cfg = portfolio_dict["scss"]
        scss_rates = load_scss_interest_rates(
            replay_from=settings.get("replay_from"),
            save_replay=settings.get("save_replay"),
        )
        # SCSS is valued term-locked (the rate locks at opening for the whole
        # term and re-prices only at each rollover), anchored at the holding's
        # purchase_date when given (else the analysis-window start). Reinvestment
        # is implicit so the sleeve characterises SCSS over the window rather
        # than decaying to idle cash. See docs/ARCHITECTURE.md.
        scss_series = calculate_variable_bond_cumulative_gain(
            scss_rates,
            scss_rates.index.min(),
            term_years=scss_cfg.get("term_years", 5),
            anchor_date=scss_cfg.get("purchase_date"),
        )

    if "sgb" in portfolio_dict:
        # Phase 2: each [[sgb]] entry is a distinct holding. Per-tranche
        # CIV is built from per-gram gold spot plus accrued coupons by
        # the sgb_holdings engine (Phase 1 work).
        from portfolioanalyzer.loaders.gold import load_gold_prices_per_gram
        from portfolioanalyzer.loaders.sgb_redemptions import lookup_redemption
        from portfolioanalyzer.sgb_holdings import sgb_asset_label, sgb_holding_civ

        gold_per_gram = load_gold_prices_per_gram(settings["gold_prices_file"])
        # USD/INR (FRED DEXINUS) converts each tranche's rupee coupons to USD at
        # their payment dates, keeping the CIV a consistent USD series. Freshness
        # is enforced up front by _enforce_reference_freshness, so the loader's
        # own staleness gate is disabled here to avoid double-gating.
        fx_usd_inr = load_timeseries_csv(
            settings["fx_usd_inr_file"],
            settings["fx_date_format"],
            max_delay_days=None,
        ).value_series()
        for entry in portfolio_dict["sgb"]:
            asset_name = sgb_asset_label(entry)
            # Early-redeemed holdings exit at RBI's announced premature-redemption
            # price (looked up from data/funds/sgb_redemptions.csv); HTM holdings
            # mark to gold spot through the window. See docs/ARCHITECTURE.md.
            redemption_date = redemption_inr = None
            if entry.get("valuation", "htm") == "redeemed":
                row = lookup_redemption(
                    entry["tranche_id"], entry.get("redemption_date")
                )
                redemption_date = pd.Timestamp(row["redemption_date"]).date()
                redemption_inr = row["inr_per_gram"]
            sgb_series_by_tranche[asset_name] = sgb_holding_civ(
                tranche_id=entry["tranche_id"],
                units_grams=entry["units_grams"],
                gold_prices=gold_per_gram,
                fx_inr_per_usd=fx_usd_inr,
                redemption_date=redemption_date,
                redemption_inr_per_gram=redemption_inr,
            )

    if "gold" in portfolio_dict:
        from portfolioanalyzer.loaders.gold import load_gold_prices

        gold_series = load_gold_prices(settings["gold_prices_file"])

    # --as-of YYYY-MM-DD: trim every loaded series to <= as_of so the
    # downstream math (CIV, returns, metrics, drawdowns) is bit-stable
    # regardless of when mfapi/benchmark/risk-free data was fetched.
    as_of = settings.get("as_of")
    if as_of is not None:
        if aligned_portfolio_civs is not None and not aligned_portfolio_civs.empty:
            aligned_portfolio_civs = aligned_portfolio_civs[
                aligned_portfolio_civs.index <= as_of
            ]
        if ppf_series is not None:
            ppf_series = ppf_series[ppf_series.index <= as_of]
        if scss_series is not None:
            scss_series = scss_series[scss_series.index <= as_of]
        if gold_series is not None:
            gold_series = gold_series[gold_series.index <= as_of]
        sgb_series_by_tranche = {
            name: s[s.index <= as_of]
            for name, s in sgb_series_by_tranche.items()
        }
        unaligned_portfolio_civs = {
            name: df[df.index <= as_of]
            for name, df in unaligned_portfolio_civs.items()
        }
        if benchmark_returns_series is not None:
            benchmark_returns_series = benchmark_returns_series[
                benchmark_returns_series.index <= as_of
            ]

    # === ROBUST PORTFOLIO START DATE LOGIC ===
    asset_series_list = [
        aligned_portfolio_civs,
        ppf_series,
        scss_series,
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
        # Freshness already enforced by _enforce_reference_freshness (or
        # skipped in deterministic modes); no second gate here.
        max_allowed_delay_days=None,
    )

    if settings.get("lookback"):
        cutoff = to_cutoff_date(settings["lookback"], as_of=settings.get("as_of"))
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
    # drawdown_threshold is carried in percent (CLI/config, default 5.0);
    # TimeseriesCIV.max_drawdowns wants a fraction, hence /100.
    max_drawdowns = portfolio_civ_series.max_drawdowns(
        threshold=settings["drawdown_threshold"] / 100
    )
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
    from portfolioanalyzer.fund_lifecycle import build_assets_meta
    as_of_for_meta = (
        settings["as_of"].date() if settings.get("as_of") is not None else None
    )
    assets_meta = build_assets_meta(
        portfolio_dict, unaligned_portfolio_civs, as_of=as_of_for_meta
    )
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
            from portfolioanalyzer.drawdowns_csv import write_drawdowns_csv
            dd_path = os.path.join(settings["output_dir"], stem + ".drawdowns.csv")
            write_drawdowns_csv(max_drawdowns, dd_path)
            print(f"📄 Drawdown table written to {dd_path}")
            # Sibling per-asset CSV: one row per asset with inauguration /
            # last-NAV / status (LIVE / DEFUNCT / N/A).
            from portfolioanalyzer.fund_lifecycle import write_assets_csv
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

    # Build the self-documenting payload that rides with the PNG: a
    # human-readable metrics block + reference-data provenance embedded in the
    # PNG tEXt metadata (machine-recoverable, far more usable than the
    # positional CSV row), plus a visible provenance footnote on the plot.
    from portfolioanalyzer import output_metadata as om

    if settings.get("as_of") is not None:
        run_label = f"as-of {settings['as_of'].date()}"
    elif settings.get("replay_from"):
        run_label = "replay"
    else:
        run_label = "live"
    metric_pairs = [
        ("Mean Risk-Free", f"{risk_free_rate_annual * 100:.4f}%"),
        ("CAGR", f"{cagr:.2f}%"),
        ("Volatility", f"{vol:.2f}%"),
        ("Sharpe", f"{metrics['Sharpe Ratio']:.4f}"),
        ("Sortino", f"{metrics['Sortino Ratio']:.4f}"),
        ("Alpha", f"{alpha:.2f}%" if alpha is not None else "N/A"),
        ("Beta", f"{beta:.4f}" if beta is not None else "N/A"),
        ("Drawdowns", str(len(max_drawdowns))),
        ("Max Drawdown", f"{max_dd:.2f}%"),
        ("Max DD Start", str(max_dd_start)),
        ("Drawdown Days", str(drawdown_days)),
        ("Recovery Days", str(recovery_days)),
    ]
    png_metadata = om.build_png_metadata(
        portfolio_label=portfolio_label,
        run_label=run_label,
        generated=pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        metric_pairs=metric_pairs,
        provenance=reference_provenance,
    )
    plot_footnote = (
        om.format_provenance_footnote(reference_provenance, run_label=run_label)
        if reference_provenance
        else None
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
            png_metadata=png_metadata,
            footnote=plot_footnote,
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
            footnote=plot_footnote,
        )

    if settings["output_csv"] or settings["output_snapshot"]:
        os.makedirs(settings["output_dir"], exist_ok=True)


def build_parser():
    """Construct the argument parser.

    Separated from :func:`parse_arguments` so tests can introspect the real
    option set (the single source of truth for documented CLI flags) without
    argparse consuming ``sys.argv``. See ``tests/unit/test_docs_consistency.py``.
    """
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
        # No argparse default: when the flag is absent this stays None so
        # build_settings can fall back to the config-file value (and only then
        # the 5.0 built-in). A baked-in default here would shadow config.
        default=None,
        help="Drawdown threshold, in percent (default 5.0; config may override).",
    )
    parser.add_argument(
        "--metrics-method",
        choices=["daily", "monthly"],
        # No argparse default (see --max-drawdown-threshold): absent → None so
        # build_settings falls back to config, then the "daily" built-in.
        default=None,
        help="Frequency for return/risk calculations (default daily; config may override).",
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
        help="Run non-interactively (assume 'yes' to any prompt).",
    )
    parser.add_argument(
        "--allow-stale",
        dest="allow_stale",
        action="store_true",
        help=(
            "Proceed even when reference data (benchmark / risk-free) cannot "
            "be certified current. By default such a run is BLOCKED, because "
            "stale reference data corrupts alpha/beta/Sharpe/Sortino. This is "
            "the single override; it prints a warning naming the degraded "
            "metrics. No effect under --as-of / --replay-from (which neither "
            "fetch nor block)."
        ),
    )
    parser.add_argument(
        "--as-of",
        dest="as_of",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Evaluate the portfolio as of this date. NAV/CIV series are "
            "trimmed to <= this date, the staleness gate uses it as the "
            "reference, and --lookback computes from it. Makes runs "
            "deterministic regardless of when the data was captured. "
            "Standard finance term for 'rewind the world to this date'."
        ),
    )
    replay_group = parser.add_mutually_exclusive_group()
    replay_group.add_argument(
        "--replay-from",
        dest="replay_from",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Read NAV/SCSS data from local fixtures in DIR instead of the "
            "network (DIR/navs/<fund>.csv and DIR/scss_nsi.html). Combined "
            "with --as-of, makes a run fully offline and deterministic."
        ),
    )
    replay_group.add_argument(
        "--save-replay",
        dest="save_replay",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "During a live run, write the fetched NAV/SCSS data into DIR so "
            "it can later be used with --replay-from. The capture mechanism "
            "for replay fixtures."
        ),
    )
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Show full tracebacks for debugging."
    )

    # Retired freshness knobs. The redesign replaced warn/skip/tune-the-age
    # toggles with a single block-by-default invariant + --allow-stale. These
    # are hard-removed (not aliased): a tombstone that fails fast with a
    # pointer beats a silent no-op or a stale generic "unrecognized argument".
    class _RemovedFlag(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            parser.error(
                f"{option_string} has been removed; freshness is now enforced "
                "automatically — use --allow-stale to proceed on stale data."
            )

    for _dead in ("--skip-age-check", "--no-auto-update", "--max-riskfree-delay", "-mrd"):
        parser.add_argument(
            _dead, action=_RemovedFlag, nargs="?", default=argparse.SUPPRESS,
            help=argparse.SUPPRESS,
        )

    return parser


def parse_arguments():
    parser = build_parser()
    args = parser.parse_args()
    # overwrite the module‑level debug flag so utils.dbg() can see it
    utils.DEBUG = args.debug
    return args


def build_settings(args, config: dict) -> dict:
    """Merge CLI args (``args``) over a config-file dict into the run settings.

    Precedence is CLI > config-file > built-in default. Pure (no I/O): given
    the same ``args`` namespace and ``config`` dict it always returns the same
    settings, which is what makes the precedence logic unit-testable apart from
    argparse and ``main()``.
    """
    return {
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
        # Block-by-default freshness invariant; --allow-stale is the single
        # override (config may set it too for non-interactive setups).
        "allow_stale": args.allow_stale or config.get("allow_stale", False),
        "quiet": args.quiet or config.get("quiet", False),
        "debug": args.debug or config.get("debug", False),
        "lookback": args.lookback or config.get("lookback"),  # None → full history
        # Default risk-free source is FRED INDIRLTLT01STM (India 10Y govt
        # bond rate) — the auto-updatable, no-auth feed (see
        # loaders.data_update). The legacy manual investing.com 10Y CSV
        # can still be selected via config if preferred.
        "risk_free_rates_file": config.get(
            "risk_free_rates_file", "data/reference/INDIRLTLT01STM.csv"
        ),
        "use_benchmark": config.get("use_benchmark", True),
        "benchmark_name": config.get("benchmark_name", "NIFTY Total Returns Index"),
        "benchmark_file": config.get(
            "benchmark_returns_file", "data/reference/NIFTY Total Returns Historical Data.csv"
        ),
        "benchmark_date_format": config.get("benchmark_date_format", "%m/%d/%Y"),
        # Gold price series (LBMA Gold Price PM fix, USD/oz, daily) — auto-
        # refreshed via loaders.data_update and block-gated like the other
        # reference feeds for gold/SGB-bearing portfolios. Config may repoint it
        # (the golden replay pins it to a frozen copy for determinism).
        "gold_prices_file": config.get(
            "gold_prices_file", "data/reference/gold_lbma_usd_daily.csv"
        ),
        # USD/INR FX series (FRED DEXINUS, INR per USD, daily) — auto-refreshed
        # via loaders.data_update and block-gated for SGB-bearing portfolios,
        # whose rupee coupons are converted to USD at each cash-flow date. Config
        # may repoint it (the golden replay pins it to a frozen copy).
        "fx_usd_inr_file": config.get(
            "fx_usd_inr_file", "data/reference/DEXINUS.csv"
        ),
        "fx_date_format": config.get("fx_date_format", "%Y-%m-%d"),
        "riskfree_date_format": config.get("riskfree_date_format", "%Y-%m-%d"),
        # --as-of YYYY-MM-DD pins the evaluation date for determinism;
        # parsed once here so downstream code sees a Timestamp, not a str.
        "as_of": (
            pd.Timestamp(args.as_of).normalize()
            if args.as_of is not None
            else None
        ),
        # Offline replay: read fixtures from / write fixtures to DIR.
        # Mutually exclusive at the argparse layer.
        "replay_from": args.replay_from,
        "save_replay": args.save_replay,
    }


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
        settings = build_settings(args, config)
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
