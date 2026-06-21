# Data Loader Module
# This module provides tools for loading and processing data related to
# financial portfolios, including functions to fetch NAV data, risk-free rates,
# and benchmark indices, align data to common ranges, and calculate fund
# allocations.
#
# Function Table
# 1. Top-Level Functions:
#    - align_portfolio_civs
#    - load_portfolio_details
#    - fetch_and_standardize_risk_free_rates
#    - align_dynamic_risk_free_rates
#    - fetch_navs_of_mutual_fund
#    - load_ppf_interest_rates

import os
import re

import pandas as pd
import toml


def load_config_toml(config_path: str) -> dict:
    """Load general runtime settings from a config TOML file."""
    if not os.path.exists(config_path):
        return {}
    return toml.load(config_path)


def load_portfolio_toml(portfolio_path: str) -> dict:
    """Load portfolio-specific structure and fund metadata."""
    if not os.path.exists(portfolio_path):
        raise FileNotFoundError(f"Portfolio file not found: {portfolio_path}")
    return toml.load(portfolio_path)


# Backward-compatibility re-export. Implementation lives in benchmark_loader.
from portfolioanalyzer.loaders.benchmark import load_timeseries_csv  # noqa: E402, F401


def _fund_slug(name: str) -> str:
    """Filesystem-safe slug for a fund name; the stem of its replay CSV."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "fund"


# Get portfolio CIVs
def fetch_portfolio_civs(portfolio, replay_from=None, save_replay=None):
    """Return ``{fund_name: NAV DataFrame}`` for the portfolio's funds.

    Network by default (mfapi.in). With ``replay_from`` set, each fund's
    NAV history is read from ``<replay_from>/navs/<slug>.csv`` instead, so
    a run is fully offline; combined with ``--as-of`` it is deterministic.
    With ``save_replay`` set, each fetched series is written there so the
    fixtures can be regenerated.
    """
    portfolio_civs = {}
    for fund in portfolio["funds"]:
        name = fund["name"]
        if replay_from is not None:
            path = os.path.join(replay_from, "navs", f"{_fund_slug(name)}.csv")
            df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        else:
            df = fetch_navs_of_mutual_fund(fund["url"])
        if save_replay is not None:
            navs_dir = os.path.join(save_replay, "navs")
            os.makedirs(navs_dir, exist_ok=True)
            df.to_csv(os.path.join(navs_dir, f"{_fund_slug(name)}.csv"))
        portfolio_civs[name] = df
    return portfolio_civs


# Align and combine CIV data
def align_portfolio_civs(portfolio_civs):
    """
    Align and combine CIV data for all funds to a common date range.

    Parameters:
        portfolio_civs (dict): Dictionary of fund names to DataFrames containing CIV data.

    Returns:
        pd.DataFrame: Combined CIV data aligned to a common date range.
    """
    # Determine the overlapping date range for all funds
    common_start_date = max(civ.index.min() for civ in portfolio_civs.values())
    common_end_date = min(civ.index.max() for civ in portfolio_civs.values())

    # Align each fund's CIV data to the common date range
    aligned_civs = {
        name: civ.loc[common_start_date:common_end_date]
        for name, civ in portfolio_civs.items()
    }

    # Combine aligned CIV data into a single DataFrame.
    # Starts with a dictionary and ends with a DataFrame.
    combined_civs = pd.concat({name: civ for name, civ in aligned_civs.items()}, axis=1)
    aligned_combined_civs = combined_civs.ffill()
    return aligned_combined_civs
def get_benchmark_gain_daily(benchmark_data):
    """
    Get usefully indexed benchmark historical NAVs.

    Parameters:
        benchmark_data: pd.DataFrame containing historical data indexed by date.

    Returns:
        pd.Series: Benchmark daily returns indexed by date.
    """
    # Ensure the index (dates) is treated as a datetime column
    benchmark_data.value_series().index = pd.to_datetime(benchmark_data.value_series().index, errors="coerce").tz_localize(None)
    # Assign the index name to "date"
    benchmark_data.value_series().index.name = "date"
    # Calculate daily returns
    benchmark_gain_daily = benchmark_data.value_series().pct_change().fillna(0)
    # Set the index name to "date" so it matches expected_result
    benchmark_gain_daily.index.name = "date"
    return benchmark_gain_daily


# Backward-compatibility re-export. Implementation lives in scss_loader.
from portfolioanalyzer.loaders.scss import load_scss_interest_rates  # noqa: E402, F401


# Load the TOML file
def load_portfolio_details(toml_file_path):
    """
    Load portfolio details from a TOML file with validation.

    Parameters:
        toml_file_path (str): Path to the TOML file.

    Returns:
        dict: Parsed portfolio details.

    Raises:
        ValueError: If the TOML data is invalid or missing required fields.
        FileNotFoundError: If the TOML file does not exist.
    """
    try:
        portfolio_details = load_portfolio_toml(toml_file_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {toml_file_path}") from e
    except toml.TomlDecodeError as e:
        raise ValueError(f"Invalid TOML format: {e}") from e

    # Validate the top-level keys
    if "label" not in portfolio_details:
        raise ValueError("Missing required top-level key: 'label'")
    
    # Check that at least one asset exists (including new asset types)
    valid_asset_keys = ["funds", "ppf", "gold", "sgb", "scss"]
    if not any(key in portfolio_details for key in valid_asset_keys):
        raise ValueError("TOML file specifies no assets")
    
    errors = []

    # Validate funds if present
    if "funds" in portfolio_details:
        if not isinstance(portfolio_details["funds"], list):
            errors.append("'funds' must be a list")
        else:
            for i, fund in enumerate(portfolio_details["funds"], start=1):
                fund_id = fund.get("name", f"fund #{i}")
                required_fund_keys = ["name", "url", "allocation", "asset_allocation"]
                for key in required_fund_keys:
                    if key not in fund:
                        errors.append(f"Missing required key '{key}' in investment '{fund_id}'")
                if "allocation" in fund and (
                    not isinstance(fund["allocation"], (float, int))
                    or not (0 <= fund["allocation"] <= 1)
                ):
                    errors.append(
                        f"Invalid allocation value for investment '{fund_id}': "
                        "Must be between 0 and 1"
                    )
                if "asset_allocation" in fund:
                    if not isinstance(fund["asset_allocation"], dict):
                        errors.append(f"'asset_allocation' must be a dictionary for investment '{fund_id}'")
                    else:
                        required_asset_keys = ["equity", "debt", "real_estate", "commodities", "cash"]
                        for key in required_asset_keys:
                            if key not in fund["asset_allocation"]:
                                errors.append(f"Missing key in 'asset_allocation' for investment '{fund_id}': '{key}'")
                            else:
                                value = fund["asset_allocation"][key]
                                if not isinstance(value, (float, int)) or value < 0:
                                    errors.append(f"Invalid value for '{key}' in 'asset_allocation' of investment '{fund_id}': Must be a non-negative number")

    # Validate PPF if present
    if "ppf" in portfolio_details:
        ppf = portfolio_details["ppf"]
        ppf_id = ppf.get("name", "PPF section")
        if "allocation" not in ppf:
            errors.append(f"Missing required key 'allocation' in {ppf_id}")
        else:
            if not isinstance(ppf["allocation"], (float, int)) or not (0 <= ppf["allocation"] <= 1):
                errors.append(f"Invalid allocation value for {ppf_id}: Must be between 0 and 1")

    # Validate Gold if present
    if "gold" in portfolio_details:
        gold = portfolio_details["gold"]
        gold_id = gold.get("name", "Gold section")
        if "allocation" not in gold:
            errors.append(f"Missing required key 'allocation' in {gold_id}")
        else:
            if not isinstance(gold["allocation"], (float, int)) or not (0 <= gold["allocation"] <= 1):
                errors.append(f"Invalid allocation value for {gold_id}: Must be between 0 and 1")
    
    # Validate SGB if present. Phase 2 of the SGB modeling refactor
    # requires the [[sgb]] *list* schema — each tranche is a distinct
    # investment with its own tranche_id, units_grams, and allocation.
    # The legacy [sgb] *dict* form (single allocation, no tranche info)
    # is rejected with a clear migration message.
    if "sgb" in portfolio_details:
        sgb_section = portfolio_details["sgb"]
        if isinstance(sgb_section, dict):
            errors.append(
                "Legacy [sgb] dict schema is no longer supported. "
                "Replace with one or more [[sgb]] entries, each with "
                "tranche_id, units_grams, and allocation."
            )
        elif isinstance(sgb_section, list):
            for i, entry in enumerate(sgb_section, start=1):
                entry_id = entry.get("tranche_id", f"sgb #{i}")
                if "tranche_id" not in entry:
                    errors.append(f"Missing required key 'tranche_id' in [[sgb]] #{i}")
                if "units_grams" not in entry:
                    errors.append(f"Missing required key 'units_grams' in [[sgb]] {entry_id}")
                else:
                    units = entry["units_grams"]
                    if not isinstance(units, (float, int)) or units <= 0:
                        errors.append(
                            f"Invalid units_grams for [[sgb]] {entry_id}: must be a positive number"
                        )
                if "allocation" not in entry:
                    errors.append(f"Missing required key 'allocation' in [[sgb]] {entry_id}")
                else:
                    alloc = entry["allocation"]
                    if not isinstance(alloc, (float, int)) or not (0 <= alloc <= 1):
                        errors.append(
                            f"Invalid allocation for [[sgb]] {entry_id}: must be between 0 and 1"
                        )
    
    # Validate SCSS if present
    if "scss" in portfolio_details:
        scss = portfolio_details["scss"]
        scss_id = scss.get("name", "SCSS section")
        if "allocation" not in scss:
            errors.append(f"Missing required key 'allocation' in {scss_id}")
        else:
            if not isinstance(scss["allocation"], (float, int)) or not (0 <= scss["allocation"] <= 1):
                errors.append(f"Invalid allocation value for {scss_id}: Must be between 0 and 1")
        # Optional term-lock keys (fail-fast on bad values, no silent fallback).
        if "purchase_date" in scss:
            try:
                pd.Timestamp(scss["purchase_date"])
            except (ValueError, TypeError):
                errors.append(
                    f"Invalid purchase_date for {scss_id}: must be a parseable date"
                )
        if "term_years" in scss:
            term = scss["term_years"]
            if not isinstance(term, (float, int)) or isinstance(term, bool) or term <= 0:
                errors.append(
                    f"Invalid term_years for {scss_id}: must be a positive number"
                )

    if errors:
        all_errors = "\n".join(errors)
        raise ValueError("TOML file errors detected:\n" + all_errors)

    # Validate total allocation (this function remains unchanged)
    validate_allocations(portfolio_details)

    return portfolio_details


def extract_weights(portfolio_dict):
    """
    Extracts a dictionary of asset weights from the portfolio_dict,
    covering mutual funds and other assets like Gold, PPF, etc.
    Raises ValueError if any listed asset lacks a valid allocation.
    """
    weights = {}

    # Helper to extract and validate one asset's allocation
    def add_weight(key, label):
        entry = portfolio_dict.get(key)
        if entry is None or "allocation" not in entry:
            raise ValueError(f"❌ Missing 'allocation' for {label} in portfolio TOML.")
        weights[label] = entry["allocation"]

    if "funds" in portfolio_dict:
        for fund in portfolio_dict["funds"]:
            if "name" not in fund or "allocation" not in fund:
                raise ValueError("❌ Each fund must have 'name' and 'allocation'")
            weights[fund["name"]] = fund["allocation"]

    for key, label in [
        ("gold", "Gold"),
        ("ppf", "PPF"),
        ("scss", "SCSS"),
    ]:
        if key in portfolio_dict:
            add_weight(key, label)

    # SGB: one weight entry per [[sgb]] tranche. Keyed by tranche_id
    # so the same key drops straight into PortfolioTimeseries.assets.
    if "sgb" in portfolio_dict:
        for entry in portfolio_dict["sgb"]:
            if "tranche_id" not in entry or "allocation" not in entry:
                raise ValueError(
                    "❌ Each [[sgb]] entry must have 'tranche_id' and 'allocation'"
                )
            weights[f"SGB {entry['tranche_id']}"] = entry["allocation"]

    return weights


def validate_allocations(portfolio_details, tol=0.01):
    total_allocation = 0
    if "funds" in portfolio_details:
        total_allocation += sum(fund["allocation"] for fund in portfolio_details["funds"])
    if "ppf" in portfolio_details:
        total_allocation += portfolio_details["ppf"].get("allocation", 0)
    if "gold" in portfolio_details:
        total_allocation += portfolio_details["gold"].get("allocation", 0)
    if "sgb" in portfolio_details:
        total_allocation += sum(
            entry.get("allocation", 0) for entry in portfolio_details["sgb"]
        )
    if "scss" in portfolio_details:
        total_allocation += portfolio_details["scss"].get("allocation", 0)
    if abs(total_allocation - 1.0) > tol:
        raise ValueError(f"Total allocation is {total_allocation:.4f}, but it must sum to 1.00 within tolerance {tol}.")


# Fetch NAV data
def fetch_navs_of_mutual_fund(url, retries=10, timeout=20):
    """Deprecated alias retained for backward compatibility.

    The implementation lives in ``mutual_fund_loader.fetch_navs``.
    """
    from portfolioanalyzer.loaders.mutual_fund import fetch_navs

    return fetch_navs(url, retries=retries, timeout=timeout)


# Backward-compatibility re-exports. Implementation lives in ppf_loader.
from portfolioanalyzer.loaders.ppf import load_ppf_civ, load_ppf_interest_rates  # noqa: E402, F401

# Load risk-free rate data
# Backward-compatibility re-exports. Implementations live in risk_free_loader.
from portfolioanalyzer.loaders.risk_free import (  # noqa: E402, F401
    align_dynamic_risk_free_rates,
    fetch_and_standardize_risk_free_rates,
)
