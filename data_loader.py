# Data Loader Module
# This module provides tools for loading and processing data related to
# financial portfolios, including functions to fetch NAV data, risk-free rates,
# and benchmark indices, align data to common ranges, and calculate fund
# allocations.
#
# Function Table
# 1. Top-Level Functions:
#    - get_aligned_portfolio_civs
#    - align_portfolio_civs
#    - get_benchmark_navs
#    - load_portfolio_details
#    - fetch_and_standardize_risk_free_rates
#    - align_dynamic_risk_free_rates
#    - extract_fund_allocations
#    - fetch_navs_of_mutual_fund
#    - load_ppf_interest_rates
#    - extract_fom_values
#    - align_fund_data
#    - get_dynamic_risk_free_rate

import os

import pandas as pd
import toml

from utils import info

# DEBUG flag is set by main.py (‑d/‑‑debug). Fallback = False.
try:
    from main import DEBUG
except ImportError:
    DEBUG = False  # fallback default if main hasn't been run yet


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


def check_time_index_cleanliness(df, name="DataFrame"):
    import pandas as pd


    idx = df.index
    problems = []

    if not isinstance(idx, pd.DatetimeIndex):
        problems.append("Index is not a DatetimeIndex — possibly string-based. Was date_format applied correctly?")

    if idx.hasnans:
        problems.append("Index contains NaT values — likely due to failed date parsing.")

    if not idx.is_monotonic_increasing:
        problems.append("Index is not sorted.")

    if not idx.is_unique:
        problems.append("Index contains duplicate timestamps.")

    if problems:
        info(f"⚠️  {name} time index issues detected:")
        for p in problems:
            info(f"   - {p}")
        # Show sample of problematic index
        sample = list(idx[:5])
        info(f"   → First few index values: {sample}")
    else:
        info(f"{name} time index appears clean.")


# Backward-compatibility re-export. Implementation lives in benchmark_loader.
from benchmark_loader import load_timeseries_csv  # noqa: E402, F401


def get_aligned_portfolio_civs(portfolio):
    """
    Load and align the CIVs from each fund in a portfolio.

    Parameters:
        portfolio: portfolio allocations to each component fund and
            each fund's asset allocations

    Returns:
        pd.DataFrame: Aligned NAV data for all funds in the portfolio.
    """

    portfolio_civs = fetch_portfolio_civs(portfolio)
    aligned_civs = align_portfolio_civs(portfolio_civs)
    # Flatten the MultiIndex columns by removing the second level ('nav')
    aligned_civs.columns = aligned_civs.columns.droplevel(1)
    return aligned_civs


# Get portfolio CIVs
def fetch_portfolio_civs(portfolio):
    portfolio_civs = {
        fund["name"]: fetch_navs_of_mutual_fund(fund["url"])
        for fund in portfolio["funds"]
    }
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
from scss_loader import load_scss_interest_rates  # noqa: E402, F401


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
    valid_asset_keys = ["funds", "ppf", "gold", "sgb", "scss", "rec_bond"]
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
    
    # Validate REC Bond if present
    if "rec_bond" in portfolio_details:
        rec = portfolio_details["rec_bond"]
        rec_id = rec.get("name", "REC Bond section")
        if "allocation" not in rec:
            errors.append(f"Missing required key 'allocation' in {rec_id}")
        elif not isinstance(rec["allocation"], (float, int)) or not (0 <= rec["allocation"] <= 1):
            errors.append(f"Invalid allocation value for {rec_id}: Must be between 0 and 1")
        # Optionally, validate coupon if provided.
        if "coupon" in rec and (
            not isinstance(rec["coupon"], (float, int)) or rec["coupon"] <= 0
        ):
            errors.append(f"Invalid coupon value for {rec_id}: Must be a positive number")
    
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
        ("rec_bond", "REC"),
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
    if "rec_bond" in portfolio_details:
        total_allocation += portfolio_details["rec_bond"].get("allocation", 0)
    if abs(total_allocation - 1.0) > tol:
        raise ValueError(f"Total allocation is {total_allocation:.4f}, but it must sum to 1.00 within tolerance {tol}.")


# Fetch NAV data
def fetch_navs_of_mutual_fund(url, retries=10, timeout=20):
    """Deprecated alias retained for backward compatibility.

    The implementation lives in ``mutual_fund_loader.fetch_navs``.
    """
    from mutual_fund_loader import fetch_navs

    return fetch_navs(url, retries=retries, timeout=timeout)


# Backward-compatibility re-exports. Implementation lives in ppf_loader.
from ppf_loader import load_ppf_civ, load_ppf_interest_rates  # noqa: E402, F401


def calculate_gold_cumulative_gain(gold_data, portfolio_start_date):
    """
    Compute a relative cumulative gain series from a gold price series.
    Assumes that at the portfolio start date, the relative value is 1.0.
    """
    # Restrict to dates on/after portfolio start.
    gold_data = gold_data.loc[gold_data.index >= portfolio_start_date]
    if gold_data.empty:
        raise ValueError("No gold price data available after portfolio start date.")

    # Normalize: divide by the price on portfolio_start_date.
    base_price = gold_data.iloc[0]["price"]
    gold_data = gold_data.copy()
    gold_data["gold"] = gold_data["price"] / base_price  # Should be 1.0

    # Reindex to daily frequency.
    gold_data = gold_data.asfreq("D", method="ffill")  # Ensure no missing dates.

    return gold_data[["gold"]]


# Extract first-of-the-month (FOM) values
def extract_fom_values(nav_data):
    """
    Extract the first-of-the-month (FOM) values from CIV data.

    Parameters:
        nav_data (pd.DataFrame): DataFrame containing CIV data indexed by date.

    Returns:
        pd.DataFrame: FOM values.
    """
    fom_values = nav_data.loc[nav_data.index.is_month_start]
    return fom_values


# Load risk-free rate data
# Backward-compatibility re-exports. Implementations live in risk_free_loader.
from risk_free_loader import (  # noqa: E402, F401
    align_dynamic_risk_free_rates,
    fetch_and_standardize_risk_free_rates,
)
