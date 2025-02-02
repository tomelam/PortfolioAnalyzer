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
# 2. Low-Level Functions:
#    - fetch_yahoo_finance_data
#    - massage_yahoo_data
#    - download_csv
#    - file_modified_within

import pandas as pd
import requests
import toml
import os
import yfinance as yf
from datetime import timedelta, datetime


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
    Get usefully indexed benchmark historical NAVs using Yahoo Finance.

    Parameters:
        benchmark_data: pd.DataFrame containing historical data indexed by date.
        refresh_hours (int): Number of hours before refreshing cached data (default: 6).
        period (str): Period to fetch data for (default: "max").

    Returns:
        pd.Series: Benchmark daily returns indexed by date.
    """
    # Ensure the index (dates) is treated as a datetime column
    benchmark_data.index = pd.to_datetime(
        benchmark_data.index, errors="coerce"
    ).tz_localize(None)
    if "Close" in benchmark_data.columns:
        benchmark_data.rename(columns={"Close": "Close"}, inplace=True)
    # Normalize to tz-naive
    benchmark_data["Date"] = benchmark_data.index  # Optional: Create a Date column

    # Calculate daily returns
    benchmark_gain_daily = benchmark_data["Close"].pct_change().dropna()
    return benchmark_gain_daily


# Function to download data from a given URL and save it as a CSV file
def download_csv(url, output_file):
    """
    Download a CSV file from a given URL and save it locally.

    Parameters:
        url (str): The URL to download the CSV file from.
        output_file (str): The path to save the downloaded CSV file.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(output_file, "wb") as file:
            file.write(response.content)
    except requests.RequestException as e:
        raise RequestException(f"Failed to download data from {url}: {e}")


# Check if the file was modified within a specific time frame
def file_modified_within(file_path, hours):
    """
    Check if a file was modified within the last specified hours.

    Parameters:
        file_path (str): Path to the file.
        hours (int): The time frame in hours.

    Returns:
        bool: True if the file was modified within the time frame, False otherwise.
    """
    if os.path.exists(file_path):
        modification_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        return (datetime.now() - modification_time).total_seconds() < hours * 3600
    return False


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
        # Load the TOML file
        portfolio_details = toml.load(toml_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {toml_file_path}")
    except toml.TomlDecodeError as e:
        raise ValueError(f"Invalid TOML format: {e}")

    # Validate the top-level keys
    required_top_level_keys = ["label", "funds"]
    for key in required_top_level_keys:
        if key not in portfolio_details:
            raise ValueError(f"Missing required top-level key: '{key}'")

    # Validate "funds" section
    if (
        not isinstance(portfolio_details["funds"], list)
        or len(portfolio_details["funds"]) == 0
    ):
        raise ValueError("'funds' must be a non-empty list")

    for fund in portfolio_details["funds"]:
        # Validate required keys in each fund
        required_fund_keys = ["name", "url", "allocation", "asset_allocation"]
        for key in required_fund_keys:
            if key not in fund:
                raise ValueError(f"Missing required key in fund: '{key}'")

        # Validate "allocation"
        if not isinstance(fund["allocation"], (float, int)) or not (
            0 <= fund["allocation"] <= 1
        ):
            raise ValueError(
                f"Invalid allocation value for fund '{fund.get('name', '<unknown>')}': Must be between 0 and 1"
            )

        # Validate "asset_allocation"
        if not isinstance(fund["asset_allocation"], dict):
            raise ValueError(
                f"'asset_allocation' must be a dictionary for fund '{fund.get('name', '<unknown>')}'"
            )

        required_asset_keys = ["equity", "debt", "real_estate", "commodities", "cash"]
        for key in required_asset_keys:
            if key not in fund["asset_allocation"]:
                raise ValueError(
                    f"Missing key in 'asset_allocation' for fund '{fund.get('name', '<unknown>')}': '{key}'"
                )
            if (
                not isinstance(fund["asset_allocation"][key], (float, int))
                or fund["asset_allocation"][key] < 0
            ):
                raise ValueError(
                    f"Invalid value for '{key}' in 'asset_allocation' of fund '{fund.get('name', '<unknown>')}': Must be a non-negative number"
                )

    return portfolio_details


# Parse the equity, debt, and cash allocations of the funds in a portfolio
def extract_fund_allocations(portfolio):
    """
    Extract individual fund allocations (equity, debt, cash) from a portfolio object.

    Parameters:
        portfolio (dict): The portfolio data loaded from the TOML file.

    Returns:
        list of dict: A list where each item represents a fund and its asset allocations.
    """
    fund_allocations = []
    for fund in portfolio["funds"]:
        fund_allocations.append(
            {
                "name": fund["name"],
                "allocation": fund["allocation"],
                "equity": fund["asset_allocation"]["equity"],
                "debt": fund["asset_allocation"]["debt"],
                "real_estate": fund["asset_allocation"]["real_estate"],
                "commodities": fund["asset_allocation"]["commodities"],
                "cash": fund["asset_allocation"]["cash"],
            }
        )
    return fund_allocations


# Fetch NAV data
def fetch_navs_of_mutual_fund(url, retries=10, timeout=20):
    """
    Fetch NAV data for a fund from the given API URL.

    Parameters:
        url (str): API endpoint for fetching NAV data.
        retries (int): Number of retry attempts on failure.
        timeout (int): Request timeout duration in seconds.

    Returns:
        pd.DataFrame: DataFrame containing NAV data indexed by date.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if "data" not in data or not data["data"]:
                raise KeyError(
                    f"'data' key missing or empty in API response from {url}"
                )
            nav_data = pd.DataFrame(data["data"])
            nav_data["date"] = pd.to_datetime(
                nav_data["date"], dayfirst=True, errors="coerce"
            )
            nav_data["nav"] = nav_data["nav"].astype(float)
            return nav_data.set_index("date").sort_index()
        except requests.RequestException as e:
            print(
                f"[Error] Request failed for {url} (Attempt {attempt + 1}/{retries}): {e}"
            )
        except (ValueError, KeyError) as e:
            print(
                f"[Error] Data processing error for {url} (Attempt {attempt + 1}/{retries}): {e}"
            )
    raise RuntimeError(f"Failed to fetch NAV data from {url} after {retries} retries")


# Load the PPF interest rates from a CSV file
def load_ppf_interest_rates(csv_file_path="ppf_interest_rates.csv"):
    """
    Parameters:
        csv_file_path (str): Path to the CSV file containing PPF interest rates.

    Returns:
        pd.DataFrame: A DataFrame with columns "rate" and "date", indexed by "date" and sorted.
    """
    try:
        if csv_file_path is None:
            csv_file_path = "ppf_interest_rates.csv"

        # Read the CSV file into a DataFrame
        ppf_data = pd.read_csv(csv_file_path)

        # Ensure the required columns are present
        if "date" not in ppf_data.columns or "rate" not in ppf_data.columns:
            raise ValueError("CSV file must contain 'date' and 'rate' columns.")

        # Convert "date" to datetime and set it as the index
        ppf_data["date"] = pd.to_datetime(
            ppf_data["date"], format="%Y-%m-%d", errors="coerce"
        )
        ppf_data.dropna(subset=["date"], inplace=True)  # Drop rows with invalid dates
        ppf_data.set_index("date", inplace=True)

        # Ensure "rate" column is numeric
        ppf_data["rate"] = pd.to_numeric(ppf_data["rate"], errors="coerce")
        ppf_data.dropna(subset=["rate"], inplace=True)  # Drop rows with invalid rates

        # Sort the DataFrame by the date index
        ppf_data.sort_index(inplace=True)

        return ppf_data

    except Exception as e:
        raise FileNotFoundError


# TODO: Replace the word "relative" with "normalized" or "gain" in the variable name.


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


# Fetch Yahoo Finance data
def fetch_yahoo_finance_data(ticker, refresh_hours=6, period="max"):
    """
    Fetch historical data from Yahoo Finance for a given ticker symbol.

    Parameters:
        ticker (str): Yahoo Finance ticker symbol.
        period (str): The time period for historical data (default is "max").
        refresh_hours (int): Number of hours before refreshing the data (default is 6).

    Returns:
        pd.DataFrame: Historical data indexed by date.
    """

    file_path = f"{ticker.replace('^', '').replace('/', '_')}.csv"
    if not file_modified_within(file_path, refresh_hours):
        print(
            f"The data for {ticker} is outdated or missing. Fetching it using yfinance..."
        )
        yf_ticker = yf.Ticker(ticker)
        raw = yf_ticker.history(period=period)
        raw.to_csv(file_path)
        print(f"Data downloaded successfully and saved to {file_path}.")
    else:
        raw = pd.read_csv(file_path)

    data = rename_yahoo_data_columns(raw)
    """
    # Convert the index to a datetime column
    data["date"] = pd.to_datetime(data.index, errors="coerce")
    data.reset_index(
        drop=True, inplace=True
    )  # Drop the old index    if "Date" not in data.columns:
    """
    assert data.index.name == "Date", "Index name mismatch: expected 'Date'"
    return data


def rename_yahoo_data_columns(data):
    """
    Ensure proper renaming of columns and consistent index setting.
    """
    if "Datetime" in data.columns:
        data.rename(columns={"Datetime": "Date"}, inplace=True)
    elif "date" in data.columns:
        data.rename(columns={"date": "Date"}, inplace=True)

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data.set_index("Date", inplace=True)

    # Explicitly set the index name to "Date"
    if data.index.name != "Date":
        data.index.name = "Date"
        print(f"Index name corrected to 'Date': {data.index.name}")
    return data


# Load risk-free rate data
def fetch_and_standardize_risk_free_rates(file_path, url=None):
    """
    Load risk-free rate data from a CSV file, downloading it if a URL is provided.

    Parameters:
        file_path (str): Path to the CSV file.
        url (str, optional): URL to download the CSV file from. If provided, the file is downloaded first.

    Returns:
        pd.DataFrame: Risk-free rate data indexed by date.
    """
    if url and not file_modified_within(file_path, 24):
        print(
            "The risk-free-rate data is outdated or missing. Fetching it from the web..."
        )
        download_csv(url, file_path)
    else:
        print("The risk-free-rate data is fresh. No need to download it again now.")

    risk_free_data = pd.read_csv(file_path)
    risk_free_data.rename(
        columns={"observation_date": "date", "INDIRLTLT01STM": "rate"}, inplace=True
    )
    risk_free_data["date"] = pd.to_datetime(
        risk_free_data["date"], format="%Y-%m-%d", errors="coerce"
    )
    risk_free_data["rate"] = (
        risk_free_data["rate"].astype(float) / 100
    )  # Convert to annualized decimal
    risk_free_data.set_index("date", inplace=True)
    return risk_free_data


# Interpolate risk-free rates to match portfolio dates
def align_dynamic_risk_free_rates(portfolio_returns, risk_free_data):
    """
    Align and interpolate risk-free rates to match portfolio return dates.

    Parameters:
        portfolio_returns (pd.Series): Portfolio daily returns.
        risk_free_data (pd.DataFrame): Risk-free rate data.

    Returns:
        pd.Series: Interpolated risk-free rates.
    """
    aligned_rates = risk_free_data.reindex(portfolio_returns.index).interpolate(
        method="time"
    )
    filled_rates = aligned_rates["rate"].ffill().bfill()
    # return filled_rates.mean()
    return filled_rates
